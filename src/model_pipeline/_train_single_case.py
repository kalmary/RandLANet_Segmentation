import copy
import pathlib as pth
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from _data_loader import make_loader

from RandLANet_CB import RandLANet

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import FocalLoss
from utils import wrap_hist

from tqdm import tqdm
from typing import Union, Generator


def compute_pos_weights_cloud(
    data_dir,
    num_classes: int,
    power: float = 0.25,
) -> torch.Tensor:
    """Compute class weights while allowing -1 context-only points."""
    paths = sorted(pth.Path(data_dir).glob("*.npy"))
    if not pth.Path(data_dir).exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    if not paths:
        raise FileNotFoundError(f"No .npy point-cloud tiles in {data_dir}")

    counts = np.zeros(num_classes, dtype=np.int64)
    for path in paths:
        data = np.load(path, mmap_mode="r")
        if data.ndim != 2 or data.shape[1] != 5 or len(data) == 0:
            raise ValueError(f"Expected non-empty (N, 5) data in {path}")
        values = np.asarray(data[:, 4])
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"Invalid labels in {path}")
        labels = values.astype(np.int64)
        if np.any((labels < -1) | (labels >= num_classes)):
            raise ValueError(f"Label outside configured classes in {path}")
        labels = labels[labels >= 0]
        counts += np.bincount(labels, minlength=num_classes)

    weights = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    if weights.max() == 0:
        raise ValueError(f"No labeled points in {data_dir}")
    weights /= weights.max()
    return torch.from_numpy(weights.astype(np.float32))


def _labeled_outputs(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten logits and remove context-only targets marked with -1."""
    flattened_outputs = outputs.movedim(1, -1).reshape(-1, outputs.shape[1])
    flattened_labels = labels.reshape(-1)
    valid = flattened_labels >= 0
    if not torch.any(valid):
        raise RuntimeError("Sampled crop contains no labeled points")
    return flattened_outputs[valid], flattened_labels[valid]


def _configured_epoch_lengths(training_dict: dict) -> tuple[int, int]:
    totals = (
        training_dict.get('train_steps_per_epoch', 500),
        training_dict.get('validation_steps', 100),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in totals
    ):
        raise ValueError(
            "train_steps_per_epoch and validation_steps must be positive integers"
        )
    return totals


def _build_scheduler(optimizer, training_dict: dict, total_t: int | None):
    if total_t is None:
        return ReduceLROnPlateau(optimizer, mode="min")

    return OneCycleLR(
        optimizer,
        max_lr=training_dict['learning_rate'],
        total_steps=total_t * training_dict['epochs'],
        pct_start=training_dict['pc_start'],
        anneal_strategy='cos',
        div_factor=training_dict['div_factor'],
        final_div_factor=training_dict['final_div_factor'],
        cycle_momentum=True,
    )


def train_model(training_dict: dict) -> Union[Generator[tuple[nn.Module, dict], None, None],
                                              Generator[tuple[None, dict], None, None]]:
    device_gpu = torch.device('cuda')
    device_loss = device_gpu

    try:

        weight_power = training_dict.get('class_weight_power', 0.5)
        class_weights_t = compute_pos_weights_cloud(data_dir=training_dict['data_path_train'],
                                                num_classes=training_dict['num_classes'],
                                                power=weight_power)

        
        class_weights_v = compute_pos_weights_cloud(data_dir=training_dict['data_path_val'],
                                                num_classes=training_dict['num_classes'],
                                                power=weight_power
        )


        total_t, total_v = _configured_epoch_lengths(training_dict)
        batches_per_file = training_dict.get('batches_per_file', 8)
        class_aware_fraction = training_dict.get(
            'class_aware_center_fraction', 0.5
        )

        trainLoader, _ = make_loader(
            data_dir=training_dict['data_path_train'],
            num_points=training_dict['num_points'],
            batch_size=training_dict['batch_size'],
            query_workers=training_dict.get('query_workers', 7),
            shuffle=True,
            class_weights=class_weights_t.numpy(),
            max_seen=training_dict.get('max_seen', 3),
            steps_per_epoch=total_t,
            batches_per_file=batches_per_file,
            class_aware_fraction=class_aware_fraction,
        )

        valLoader, _ = make_loader(
            data_dir=training_dict['data_path_val'],
            num_points=training_dict['num_points'],
            batch_size=training_dict['batch_size'],
            query_workers=training_dict.get('query_workers', 7),
            shuffle=False,
            max_seen=1,
            steps_per_epoch=total_v,
            batches_per_file=batches_per_file,
            class_aware_fraction=0.0,
        )

        repeat_pbar = tqdm(range(training_dict['train_repeat']), 
                            desc="Training Repetition", 
                            unit="repeat",
                            position=1,
                            leave=False) 

        for _ in repeat_pbar:
            if training_dict['model'] is None:
                model = RandLANet(model_config=training_dict['model_config'],
                                n_classes=training_dict['num_classes'])
            else:
                model = copy.deepcopy(training_dict['model'])

            model.to(training_dict['device'])

            criterion_t = FocalLoss(
                alpha=class_weights_t.to(device_loss),
                gamma=training_dict['focal_loss_gamma'],
                smoothing=0.1,
                reduction='mean',
            ).to(device_loss)

            criterion_v = FocalLoss(
                alpha=class_weights_v.to(device_loss),
                gamma=training_dict['focal_loss_gamma'],
                smoothing=0.1,
                reduction='mean',
            ).to(device_loss)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=training_dict['learning_rate'],
                weight_decay=training_dict['weight_decay'],
            )
            scheduler = _build_scheduler(optimizer, training_dict, total_t)

            loss_hist = []
            acc_hist = []
            miou_hist = []

            loss_v_hist = []
            acc_v_hist = []
            miou_v_hist = []

            epoch_pbar = tqdm(range(training_dict['epochs']), 
                                desc="Epoch Progress", 
                                unit="epoch",
                                position=2, 
                                leave=False) 

            for epoch in epoch_pbar:

                epoch_loss_t = 0.
                epoch_loss_v = 0.

                epoch_samples_t = 0
                epoch_samples_v = 0

                confusion_v = torch.zeros(
                    (training_dict['num_classes'], training_dict['num_classes']),
                    dtype=torch.int64,
                )


                progressbar_t = tqdm(trainLoader, 
                                        desc=f"Epoch training {epoch+1}/ {training_dict['epochs']}", 
                                        total=total_t, 
                                        position=3,
                                        leave=False)
                
                model.train(True)
                
                for batch_x, batch_y in progressbar_t:
                    
                    batch_x = batch_x.to(training_dict['device'])
                    outputs = model(batch_x)

                    outputs = outputs.to(device_loss)
                    batch_y = batch_y.to(device_loss)

                    labeled_outputs, labeled_targets = _labeled_outputs(
                        outputs, batch_y
                    )
                    loss_t = criterion_t(labeled_outputs, labeled_targets)

                    optimizer.zero_grad()
                    loss_t.backward()
                    optimizer.step()

                    if (
                        isinstance(scheduler, OneCycleLR)
                        and scheduler.last_epoch < scheduler.total_steps
                    ):
                        scheduler.step()

                    current_lr = optimizer.param_groups[0]['lr']

                    labeled_count = labeled_targets.numel()
                    epoch_loss_t += loss_t.item() * labeled_count
                    # epoch_accuracy_t += accuracy_t * batch_y.size(0)
                    epoch_samples_t += labeled_count

                    avg_loss_t = epoch_loss_t / epoch_samples_t
                    # avg_accuracy_t = epoch_accuracy_t / epoch_samples_t

                    progressbar_t.set_postfix({
                        "Loss_train": f"{avg_loss_t:.6f}",
                        # "Acc_train": f"{avg_accuracy_t:.6f}",
                        "learning_rate": f"{current_lr:.10f}"
                    })

                if epoch_samples_t == 0:
                    raise RuntimeError("Training dataset produced no batches")

                loss_hist.append(avg_loss_t)
                acc_hist.append(-1.)  # Not computed for training
                miou_hist.append(-1.)  # Not computed for training

                progressbar_v = tqdm(valLoader, desc=f"Epoch validation {epoch + 1}/ {training_dict['epochs']}", 
                                    position=3, total = total_v, leave=False)
                model.eval()

                with torch.no_grad():
                    for batch_x, batch_y in progressbar_v:

                        batch_x = batch_x.to(training_dict['device'])
                        outputs = model(batch_x)

                        outputs = outputs.to(device_loss)
                        batch_y = batch_y.to(device_loss)

                        labeled_outputs, labeled_targets = _labeled_outputs(
                            outputs, batch_y
                        )
                        loss_v = criterion_v(labeled_outputs, labeled_targets)

                        predictions = labeled_outputs.argmax(dim=1)
                        target_cpu = labeled_targets.cpu()
                        prediction_cpu = predictions.cpu()
                        encoded = (
                            target_cpu * training_dict['num_classes']
                            + prediction_cpu
                        )
                        confusion_v += torch.bincount(
                            encoded,
                            minlength=training_dict['num_classes'] ** 2,
                        ).reshape(
                            training_dict['num_classes'],
                            training_dict['num_classes'],
                        )

                        point_count = labeled_targets.numel()
                        epoch_loss_v += loss_v.item() * point_count
                        epoch_samples_v += point_count

                        avg_loss_v = epoch_loss_v / epoch_samples_v
                        avg_accuracy_v = (
                            confusion_v.diag().sum().item()
                            / confusion_v.sum().item()
                        )
                        ground_truth = confusion_v.sum(dim=1)
                        predictions_per_class = confusion_v.sum(dim=0)
                        union = ground_truth + predictions_per_class - confusion_v.diag()
                        valid = ground_truth > 0
                        avg_miou_v = (
                            confusion_v.diag()[valid].float()
                            / union[valid].clamp_min(1).float()
                        ).mean().item()

                        progressbar_v.set_postfix({
                            "Loss_val": f"{avg_loss_v:.6f}",
                            "Acc_val": f"{avg_accuracy_v:.6f}",
                            "mIoU_val": f"{avg_miou_v:.6f}"
                        })

                if epoch_samples_v == 0:
                    raise RuntimeError("Validation dataset produced no batches")
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_loss_v)

                loss_v_hist.append(avg_loss_v)
                acc_v_hist.append(avg_accuracy_v)
                miou_v_hist.append(avg_miou_v)

                # early_stopping.check_early_stop(loss_v_hist[-1])

                hist_dict = wrap_hist(acc_hist = acc_hist,
                                        loss_hist = loss_hist,
                                        miou_hist = miou_hist,
                                        acc_v_hist = acc_v_hist,
                                        loss_v_hist = loss_v_hist,
                                        miou_v_hist = miou_v_hist)

                yield model, hist_dict


                epoch_pbar.set_postfix({
                    "Loss_train": f"{avg_loss_t:.6f}",
                    "Loss_val": f"{avg_loss_v:.6f}",
                    "Acc_val": f"{avg_accuracy_v:.6f}",
                    "learning_rate_max": f"{training_dict['learning_rate']:.10f}"
                })

    except Exception:
        if "model" in locals():
            del model
        torch.cuda.empty_cache()
        raise
