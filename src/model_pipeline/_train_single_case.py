import torch
from torchinfo import summary
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from RandLANet_CB import RandLANet

from _data_loader import *

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import compute_mIoU, calculate_weighted_accuracy
from utils import calculate_class_weights, get_dataset_len, FocalLoss_ArcFace, FocalLoss
from utils import wrap_hist

from tqdm import tqdm
from typing import Union, Generator

def train_model(training_dict: dict) -> Union[Generator[tuple[nn.Module, dict], None, None],
                                              Generator[tuple[None, dict], None, None]]:
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_gpu

    train_dataset = Dataset(base_dir=training_dict['data_path_train'],
                                      num_points=training_dict['num_points'],
                                      batch_size=training_dict['batch_size'],
                                      shuffle=True,
                                      device=device_loader)

    trainLoader = DataLoader(train_dataset,
                             batch_size=None,
                             num_workers = 10,
                             pin_memory=True)
    



    val_dataset = Dataset(base_dir=training_dict['data_path_val'],
                                    num_points=training_dict['num_points'],
                                    batch_size=training_dict['batch_size'],
                                    shuffle=False,
                                    device=device_loader)

    valLoader = DataLoader(val_dataset,
                             batch_size=None,
                             num_workers = 10,
                             pin_memory=True)
    try:

        total_t = get_dataset_len(trainLoader)
        total_v = get_dataset_len(valLoader)

        class_weights_t = calculate_class_weights(loader = trainLoader, 
                                                num_classes=training_dict['num_classes'], 
                                                method='effective',
                                                total=total_t, 
                                                device=device_loss,
                                                verbose=False)
        
        class_weights_v = calculate_class_weights(loader = valLoader, 
                                                num_classes=training_dict['num_classes'], 
                                                method='effective',
                                                total=total_v, 
                                                device=device_loss,
                                                verbose=False)
        
        train_dataset = Dataset(base_dir=training_dict['data_path_train'],
                                    num_points=training_dict['num_points'],
                                    batch_size=training_dict['batch_size'],
                                    shuffle=True,
                                    weights=class_weights_t,
                                    device=device_loader)

        trainLoader = DataLoader(train_dataset,
                                batch_size=None,
                                num_workers = 10,
                                pin_memory=True)
        



        val_dataset = Dataset(base_dir=training_dict['data_path_val'],
                                        num_points=training_dict['num_points'],
                                        batch_size=training_dict['batch_size'],
                                        shuffle=False,
                                        weights=class_weights_v,
                                        device=device_loader)

        valLoader = DataLoader(val_dataset,
                                batch_size=None,
                                num_workers = 10,
                                pin_memory=True)



        if training_dict['model'] is None:
            model = RandLANet(model_config=training_dict['model_config'],
                            num_classes=training_dict['num_classes'])
        else:
            model = training_dict['model']

        model.to(training_dict['device'])


        # criterion_t = FocalLoss_ArcFace(alpha=class_weights_t.to(device_loss),
        #                                       gamma=training_dict['focal_loss_gamma'],
        #                                       reduction='mean').to(device_loss) # TODO double check - Labels smoothing is good for better generalization, but exact impact must be investigated
        
        # criterion_v = FocalLoss_ArcFace(gamma=training_dict['focal_loss_gamma'],
        #                                       alpha=class_weights_v.to(device_loss),
        #                                       reduction='mean').to(device_loss)
        
        criterion_t = FocalLoss(alpha=class_weights_t.to(device_loss),
                                gamma=training_dict['focal_loss_gamma'],
                                smoothing=0.1,
                                reduction='mean').to(device_loss) # TODO double check - Labels smoothing is good for better generalization, but exact impact must be investigated
        
        criterion_v = FocalLoss(alpha=class_weights_v.to(device_loss),
                                gamma=training_dict['focal_loss_gamma'],
                                smoothing=0.1,
                                reduction='mean').to(device_loss)

        optimizer = optim.AdamW(model.parameters(), lr = training_dict['learning_rate'], weight_decay=training_dict['weight_decay'])

        scheduler = OneCycleLR(
            optimizer,
            max_lr=training_dict['learning_rate'],  # Example: Adjust based on your LR range test
            total_steps=total_t*training_dict['epochs'],
            pct_start=training_dict['pc_start'],  # % of steps for warm-up
            anneal_strategy='cos',  # Cosine annealing
            div_factor=training_dict['div_factor'],  # Initial LR will be max_lr / x
            final_div_factor=training_dict['final_div_factor'],  # Final LR
            cycle_momentum=True  # Cycle momentum as well
        )

        loss_hist = []
        acc_hist = []
        miou_hist = []

        loss_v_hist = []
        acc_v_hist = []
        miou_v_hist = []


        repeat_pbar = tqdm(range(training_dict['train_repeat']), 
                            desc="Training Repetition", 
                            unit="repeat",
                            position=1, 
                            leave=False) 

        for _ in repeat_pbar:

            epoch_pbar = tqdm(range(training_dict['epochs']), 
                                desc="Epoch Progress", 
                                unit="epoch",
                                position=2, 
                                leave=False) 

            for epoch in epoch_pbar:

                epoch_loss_t = 0.
                epoch_loss_v = 0.

                epoch_accuracy_t = 0.
                epoch_accuracy_v = 0.

                epoch_samples_t = 0
                epoch_samples_v = 0

                epoch_miou_v = 0.


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

                    loss_t = criterion_t(outputs, batch_y)

                    optimizer.zero_grad()
                    loss_t.backward()
                    optimizer.step()

                    try:
                        scheduler.step()
                    except Exception:
                        pass

                    # accuracy_t = calculate_weighted_accuracy(outputs, batch_y, weights=class_weights_t)

                    current_lr = optimizer.param_groups[0]['lr']

                    epoch_loss_t += loss_t.item() * batch_y.size(0)
                    # epoch_accuracy_t += accuracy_t * batch_y.size(0)
                    epoch_samples_t += batch_y.size(0)

                    avg_loss_t = epoch_loss_t / epoch_samples_t
                    # avg_accuracy_t = epoch_accuracy_t / epoch_samples_t

                    progressbar_t.set_postfix({
                        "Loss_train": f"{avg_loss_t:.6f}",
                        # "Acc_train": f"{avg_accuracy_t:.6f}",
                        "learning_rate": f"{current_lr:.10f}"
                    })

                loss_hist.append(avg_loss_t)
                acc_hist.append(-1.)  # Not computed for training
                miou_hist.append(-1.)  # Not computed for training

                progressbar_v = tqdm(valLoader, desc=f"Epoch validation {epoch + 1}/ {training_dict['epochs']}", total=total_v, position=3, leave=False)
                model.eval()

                with torch.no_grad():
                    for batch_x, batch_y in progressbar_v:

                        batch_x = batch_x.to(training_dict['device'])
                        outputs = model(batch_x)

                        outputs = outputs.to(device_loss)
                        batch_y = batch_y.to(device_loss)

                        loss_v = criterion_v(outputs, batch_y)

                        accuracy_v = calculate_weighted_accuracy(outputs, batch_y, weights=class_weights_v)

                        mIoU, _ = compute_mIoU(outputs, batch_y, training_dict['num_classes'])


                        epoch_loss_v += loss_v.item() * batch_y.size(0)
                        epoch_accuracy_v += accuracy_v * batch_y.size(0)
                        epoch_miou_v += mIoU * batch_y.size(0)
                        epoch_samples_v += batch_y.size(0)

                        avg_loss_v = epoch_loss_v / epoch_samples_v
                        avg_accuracy_v = epoch_accuracy_v / epoch_samples_v
                        avg_miou_v = epoch_miou_v / epoch_samples_v

                        progressbar_v.set_postfix({
                            "Loss_val": f"{avg_loss_v:.6f}",
                            "Acc_val": f"{avg_accuracy_v:.6f}",
                            "mIoU_val": f"{avg_miou_v:.6f}"
                        })

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

    except Exception as e:
        print(f"Error during training: {e}")
        try:
            del model
        except Exception as e:
            pass
        torch.cuda.empty_cache()
        yield None, {}