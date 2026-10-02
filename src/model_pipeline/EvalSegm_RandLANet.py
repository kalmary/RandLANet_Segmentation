import torch
import torch.nn as nn
import numpy as np
import laspy
from torch.utils.data import DataLoader
from torchinfo import summary

import argparse
from tqdm import tqdm
import pathlib as pth
from typing import Any, TypedDict



import sys

if __package__:
    from .RandLANet_CB import RandLANet
    from ._data_loader import Dataset
    from ..utils import (
        load_json, load_model, convert_str_values, get_dataset_len,
        compute_pos_weights_h5, compute_mIoU, FocalLoss, get_intLabels,
        get_Probabilities, Plotter, ClassificationReport,
    )
else:
    src_dir = pth.Path(__file__).parent.parent
    sys.path.append(str(src_dir))
    from RandLANet_CB import RandLANet
    from _data_loader import Dataset
    from utils import (
        load_json, load_model, convert_str_values, get_dataset_len,
        compute_pos_weights_h5, compute_mIoU, FocalLoss, get_intLabels,
        get_Probabilities, Plotter, ClassificationReport,
    )


class EvaluationMetrics(TypedDict):
    accuracy: float
    miou: float
    class_iou: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray


class CollectionSummary(TypedDict):
    sample_paths: list[pth.Path]
    processed_files: int
    labeled_files: int
    sampled_points: int


def _cloud_files(raw_path: pth.Path) -> list[pth.Path]:
    raw_path = pth.Path(raw_path)
    if raw_path.exists() and not raw_path.is_dir():
        raise NotADirectoryError(f'raw_path is not a directory: {raw_path}')
    files = sorted(
        path for path in raw_path.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.las', '.laz'}
    ) if raw_path.is_dir() else []
    if not files:
        raise FileNotFoundError(f'No LAS or LAZ files in {raw_path}')
    return files


def _device(name: str) -> torch.device:
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(name)


def _stratified_indices(
    targets: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    targets = np.asarray(targets).reshape(-1)
    if max_points <= 0:
        raise ValueError('max_points must be positive')
    if targets.size <= max_points:
        return np.arange(targets.size, dtype=np.int64)

    classes, counts = np.unique(targets, return_counts=True)
    quotas = np.zeros(len(classes), dtype=np.int64)
    remaining = max_points
    while remaining:
        active = np.flatnonzero(quotas < counts)
        share, extra = divmod(remaining, len(active))
        requested = np.full(len(active), share, dtype=np.int64)
        requested[:extra] += 1
        additions = np.minimum(requested, counts[active] - quotas[active])
        quotas[active] += additions
        remaining -= int(additions.sum())

    selected = [
        rng.choice(np.flatnonzero(targets == label), size=quota, replace=False)
        for label, quota in zip(classes, quotas)
        if quota
    ]
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> EvaluationMetrics:
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError('predictions and targets must have the same shape')
    if targets.size == 0:
        raise ValueError('predictions and targets cannot be empty')
    for name, values in [('predictions', predictions), ('targets', targets)]:
        if values.min() < 0 or values.max() >= num_classes:
            raise ValueError(f'{name} are outside the model class range')

    miou, class_iou = compute_mIoU(
        torch.from_numpy(predictions.astype(np.int64, copy=False)),
        torch.from_numpy(targets.astype(np.int64, copy=False)),
        num_classes,
    )
    return {
        'accuracy': float(np.mean(predictions == targets)),
        'miou': miou,
        'class_iou': class_iou.cpu().numpy(),
        'predictions': predictions,
        'targets': targets,
    }


def collect_samples(
    segmenter: Any,
    files: list[pth.Path],
    max_points: int,
    temp_dir: pth.Path,
    rng: np.random.Generator,
    verbose: bool = True,
) -> CollectionSummary:
    sample_paths: list[pth.Path] = []
    labeled_files = 0
    sampled_points = 0
    iterator = tqdm(files, desc='Evaluating files', unit='file') if verbose else files
    for file_index, file_path in enumerate(iterator):
        cloud = laspy.read(file_path)
        points = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)))
        intensity = np.asarray(cloud.intensity)
        stored_targets = np.asarray(cloud.classification, dtype=np.int64)
        if len(intensity) != len(points) or len(stored_targets) != len(points):
            raise ValueError(f'{file_path}: point attributes have mismatched lengths')

        predictions = np.asarray(segmenter.segment_pcd(points, intensity))
        if predictions.shape != (len(points),):
            raise ValueError(
                f'{file_path}: expected {len(points)} predictions, got {predictions.shape}'
            )
        if predictions.size and (
            predictions.min() < 0 or predictions.max() >= segmenter.n_classes
        ):
            raise ValueError(f'{file_path}: predictions outside model class range')

        labeled_indices = np.flatnonzero(stored_targets > 0)
        if labeled_indices.size == 0:
            continue
        targets = stored_targets[labeled_indices] - 1
        if targets.min() < 0 or targets.max() >= segmenter.n_classes:
            raise ValueError(f'{file_path}: targets outside model class range')
        chosen = _stratified_indices(targets, max_points, rng)
        sample = np.column_stack((targets[chosen], predictions[labeled_indices][chosen]))
        sample_path = pth.Path(temp_dir) / f'sample_{file_index:06d}.npy'
        np.save(sample_path, sample.astype(np.int16, copy=False))
        sample_paths.append(sample_path)
        labeled_files += 1
        sampled_points += len(sample)

    return {
        'sample_paths': sample_paths,
        'processed_files': len(files),
        'labeled_files': labeled_files,
        'sampled_points': sampled_points,
    }
def _eval_model(config_dict: dict,
                model: nn.Module) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_cpu
    
    test_dataset = Dataset(base_dir=config_dict['data_path_test'],
                                    num_points=config_dict['num_points'],
                                    batch_size=config_dict['batch_size'],
                                    shuffle=False,
                                    device=device_loader)

    testLoader = DataLoader(test_dataset,
                             batch_size=None,
                             num_workers = 14,
                             pin_memory=False)

    total = get_dataset_len(testLoader, verbose=False)
    weights = compute_pos_weights_h5(config_dict['data_path_test'], 
                                      config_dict['num_classes'], 
                                      power=0.25)
    weights = weights.to(device_loss)
    
    criterion = FocalLoss(alpha=weights.to(device_loss),
                                gamma=config_dict['focal_loss_gamma']).to(device_loss)

    loss_per_epoch = 0.
    epoch_samples = 0

    all_predictions = []
    all_probs = np.zeros((0, config_dict['num_classes']))
    all_labels = []

    pbar = tqdm(testLoader, total=total, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_x, batch_y in pbar:

            model.eval()
            batch_x = batch_x.to(config_dict['device'])


            outputs = model(batch_x)
            outputs = outputs.to(device_loss)
            batch_y = batch_y.to(device_loss)

            loss = criterion(outputs, batch_y)
            
            loss_per_epoch += loss.item()*batch_y.size(0)

            epoch_samples += batch_y.size(0)

            total_loss = loss_per_epoch / epoch_samples


            all_labels.extend(batch_y.cpu().tolist())

            probs = get_Probabilities(outputs.cpu())
            int_preds = get_intLabels(probs)

            probs = probs.numpy()
            int_preds = int_preds.numpy()

            all_probs = np.concatenate([all_probs, probs.reshape(-1, config_dict['num_classes'])], axis=0)
            all_predictions.extend(int_preds)

    return total_loss, np.asarray(all_labels), np.asarray(all_predictions)

def eval_model_front(config_dict: dict,
         model: nn.Module,
         paths: list[pth.Path]):

    model_path = paths[0]
    model_name = model_path.stem

    plot_dir = paths[1]

    total_loss, all_labels, all_predictions  = _eval_model(config_dict=config_dict,
                                                                                      model=model)
    
    miou, avg_iou_pc = compute_mIoU(torch.asarray(all_predictions), torch.asarray(all_labels), config_dict['num_classes'])
    
    print('='*20)
    print('MODEL TESTED')
    print('Model path', model_path)
    print('Loss: ', total_loss)
    print('mIoU: ', miou)
    print('IoU per class: ', avg_iou_pc)
    print('Plots saved to:', plot_dir)
    print('='*20)
    

    miou_report = f'mIoU: {miou}\nIoU per class: {avg_iou_pc}'
    ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_{model_name}.txt'),
                         pred=all_predictions,
                         target=all_labels,
                         additional_info=miou_report)

def test_function(config_dict: dict,
                  model):
    val_dataset = Dataset(base_dir=config_dict['data_path_test'],
                                    num_points=config_dict['num_points'],
                                    batch_size=config_dict['batch_size'],
                                    shuffle=False,
                                    device=torch.device('gpu'))

    valLoader = DataLoader(val_dataset,
                             batch_size=None,
                             num_workers = 14,
                             pin_memory=False)
    
    batch_x, batch_y = next(iter(valLoader))
    batch_x = batch_x.to(config_dict['device'])

    model.eval()
    outputs = model(batch_x)

    if outputs.shape == (batch_x.shape[0], config_dict['num_classes']):
        print('Model works as expected')
    else:
        print(f'Model does not work as expected\n')
        print(f'Expected output shape: (batch_x.shape[0], {config_dict["num_classes"]})\nReceived: {outputs.shape}')

def parser():
        
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """
    
    parser = argparse.ArgumentParser(
        description="Script for testing the choosen model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID."
        )
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: testing mode - check if model compiles and works as expected\n'
            '1: evaluate trained model'
        )
    )

    return parser.parse_args()

def main():
    args = parser()
    base_path = pth.Path(__file__).parent
    device_name = args.device
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')

    model_name = args.model_name
    model_name_no_num = model_name.rsplit('_', 1)[0]

    model_dir = base_path.joinpath(f'training_results/{model_name_no_num}')
    config_trained_dir = model_dir.joinpath('dict_files')
    model_path = config_trained_dir.joinpath(f'{model_name}_config.json')
    plot_dir = model_dir.joinpath('plots')

    config_dict = load_json(model_path)
    config_dict = convert_str_values(config_dict)
    config_dict['device'] = device
    
    model = RandLANet(model_config=config_dict['model_config'], n_classes=config_dict['num_classes'])
    model = load_model(file_path=model_dir.joinpath(f'{model_name}.pt'),
                       model=model,
                       device=device)
    model.eval()

    if args.mode == 0:
        test_function(config_dict, model)
    elif args.mode == 1:
        eval_model_front(config_dict=config_dict,
                         model=model,
                         paths=[model_path,
                                plot_dir])
        



if __name__ == '__main__':
    main()
