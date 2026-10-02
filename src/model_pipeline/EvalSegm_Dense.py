import argparse
import pathlib as pth
import sys
import tempfile
from typing import Any, Sequence, TypedDict

import laspy
import numpy as np
import torch
from tqdm import tqdm

if __package__:
    from ..array_processing import SegmentClass
    from ..utils import ClassificationReport, compute_mIoU
else:
    src_dir = pth.Path(__file__).parent.parent
    sys.path.append(str(src_dir))
    from array_processing import SegmentClass
    from utils import ClassificationReport, compute_mIoU


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


class ArtifactPaths(TypedDict):
    model_dir: pth.Path
    config_dir: pth.Path
    model_path: pth.Path
    config_path: pth.Path
    report_dir: pth.Path


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return number


def _model_name(value: str) -> str:
    if not value or value.lower().endswith('.pt'):
        raise argparse.ArgumentTypeError('model_name must be non-empty and omit .pt')
    return value


def _raw_directory(value: str) -> pth.Path:
    path = pth.Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'raw_path is not a directory: {path}')
    return path


def parser(args: Sequence[str] | None = None) -> argparse.Namespace:
    cli = argparse.ArgumentParser(description='Evaluate complete raw LAS/LAZ clouds.')
    cli.add_argument('--model_name', required=True, type=_model_name)
    cli.add_argument('--raw_path', required=True, type=_raw_directory)
    cli.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    cli.add_argument('--mode', choices=[0, 1], type=int, default=0)
    cli.add_argument('--max_points', type=_positive_int, default=50_000)
    return cli.parse_args(args)


def _cloud_files(raw_path: pth.Path) -> list[pth.Path]:
    raw_path = pth.Path(raw_path)
    if raw_path.exists() and not raw_path.is_dir():
        raise NotADirectoryError(f'raw_path is not a directory: {raw_path}')
    files = sorted(path for path in raw_path.rglob('*') if path.is_file() and path.suffix.lower() in {'.las', '.laz'}) if raw_path.is_dir() else []
    if not files:
        raise FileNotFoundError(f'No LAS or LAZ files in {raw_path}')
    return files


def _device(name: str) -> torch.device:
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device(name)


def _model_paths(model_name: str) -> ArtifactPaths:
    base_dir = pth.Path(__file__).parent
    family = model_name.rsplit('_', 1)[0]
    model_dir = base_dir / 'training_results' / family
    config_dir = model_dir / 'dict_files'
    return {'model_dir': model_dir, 'config_dir': config_dir,
            'model_path': model_dir / f'{model_name}.pt',
            'config_path': config_dir / f'{model_name}_config.json',
            'report_dir': model_dir / 'plots'}


def _build_segmenter(model_name: str, paths: ArtifactPaths, device: torch.device) -> SegmentClass:
    return SegmentClass(voxel_size_big=100.0, overlap=0.4, scaled=True,
                        model_name=model_name, config_dir=paths['config_dir'],
                        model_dir=paths['model_dir'], device=device, verbose=True)


def _validated_segmenter(args: argparse.Namespace) -> SegmentClass:
    paths = _model_paths(args.model_name)
    for key in ('model_path', 'config_path'):
        if not paths[key].is_file():
            raise FileNotFoundError(f'Missing {key}: {paths[key]}')
    return _build_segmenter(args.model_name, paths, _device(args.device))


def _stratified_indices(targets: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
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
    selected = [rng.choice(np.flatnonzero(targets == label), size=quota, replace=False)
                for label, quota in zip(classes, quotas) if quota]
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> EvaluationMetrics:
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError('predictions and targets must have the same shape')
    if targets.size == 0:
        raise ValueError('predictions and targets cannot be empty')
    for name, values in [('predictions', predictions), ('targets', targets)]:
        if values.min() < 0 or values.max() >= num_classes:
            raise ValueError(f'{name} are outside the model class range')
    miou, class_iou = compute_mIoU(torch.from_numpy(predictions.astype(np.int64, copy=False)), torch.from_numpy(targets.astype(np.int64, copy=False)), num_classes)
    return {'accuracy': float(np.mean(predictions == targets)), 'miou': miou,
            'class_iou': class_iou.cpu().numpy(), 'predictions': predictions,
            'targets': targets}


def collect_samples(segmenter: Any, files: list[pth.Path], max_points: int,
                    temp_dir: pth.Path, rng: np.random.Generator,
                    verbose: bool = True) -> CollectionSummary:
    sample_paths: list[pth.Path] = []
    labeled_files = sampled_points = 0
    iterator = tqdm(files, desc='Evaluating files', unit='file') if verbose else files
    for file_index, file_path in enumerate(iterator):
        try:
            cloud = laspy.read(file_path)
            points = np.column_stack((
                np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)
            ))
            intensity = np.asarray(cloud.intensity)
            stored_targets = np.asarray(cloud.classification, dtype=np.int64)
            predictions = np.asarray(segmenter.segment_pcd(points, intensity))
        except Exception as error:
            raise RuntimeError(
                f'Failed to evaluate {file_path}: {error}'
            ) from error
        if len(intensity) != len(points) or len(stored_targets) != len(points):
            raise ValueError(f'{file_path}: point attributes have mismatched lengths')
        if predictions.shape != (len(points),):
            raise ValueError(f'{file_path}: expected {len(points)} predictions, got {predictions.shape}')
        if predictions.size and (predictions.min() < 0 or predictions.max() >= segmenter.n_classes):
            raise ValueError(f'{file_path}: predictions outside model class range')
        labeled_indices = np.flatnonzero(stored_targets > 0)
        if not labeled_indices.size:
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
    return {'sample_paths': sample_paths, 'processed_files': len(files),
            'labeled_files': labeled_files, 'sampled_points': sampled_points}


def run_dry_run(args: argparse.Namespace) -> None:
    files = _cloud_files(args.raw_path)
    _validated_segmenter(args)
    print(f'Dry run passed: {len(files)} LAS/LAZ files, device={args.device}')


def run_evaluation(args: argparse.Namespace) -> EvaluationMetrics:
    files = _cloud_files(args.raw_path)
    segmenter = _validated_segmenter(args)
    paths = _model_paths(args.model_name)
    with tempfile.TemporaryDirectory(prefix='randlanet-evaluation-') as directory:
        summary = collect_samples(segmenter, files, args.max_points, pth.Path(directory), np.random.default_rng(0), True)
        if not summary['sample_paths']:
            raise RuntimeError('Input files contain no labeled points')
        combined = np.concatenate([np.load(path) for path in summary['sample_paths']])
        metrics = calculate_metrics(combined[:, 1], combined[:, 0], segmenter.n_classes)
    paths['report_dir'].mkdir(exist_ok=True, parents=True)
    info = f"Accuracy: {metrics['accuracy']}\nmIoU: {metrics['miou']}\nIoU per class: {metrics['class_iou']}"
    ClassificationReport(file_path=paths['report_dir'] / f'classification_report_{args.model_name}.txt', pred=metrics['predictions'], target=metrics['targets'], additional_info=info)
    print(f"Processed files: {summary['processed_files']}")
    print(f"Sampled points: {summary['sampled_points']}")
    print(info)
    return metrics


def main() -> None:
    args = parser()
    run_dry_run(args) if args.mode == 0 else run_evaluation(args)


if __name__ == '__main__':
    main()
