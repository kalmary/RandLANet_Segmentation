import argparse
import pathlib as pth
import sys

import laspy
import numpy as np
import torch
from tqdm import tqdm


model_pipeline_dir = pth.Path(__file__).parent
src_dir = model_pipeline_dir.parent
sys.path.append(str(src_dir))

from array_processing import SegmentClass
from utils import ClassificationReport, Plotter, compute_mIoU


POINT_CLOUD_SUFFIXES = {'.las', '.laz'}
MAX_POINTS_PER_FILE = 0


def _model_name(value: str) -> str:
    if value.lower().endswith('.pt'):
        raise argparse.ArgumentTypeError('model_name must not include .pt')
    if not value:
        raise argparse.ArgumentTypeError('model_name cannot be empty')
    return value


def _input_files(input_path: pth.Path) -> list[pth.Path]:
    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob('*')
        if path.is_file()
        and path.suffix.lower() in POINT_CLOUD_SUFFIXES
        and not path.stem.endswith('_mod')
    )


def collect_labels(
    segmenter: SegmentClass,
    input_path: pth.Path,
    verbose: bool = False,
    max_points_per_file: int = MAX_POINTS_PER_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points_per_file < 0:
        raise ValueError('max_points_per_file cannot be negative')

    files = _input_files(input_path)
    if not files:
        raise FileNotFoundError(f'No LAS or LAZ files in {input_path}')

    prediction_parts = []
    target_parts = []
    iterator = files
    if verbose:
        iterator = tqdm(files, desc='Evaluating files', unit='file')

    for file_path in iterator:
        cloud = laspy.read(file_path)
        source_labels = np.asarray(cloud.classification, dtype=np.int16)
        assessed = source_labels != 0
        if not np.any(assessed):
            continue

        targets = source_labels[assessed] - 1
        if targets.max() >= segmenter.n_classes:
            raise ValueError(
                f'{file_path} contains class {targets.max() + 1}, '
                f'but the model has {segmenter.n_classes} classes'
            )

        evaluation_indices = None
        if max_points_per_file:
            evaluation_indices = np.flatnonzero(assessed)
            if len(evaluation_indices) > max_points_per_file:
                evaluation_indices = np.sort(
                    np.random.choice(
                        evaluation_indices,
                        size=max_points_per_file,
                        replace=False,
                    )
                )
            targets = source_labels[evaluation_indices] - 1

        if evaluation_indices is None:
            points = np.column_stack((cloud.x, cloud.y, cloud.z))
            intensity = np.asarray(cloud.intensity)
            expected_predictions = len(source_labels)
        else:
            points = np.column_stack(
                (
                    cloud.X[evaluation_indices] * cloud.header.x_scale
                    + cloud.header.x_offset,
                    cloud.Y[evaluation_indices] * cloud.header.y_scale
                    + cloud.header.y_offset,
                    cloud.Z[evaluation_indices] * cloud.header.z_scale
                    + cloud.header.z_offset,
                )
            )
            intensity = np.asarray(cloud.intensity[evaluation_indices])
            expected_predictions = len(evaluation_indices)

        predictions = np.asarray(
            segmenter.segment_pcd(points, intensity),
            dtype=np.int16,
        )
        if predictions.shape != (expected_predictions,):
            raise ValueError(
                f'Expected {expected_predictions} predictions for {file_path}, '
                f'got shape {predictions.shape}'
            )
        if predictions.min() < 0 or predictions.max() >= segmenter.n_classes:
            raise ValueError(f'Predictions outside the model class range for {file_path}')

        if evaluation_indices is None:
            predictions = predictions[assessed]

        prediction_parts.append(predictions.astype(np.int8))
        target_parts.append(targets.astype(np.int8))

        if verbose:
            iterator.set_postfix_str(file_path.name)

    if not target_parts:
        raise RuntimeError('Input files contain no classified points')

    return np.concatenate(prediction_parts), np.concatenate(target_parts)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict:
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError('predictions and targets must have the same shape')
    if targets.size == 0:
        raise ValueError('predictions and targets cannot be empty')

    accuracy = float(np.mean(predictions == targets))
    miou, class_iou = compute_mIoU(
        torch.from_numpy(predictions.astype(np.int64, copy=False)),
        torch.from_numpy(targets.astype(np.int64, copy=False)),
        num_classes,
    )
    return {
        'accuracy': accuracy,
        'miou': miou,
        'class_iou': class_iou.numpy(),
        'predictions': predictions,
        'targets': targets,
    }


def eval_model_front(
    segmenter: SegmentClass,
    input_path: pth.Path,
    model_path: pth.Path,
    plot_dir: pth.Path,
    verbose: bool = False,
    max_points_per_file: int = MAX_POINTS_PER_FILE,
) -> dict:
    model_name = model_path.stem
    plot_dir.mkdir(exist_ok=True, parents=True)

    predictions, targets = collect_labels(
        segmenter,
        input_path,
        verbose=verbose,
        max_points_per_file=max_points_per_file,
    )
    metrics = calculate_metrics(
        predictions,
        targets,
        segmenter.n_classes,
    )

    plotter = Plotter(segmenter.n_classes, plots_dir=plot_dir)
    plotter.cnf_matrix(
        f'confusion_matrix_{model_name}.png',
        target=metrics['targets'],
        prediction=metrics['predictions'],
        num_classes=segmenter.n_classes,
    )

    print('=' * 20)
    print('MODEL TESTED')
    print('Model path', model_path)
    print('Accuracy: ', metrics['accuracy'])
    print('mIoU: ', metrics['miou'])
    print('IoU per class: ', metrics['class_iou'])
    print('Plots saved to:', plot_dir)
    print('=' * 20)

    metrics_report = (
        f"Accuracy: {metrics['accuracy']}\n"
        f"mIoU: {metrics['miou']}\n"
        f"IoU per class: {metrics['class_iou']}"
    )
    ClassificationReport(
        file_path=plot_dir / f'classification_report_{model_name}.txt',
        pred=metrics['predictions'],
        target=metrics['targets'],
        additional_info=metrics_report,
    )
    return metrics


def parser(args=None):
    argument_parser = argparse.ArgumentParser(
        description='Dense semantic-segmentation evaluation for LAS and LAZ files.'
    )
    argument_parser.add_argument(
        '--model_name',
        type=_model_name,
        required=True,
        help='Model filename without the .pt extension.',
    )
    return argument_parser.parse_args(args)


def main():
    args = parser()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for evaluation')

    model_name_no_num = args.model_name.rsplit('_', 1)[0]
    model_dir = model_pipeline_dir / 'training_results' / model_name_no_num
    config_dir = model_dir / 'dict_files'
    model_path = model_dir / f'{args.model_name}.pt'
    plot_dir = model_dir / 'plots'

    segmenter = SegmentClass(
        model_name=args.model_name,
        config_dir=config_dir,
        model_dir=model_dir,
        device=torch.device('cuda'),
        pbar_bool=False,
    )
    input_path = pth.Path(segmenter.config['data_path_test'])
    eval_model_front(
        segmenter=segmenter,
        input_path=input_path,
        model_path=model_path,
        plot_dir=plot_dir,
        verbose=True,
        max_points_per_file=MAX_POINTS_PER_FILE,
    )


if __name__ == '__main__':
    main()
