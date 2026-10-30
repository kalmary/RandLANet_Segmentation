"""Evaluate sparse inputs without modifying the branch's inference pipeline."""

import argparse
import inspect
from pathlib import Path
import sys

import laspy
import numpy as np
from sklearn.metrics import classification_report
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.array_processing import SegmentClass


MODEL_NAME = "RandLANetDV0_2"
VOXEL_SIZES = (0.1, 0.25, 0.5, 1.0)
RAW_PATH = Path("/mnt/data-ssd/DALES/raw/dales_las/test")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--num_files", type=positive_int, default=3,
                     help="Evaluate the first N sorted LAS/LAZ files (default: 3).")
    cli.add_argument("--raw_path", type=Path, default=RAW_PATH)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--output_dir", type=Path,
                     default=PROJECT_ROOT / "evaluation_results" / "density")
    cli.add_argument("--model_dir", type=Path,
                     help="Checkpoint directory; configs must be in its dict_files folder.")
    cli.add_argument("--seed", type=int, default=42)
    return cli.parse_args(argv)


def input_files(raw_path, num_files):
    if not raw_path.is_dir():
        raise NotADirectoryError(raw_path)
    files = sorted(path for path in raw_path.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".las", ".laz"}
                   and not path.stem.endswith("_mod"))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files in {raw_path}")
    return files[:num_files]


def voxel_indices(centered_points, voxel_size):
    """Keep the original point nearest each cell center, as in prob-sampler.

    Sorting XYZ integer keys directly avoids overflow from encoding three
    voxel coordinates into one integer. Equal-distance ties use input order.
    """
    if not len(centered_points):
        raise ValueError("point cloud cannot be empty")
    keys = np.floor(centered_points / voxel_size).astype(np.int64)
    centers = (keys + 0.5) * voxel_size
    distances = np.sum((centered_points - centers) ** 2, axis=1)
    order = np.lexsort((distances, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.concatenate(([True], np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)))
    return order[first]


def build_segmenter(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model_dir = (args.model_dir or PROJECT_ROOT / "src" / "model_pipeline"
                 / "training_results" / MODEL_NAME.rsplit("_", 1)[0]).resolve()
    for path in (model_dir / f"{MODEL_NAME}.pt",
                 model_dir / "dict_files" / f"{MODEL_NAME}_config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = dict(model_name=MODEL_NAME, model_dir=model_dir,
                    config_dir=model_dir / "dict_files", device=torch.device(args.device))
    parameters = inspect.signature(SegmentClass).parameters
    settings["pbar_bool" if "pbar_bool" in parameters else "verbose"] = True
    return SegmentClass(**settings)


def predict_sparse(segmenter, points, intensity, targets):
    parameters = inspect.signature(segmenter.segment_pcd).parameters
    if "return_indices" in parameters:
        predictions, retained = segmenter.segment_pcd(points, intensity, return_indices=True)
        targets = targets[retained]
    else:
        predictions = segmenter.segment_pcd(points, intensity)
    predictions = np.asarray(predictions)
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predictions.shape}, {targets.shape}")
    return predictions, targets


def report_from_confusion(confusion):
    """Produce an exact sklearn report without storing millions of predictions."""
    truth, predicted = np.nonzero(confusion)
    if not len(truth):
        raise ValueError("Selected clouds contain no labeled evaluated points")
    # Each occupied confusion cell represents that many identical target/prediction pairs.
    return classification_report(
        truth, predicted, sample_weight=confusion[truth, predicted],
        labels=np.arange(len(confusion)), digits=4, zero_division=0,
    )


def run(args):
    files = input_files(args.raw_path, args.num_files)
    segmenter = build_segmenter(args)
    report_dir = args.output_dir / MODEL_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for voxel_size in VOXEL_SIZES:
        confusion = np.zeros((segmenter.n_classes, segmenter.n_classes), dtype=np.int64)
        raw_count = sparse_count = evaluated_count = 0
        # Every size starts from the same original files, never the previous sparse cloud.
        for file_index, path in enumerate(files):
            print(f"Voxel {voxel_size:g} m: {path.name}", flush=True)
            cloud = laspy.read(path)
            points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
            if not len(points):
                raise ValueError(f"Empty cloud: {path}")
            points -= points.mean(axis=0)
            intensity = np.asarray(cloud.intensity, dtype=np.float32)
            targets = np.asarray(cloud.classification, dtype=np.int64) - 1
            # DV0 training preprocessing scales intensity to [0, 10]. The
            # probability branch normalizes intensity inside SegmentClass.
            if MODEL_NAME == "RandLANetDV0_2":
                intensity -= intensity.min()
                maximum = intensity.max()
                if maximum > 0:
                    intensity *= 10.0 / maximum
            indices = voxel_indices(points, voxel_size)
            seed = args.seed + file_index
            np.random.seed(seed)
            torch.manual_seed(seed)
            if hasattr(segmenter, "_rng"):
                segmenter._rng = np.random.default_rng(seed)
            predictions, sparse_targets = predict_sparse(
                segmenter, points[indices], intensity[indices], targets[indices],
            )
            raw_count += len(points)
            sparse_count += len(indices)
            evaluated_count += len(predictions)
            valid = sparse_targets >= 0  # LAS class 0 provides context only.
            truth = sparse_targets[valid]
            predicted = predictions[valid].astype(np.int64)
            if len(truth):
                if (truth.max() >= segmenter.n_classes or predicted.min() < 0
                        or predicted.max() >= segmenter.n_classes):
                    raise ValueError(f"Labels outside model class range: {path}")
                confusion += np.bincount(
                    truth * segmenter.n_classes + predicted,
                    minlength=segmenter.n_classes ** 2,
                ).reshape(confusion.shape)
        report = report_from_confusion(confusion)
        report_path = report_dir / f"classification_report_voxel_{voxel_size:g}.txt"
        metadata = (f"Model: {MODEL_NAME}\nVoxel size: {voxel_size:g} m\n"
                    f"Raw points: {raw_count}\nSparse input points: {sparse_count}\n"
                    f"Returned predictions: {evaluated_count}\n"
                    f"Labeled evaluated points: {confusion.sum()}\nSeed: {args.seed}\n"
                    "Model labels are LAS classification minus one; class 0 is ignored.\n"
                    "Files:\n" + "\n".join(str(path) for path in files))
        report_path.write_text(metadata + "\n\n" + report, encoding="utf-8")
        reports.append(report_path)
        print(report)
        print(f"Saved {report_path}", flush=True)
    return reports


if __name__ == "__main__":
    run(parser())
