import argparse
from pathlib import Path

import laspy
import numpy as np
import torch
from tqdm import tqdm

try:
    from .array_processing import SegmentClass
except ImportError:
    from array_processing import SegmentClass


POINT_CLOUD_SUFFIXES = {".las", ".laz"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILES_PATH = PROJECT_ROOT / "processed_files.txt"
ERROR_FILES_PATH = PROJECT_ROOT / "error_files.txt"
TRACKING_SEPARATOR = " -> "


def _model_name(value: str) -> str:
    if value.lower().endswith(".pt"):
        raise argparse.ArgumentTypeError("model_name must not include .pt")
    if not value:
        raise argparse.ArgumentTypeError("model_name cannot be empty")
    return value


def _input_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"input path does not exist: {path}")
    if path.is_file() and path.suffix.lower() not in POINT_CLOUD_SUFFIXES:
        raise argparse.ArgumentTypeError("input file must be .las or .laz")
    if not path.is_file() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"unsupported input path: {path}")
    return path


def argparser(args=None):
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of LAS and LAZ point clouds."
    )
    parser.add_argument(
        "--model_name",
        type=_model_name,
        required=True,
        help="Model filename without the .pt extension.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display file and inference progress.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for model inference.",
    )
    parser.add_argument(
        "--input_path",
        type=_input_path,
        required=True,
        help="A LAS/LAZ file or a directory containing LAS/LAZ files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional directory for modified files.",
    )
    return parser.parse_args(args)


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in POINT_CLOUD_SUFFIXES
        and not path.stem.endswith("_mod")
    )


def _output_file(
    source_path: Path,
    input_path: Path,
    output_path: Path | None,
) -> Path:
    file_name = f"{source_path.stem}_mod{source_path.suffix}"
    if output_path is None:
        return source_path.parent / file_name
    if input_path.is_file():
        return output_path / file_name
    relative_parent = source_path.relative_to(input_path).parent
    return output_path / relative_parent / file_name


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _read_processed_sources(path: Path = PROCESSED_FILES_PATH) -> set[Path]:
    if not path.exists():
        return set()

    sources: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        source, separator, _ = line.partition(TRACKING_SEPARATOR)
        if separator and source.strip():
            sources.add(_absolute_path(Path(source.strip())))
    return sources


def _append_processed_file(
    source_path: Path,
    output_path: Path,
    tracking_path: Path = PROCESSED_FILES_PATH,
) -> None:
    source = _absolute_path(source_path).as_posix()
    output = _absolute_path(output_path).as_posix()
    with tracking_path.open("a", encoding="utf-8") as file:
        file.write(f"{source}{TRACKING_SEPARATOR}{output}\n")


def _append_error_file(
    source_path: Path,
    tracking_path: Path = ERROR_FILES_PATH,
) -> None:
    source = _absolute_path(source_path).as_posix()
    with tracking_path.open("a", encoding="utf-8") as file:
        file.write(f"{source}\n")


def process_files(args) -> list[Path]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.output_path is not None and args.output_path.exists():
        if not args.output_path.is_dir():
            raise ValueError("output_path must be a directory")

    files = _input_files(args.input_path)
    if not files:
        raise FileNotFoundError(f"No LAS or LAZ files in {args.input_path}")

    processed_sources = _read_processed_sources(PROCESSED_FILES_PATH)
    files = [
        source_path
        for source_path in files
        if _absolute_path(source_path) not in processed_sources
    ]
    if not files:
        return []

    segmenter = SegmentClass(
        model_name=args.model_name,
        device=torch.device(args.device),
        pbar_bool=args.verbose,
    )
    outputs = []
    iterator = files
    if args.verbose:
        iterator = tqdm(files, desc="Processing files", unit="file")

    for source_path in iterator:
        try:
            output_file = _output_file(
                source_path,
                args.input_path,
                args.output_path,
            )
            if output_file.exists():
                raise FileExistsError(f"Output file already exists: {output_file}")

            cloud = laspy.read(source_path)
            points = np.column_stack((cloud.x, cloud.y, cloud.z))
            intensity = np.asarray(cloud.intensity)
            labels = np.asarray(
                segmenter.segment_pcd(points, intensity),
                dtype=np.int16,
            )
            if labels.shape != (len(points),):
                raise ValueError(
                    f"Expected {len(points)} labels, got shape {labels.shape}"
                )

            labels += 1
            if labels.min() < 1 or labels.max() > 255:
                raise ValueError(
                    "Predicted LAS classifications must be within [1, 255]"
                )

            output_file.parent.mkdir(parents=True, exist_ok=True)
            cloud.classification = labels.astype(np.uint8)
            cloud.write(output_file)
            _append_processed_file(
                source_path,
                output_file,
                PROCESSED_FILES_PATH,
            )
            outputs.append(output_file)
        except Exception as error:
            _append_error_file(source_path, ERROR_FILES_PATH)
            if args.verbose:
                tqdm.write(f"Error processing {source_path}: {error}")
            continue

        if args.verbose:
            iterator.set_postfix_str(source_path.name)

    return outputs


def main():
    _ = process_files(argparser())


if __name__ == "__main__":
    main()
