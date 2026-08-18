import pathlib
import sys

import laspy
import numpy as np
import pytest
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import main


class FixedSegmenter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def segment_pcd(self, points, intensity):
        return np.arange(len(points), dtype=np.int8) % 2


def write_las(path: pathlib.Path, point_count: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.arange(point_count)
    cloud.y = np.arange(point_count) + 1
    cloud.z = np.arange(point_count) + 2
    cloud.intensity = np.arange(point_count, dtype=np.uint16)
    cloud.classification = np.full(point_count, 7, dtype=np.uint8)
    cloud.write(path)


@pytest.fixture(autouse=True)
def isolated_tracking_files(tmp_path, monkeypatch):
    FixedSegmenter.instances = []
    monkeypatch.setattr(
        main,
        "PROCESSED_FILES_PATH",
        tmp_path / "processed_files.txt",
    )
    monkeypatch.setattr(
        main,
        "ERROR_FILES_PATH",
        tmp_path / "error_files.txt",
    )


def test_parser_accepts_only_processing_arguments(tmp_path):
    input_file = tmp_path / "cloud.las"
    write_las(input_file)

    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--verbose",
        "--device", "cpu",
        "--input_path", str(input_file),
        "--output_path", str(tmp_path / "output"),
    ])

    assert args.model_name == "RandLANet_1"
    assert args.verbose
    assert args.device == "cpu"
    assert not hasattr(args, "mode")


def test_single_file_creates_modified_copy_beside_source(tmp_path, monkeypatch):
    input_file = tmp_path / "cloud.las"
    write_las(input_file)
    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--input_path", str(input_file),
    ])
    monkeypatch.setattr(main, "SegmentClass", FixedSegmenter)

    outputs = main.process_files(args)

    output_file = tmp_path / "cloud_mod.las"
    assert outputs == [output_file]
    assert output_file.exists()
    np.testing.assert_array_equal(
        laspy.read(input_file).classification,
        [7, 7, 7, 7],
    )
    np.testing.assert_array_equal(
        laspy.read(output_file).classification,
        [1, 2, 1, 2],
    )
    assert main.PROCESSED_FILES_PATH.read_text(encoding="utf-8") == (
        f"{input_file.as_posix()} -> {output_file.as_posix()}\n"
    )


def test_directory_output_preserves_relative_paths(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    write_las(input_dir / "first.las")
    write_las(input_dir / "nested" / "second.las")
    write_las(input_dir / "ignored_mod.las")
    (input_dir / "notes.txt").write_text("not a point cloud")
    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
    ])
    monkeypatch.setattr(main, "SegmentClass", FixedSegmenter)

    outputs = main.process_files(args)

    assert outputs == [
        output_dir / "first_mod.las",
        output_dir / "nested" / "second_mod.las",
    ]
    assert all(path.exists() for path in outputs)
    assert len(FixedSegmenter.instances) == 1


def test_previously_processed_file_is_skipped_even_if_it_has_an_error(
    tmp_path,
    monkeypatch,
):
    input_file = tmp_path / "cloud.las"
    output_file = tmp_path / "cloud_mod.las"
    write_las(input_file)
    main.PROCESSED_FILES_PATH.write_text(
        f"{input_file.as_posix()} -> {output_file.as_posix()}\n",
        encoding="utf-8",
    )
    main.ERROR_FILES_PATH.write_text(
        f"{input_file.as_posix()}\n",
        encoding="utf-8",
    )
    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--input_path", str(input_file),
    ])
    monkeypatch.setattr(main, "SegmentClass", FixedSegmenter)

    outputs = main.process_files(args)

    assert outputs == []
    assert not output_file.exists()
    assert not FixedSegmenter.instances


def test_failed_file_is_logged_and_other_files_continue(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    bad_file = input_dir / "bad.las"
    good_file = input_dir / "good.las"
    bad_file.parent.mkdir(parents=True)
    bad_file.write_text("not a LAS file", encoding="utf-8")
    write_las(good_file)
    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
    ])
    monkeypatch.setattr(main, "SegmentClass", FixedSegmenter)

    outputs = main.process_files(args)

    good_output = output_dir / "good_mod.las"
    assert outputs == [good_output]
    assert main.ERROR_FILES_PATH.read_text(encoding="utf-8") == (
        f"{bad_file.as_posix()}\n"
    )
    assert main.PROCESSED_FILES_PATH.read_text(encoding="utf-8") == (
        f"{good_file.as_posix()} -> {good_output.as_posix()}\n"
    )


def test_cuda_request_fails_when_cuda_is_unavailable(tmp_path, monkeypatch):
    input_file = tmp_path / "cloud.las"
    write_las(input_file)
    args = main.argparser([
        "--model_name", "RandLANet_1",
        "--device", "cuda",
        "--input_path", str(input_file),
    ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "SegmentClass", FixedSegmenter)

    with pytest.raises(RuntimeError, match="not available"):
        main.process_files(args)

    assert not FixedSegmenter.instances
