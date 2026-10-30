"""Density evaluation checks without loading model checkpoints."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "density_evaluation", ROOT / "src/model_pipeline/EvalDensity_RandLANet.py",
)
density = importlib.util.module_from_spec(spec)
spec.loader.exec_module(density)


def test_cli_defaults_and_invalid_file_count():
    args = density.parser([])
    assert args.num_files == 3
    assert args.raw_path == Path("/mnt/data-ssd/DALES/raw/dales_las/test")
    with pytest.raises(SystemExit):
        density.parser(["--num_files", "0"])


def test_voxel_selection_retains_original_point_nearest_cell_center():
    points = np.array([[0.01, 0.01, 0.01], [0.49, 0.49, 0.49],
                       [1.01, 0.01, 0.01], [1.49, 0.49, 0.49]])
    np.testing.assert_array_equal(density.voxel_indices(points, 1.0), [1, 3])
    np.testing.assert_array_equal(density.voxel_indices(points, 0.1), [0, 1, 2, 3])


def test_input_files_are_sorted_limited_and_skip_modified_clouds(tmp_path):
    for name in ("c.las", "a.las", "b.laz", "a_mod.las", "notes.txt"):
        (tmp_path / name).touch()
    assert [path.name for path in density.input_files(tmp_path, 2)] == ["a.las", "b.laz"]


def test_sparse_prediction_aligns_ground_truth_after_internal_subsampling():
    class Segmenter:
        def segment_pcd(self, points, intensity, *, return_indices=False):
            assert return_indices
            return np.array([1, 0]), np.array([2, 0])

    predictions, targets = density.predict_sparse(
        Segmenter(), np.zeros((3, 3)), np.ones(3), np.array([0, -1, 1]),
    )
    np.testing.assert_array_equal(predictions, targets)


def test_weighted_confusion_report_uses_all_points():
    report = density.report_from_confusion(np.array([[900, 100], [9, 1]]))
    # Accuracy is 901/1010, not the balanced average recall of 0.5.
    assert "0.8921" in report
    assert "0.5000" in report
    assert "1010" in report
    with pytest.raises(ValueError, match="no labeled"):
        density.report_from_confusion(np.zeros((2, 2), dtype=np.int64))


@pytest.mark.parametrize("internal_subsampling", [False, True])
def test_full_loop_reports_each_size_and_resamples_original_cloud(
    tmp_path, monkeypatch, internal_subsampling,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    cloud.x = [0.0, 0.04, 0.3, 0.9, 1.5]
    cloud.y = np.zeros(5)
    cloud.z = np.zeros(5)
    cloud.intensity = np.ones(5, dtype=np.uint16)
    cloud.classification = np.array([0, 1, 2, 1, 2], dtype=np.uint8)
    cloud.write(raw / "a.las")
    calls = []

    class DenseSegmenter:
        n_classes = 2

        def segment_pcd(self, points, intensity):
            calls.append(points.copy())
            return np.zeros(len(points), dtype=np.int64)

    class SparseSegmenter(DenseSegmenter):
        def segment_pcd(self, points, intensity, *, return_indices=False):
            predictions = super().segment_pcd(points, intensity)
            indices = np.arange(len(points) - 1, -1, -2)
            return predictions[indices], indices

    segmenter = SparseSegmenter() if internal_subsampling else DenseSegmenter()
    monkeypatch.setattr(density, "build_segmenter", lambda args: segmenter)
    args = SimpleNamespace(raw_path=raw, num_files=3, output_dir=tmp_path / "reports", seed=42)
    paths = density.run(args)

    original = np.column_stack((cloud.x, cloud.y, cloud.z))
    original -= original.mean(axis=0)
    assert len(paths) == len(calls) == 4
    for size, received, path in zip(density.VOXEL_SIZES, calls, paths):
        indices = density.voxel_indices(original, size)
        np.testing.assert_array_equal(received, original[indices])
        if internal_subsampling:
            indices = indices[np.arange(len(indices) - 1, -1, -2)]
        count = np.count_nonzero(np.asarray(cloud.classification)[indices])
        report = path.read_text()
        assert f"Labeled evaluated points: {count}\n" in report
        assert "precision" in report and "f1-score" in report
    assert not list(args.output_dir.rglob("*.csv"))
