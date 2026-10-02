from types import SimpleNamespace

import numpy as np
import pytest

from src.model_pipeline import EvalSegm_RandLANet as evaluation


def test_stratified_indices_are_exact_reproducible_and_keep_rare_classes():
    targets = np.array([0] * 8 + [1] * 3 + [2] * 2)

    first = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )
    second = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )

    assert len(first) == 6
    assert len(np.unique(first)) == 6
    np.testing.assert_array_equal(first, second)
    assert np.bincount(targets[first], minlength=3).tolist() == [2, 2, 2]


def test_stratified_indices_return_all_points_below_cap_and_validate_cap():
    indices = evaluation._stratified_indices(
        np.array([1, 0, 1]), 4, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(indices, [0, 1, 2])

    with pytest.raises(ValueError, match="positive"):
        evaluation._stratified_indices(
            np.array([0]), 0, np.random.default_rng(0)
        )


def test_metrics_report_accuracy_and_ordered_class_iou():
    metrics = evaluation.calculate_metrics(
        predictions=np.array([0, 0, 0, 1]),
        targets=np.array([0, 0, 1, 1]),
        num_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["miou"] == pytest.approx(7 / 12)
    np.testing.assert_allclose(metrics["class_iou"], [2 / 3, 1 / 2])


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([0], [0, 1], "same shape"),
        ([], [], "empty"),
        ([-1], [0], "predictions"),
        ([0], [2], "targets"),
    ],
)
def test_metrics_reject_invalid_arrays(predictions, targets, message):
    with pytest.raises(ValueError, match=message):
        evaluation.calculate_metrics(predictions, targets, num_classes=2)


def test_cloud_files_are_recursive_sorted_and_case_insensitive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = [tmp_path / "a.LAS", nested / "b.laz"]
    for path in [*expected, tmp_path / "ignored.npy"]:
        path.touch()

    assert evaluation._cloud_files(tmp_path) == expected
    with pytest.raises(NotADirectoryError):
        evaluation._cloud_files(expected[0])
    with pytest.raises(FileNotFoundError, match="No LAS or LAZ"):
        evaluation._cloud_files(tmp_path / "empty")


def test_device_rejects_unavailable_cuda(monkeypatch):
    assert evaluation._device("cpu").type == "cpu"
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        evaluation._device("cuda")
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: True)
    assert evaluation._device("cuda").type == "cuda"


def test_collect_samples_segments_whole_cloud_then_saves_labeled_pairs(
    tmp_path, monkeypatch
):
    cloud_path = tmp_path / "cloud.laz"
    cloud_path.touch()
    cloud = SimpleNamespace(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([4.0, 5.0, 6.0]),
        z=np.array([7.0, 8.0, 9.0]),
        intensity=np.array([10, 11, 12]),
        classification=np.array([0, 1, 2]),
    )
    calls = []

    class Segmenter:
        n_classes = 2

        def segment_pcd(self, points, intensity):
            calls.append((points.copy(), intensity.copy()))
            return np.array([1, 0, 1])

    monkeypatch.setattr(evaluation.laspy, "read", lambda path: cloud)
    output = tmp_path / "samples"
    output.mkdir()

    summary = evaluation.collect_samples(
        Segmenter(), [cloud_path], 50_000, output, np.random.default_rng(0), False
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (3, 3)
    np.testing.assert_array_equal(calls[0][1], [10, 11, 12])
    sample = np.load(summary["sample_paths"][0])
    np.testing.assert_array_equal(sample, [[0, 0], [1, 1]])
    assert summary["sampled_points"] == 2


def test_collect_samples_infers_unlabeled_cloud_without_writing_sample(
    tmp_path, monkeypatch
):
    path = tmp_path / "unlabeled.las"
    path.touch()
    cloud = SimpleNamespace(
        x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        intensity=np.array([1]), classification=np.array([0]),
    )
    calls = []
    segmenter = SimpleNamespace(
        n_classes=2,
        segment_pcd=lambda points, intensity: calls.append(len(points)) or np.array([0]),
    )
    monkeypatch.setattr(evaluation.laspy, "read", lambda path: cloud)

    summary = evaluation.collect_samples(
        segmenter, [path], 10, tmp_path, np.random.default_rng(0), False
    )

    assert calls == [1]
    assert summary["sample_paths"] == []
    assert summary["labeled_files"] == 0
