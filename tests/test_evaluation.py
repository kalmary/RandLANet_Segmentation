from types import SimpleNamespace
import pathlib as pth

import numpy as np
import pytest

from src.model_pipeline import EvalSegm_RandLANet as evaluation


def test_stratified_indices_are_exact_reproducible_and_keep_rare_classes():
    targets = np.array([0] * 8 + [1] * 3 + [2])

    first = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )
    second = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )

    assert len(first) == 6
    assert len(np.unique(first)) == 6
    np.testing.assert_array_equal(first, second)
    assert np.bincount(targets[first], minlength=3).tolist() == [3, 2, 1]


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


def test_parser_exposes_raw_evaluation_contract(tmp_path):
    args = evaluation.parser([
        "--model_name", "Network_2", "--raw_path", str(tmp_path)
    ])
    assert args.device == "cpu"
    assert args.mode == 0
    assert args.max_points == 50_000
    assert args.raw_path == tmp_path

    with pytest.raises(SystemExit):
        evaluation.parser([
            "--model_name", "Network_2.pt", "--raw_path", str(tmp_path)
        ])
    with pytest.raises(SystemExit):
        evaluation.parser([
            "--model_name", "Network_2", "--raw_path", str(tmp_path),
            "--max_points", "0",
        ])


def test_model_paths_keep_existing_training_results_layout():
    paths = evaluation._model_paths("Network_2")
    assert paths["model_path"].as_posix().endswith(
        "model_pipeline/training_results/Network/Network_2.pt"
    )
    assert paths["config_path"].as_posix().endswith(
        "model_pipeline/training_results/Network/dict_files/Network_2_config.json"
    )


def test_run_evaluation_removes_temporary_samples(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    cloud = raw / "cloud.laz"
    cloud.touch()
    model = tmp_path / "Network_2.pt"
    config = tmp_path / "Network_2_config.json"
    model.touch()
    config.touch()
    report_dir = tmp_path / "reports"
    temp_paths = []

    monkeypatch.setattr(evaluation, "_cloud_files", lambda path: [cloud])
    monkeypatch.setattr(evaluation, "_build_segmenter", lambda *args, **kwargs: SimpleNamespace(n_classes=2))
    monkeypatch.setattr(
        evaluation,
        "_model_paths",
        lambda name: {
            "model_dir": tmp_path, "config_dir": tmp_path,
            "model_path": model, "config_path": config,
            "report_dir": report_dir,
        },
    )

    def collect(segmenter, files, max_points, temp_dir, rng, verbose=True):
        temp_paths.append(temp_dir)
        sample = temp_dir / "sample.npy"
        np.save(sample, np.array([[0, 0], [1, 1]]))
        return {"sample_paths": [sample], "processed_files": 1,
                "labeled_files": 1, "sampled_points": 2}

    monkeypatch.setattr(evaluation, "collect_samples", collect)
    monkeypatch.setattr(evaluation, "ClassificationReport", lambda **kwargs: None)
    args = SimpleNamespace(
        model_name="Network_2", raw_path=raw, device="cpu", mode=1,
        max_points=50_000,
    )

    metrics = evaluation.run_evaluation(args)

    assert metrics["accuracy"] == 1.0
    assert temp_paths and not temp_paths[0].exists()

def test_dry_run_does_not_read_or_segment_clouds(tmp_path, monkeypatch):
    args = SimpleNamespace(
        model_name="Network_2", raw_path=tmp_path, device="cpu", mode=0,
        max_points=50_000,
    )
    monkeypatch.setattr(evaluation, "_cloud_files", lambda path: [tmp_path / "a.laz"])
    monkeypatch.setattr(evaluation, "_validated_segmenter", lambda args: object())
    monkeypatch.setattr(
        evaluation.laspy, "read",
        lambda path: pytest.fail("dry run read a cloud"),
    )

    evaluation.run_dry_run(args)


def test_collect_samples_qualifies_reader_errors_with_source_path(
    tmp_path, monkeypatch
):
    path = tmp_path / "broken.laz"
    path.touch()
    monkeypatch.setattr(
        evaluation.laspy, "read", lambda source: (_ for _ in ()).throw(ValueError("bad header"))
    )

    with pytest.raises(RuntimeError, match=r"broken\.laz.*bad header"):
        evaluation.collect_samples(
            SimpleNamespace(n_classes=2), [path], 10, tmp_path,
            np.random.default_rng(0), False,
        )


def test_run_evaluation_cleans_samples_when_later_file_fails(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    files = [raw / "first.laz", raw / "broken.laz"]
    for path in files:
        path.touch()
    good_cloud = SimpleNamespace(
        x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
        intensity=np.array([1]), classification=np.array([1]),
    )
    reads = iter([good_cloud, ValueError("bad header")])

    def read(path):
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    temp_paths = []
    real_collect = evaluation.collect_samples

    def collect(*args, **kwargs):
        temp_paths.append(args[3])
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_cloud_files", lambda path: files)
    monkeypatch.setattr(
        evaluation, "_validated_segmenter",
        lambda args: SimpleNamespace(
            n_classes=2,
            segment_pcd=lambda points, intensity: np.zeros(len(points), dtype=np.int64),
        ),
    )
    monkeypatch.setattr(evaluation, "collect_samples", collect)
    monkeypatch.setattr(evaluation.laspy, "read", read)
    args = SimpleNamespace(
        model_name="Network_2", raw_path=raw, device="cpu", mode=1,
        max_points=10,
    )

    with pytest.raises(RuntimeError, match="broken.laz"):
        evaluation.run_evaluation(args)

    assert temp_paths and not temp_paths[0].exists()
