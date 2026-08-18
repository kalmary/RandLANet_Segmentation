import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.model_pipeline import EvalSegm_RandLANet as evaluation


class PlotRecorder:
    calls = []

    def __init__(self, class_num, plots_dir):
        self.class_num = class_num
        self.plots_dir = plots_dir

    def cnf_matrix(self, *args, **kwargs):
        self.calls.append('confusion')


class FakeSegmenter:
    def __init__(self, predictions, n_classes=2):
        self.predictions = np.asarray(predictions)
        self.n_classes = n_classes
        self.calls = []

    def segment_pcd(self, points, intensity):
        self.calls.append((points.copy(), intensity.copy()))
        return self.predictions[:len(points)]


@pytest.fixture(autouse=True)
def reset_plot_recorder():
    PlotRecorder.calls = []


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / 'cloud.las'
    path.touch()
    return path


def fake_cloud(labels):
    point_count = len(labels)
    return SimpleNamespace(
        x=np.arange(point_count, dtype=np.float64),
        y=np.arange(point_count, dtype=np.float64) + 10.0,
        z=np.arange(point_count, dtype=np.float64) + 20.0,
        intensity=np.arange(point_count, dtype=np.uint16),
        classification=np.asarray(labels, dtype=np.uint8),
    )


def test_input_files_find_raw_clouds_and_ignore_modified_outputs(tmp_path):
    expected = [tmp_path / 'a.las', tmp_path / 'nested' / 'b.laz']
    (tmp_path / 'nested').mkdir()
    for path in expected + [tmp_path / 'ignored_mod.las', tmp_path / 'other.npy']:
        path.touch()

    assert evaluation._input_files(tmp_path) == expected
    assert evaluation._input_files(expected[0]) == [expected[0]]
    assert evaluation._input_files(tmp_path / 'other.npy') == []


def test_collect_labels_evaluates_each_assessed_dense_point_once(
    raw_file,
    monkeypatch,
):
    cloud = fake_cloud([0, 1, 2, 1])
    segmenter = FakeSegmenter([1, 0, 1, 0])
    monkeypatch.setattr(evaluation.laspy, 'read', lambda path: cloud)

    predictions, targets = evaluation.collect_labels(segmenter, raw_file)

    np.testing.assert_array_equal(predictions, [0, 1, 0])
    np.testing.assert_array_equal(targets, [0, 1, 0])
    assert len(segmenter.calls) == 1
    points, intensity = segmenter.calls[0]
    assert points.shape == (4, 3)
    np.testing.assert_array_equal(intensity, [0, 1, 2, 3])


def test_collect_labels_skips_unclassified_files_and_rejects_empty_result(
    raw_file,
    monkeypatch,
):
    monkeypatch.setattr(
        evaluation.laspy,
        'read',
        lambda path: fake_cloud([0, 0, 0]),
    )

    with pytest.raises(RuntimeError, match='no classified points'):
        evaluation.collect_labels(FakeSegmenter([0, 0, 0]), raw_file)


def test_collect_labels_validates_target_and_prediction_ranges(raw_file, monkeypatch):
    monkeypatch.setattr(
        evaluation.laspy,
        'read',
        lambda path: fake_cloud([1, 3]),
    )
    with pytest.raises(ValueError, match='model has 2 classes'):
        evaluation.collect_labels(FakeSegmenter([0, 1]), raw_file)

    monkeypatch.setattr(
        evaluation.laspy,
        'read',
        lambda path: fake_cloud([1, 2]),
    )
    with pytest.raises(ValueError, match='Predictions outside'):
        evaluation.collect_labels(FakeSegmenter([0, 2]), raw_file)


def test_collect_labels_rejects_missing_files_and_invalid_limit(tmp_path):
    with pytest.raises(FileNotFoundError, match='No LAS or LAZ'):
        evaluation.collect_labels(FakeSegmenter([]), tmp_path)

    raw_file = tmp_path / 'cloud.las'
    raw_file.touch()
    with pytest.raises(ValueError, match='cannot be negative'):
        evaluation.collect_labels(
            FakeSegmenter([]),
            raw_file,
            max_points_per_file=-1,
        )


def test_metrics_use_dense_predictions_once():
    metrics = evaluation.calculate_metrics(
        predictions=np.array([0, 0, 0, 1]),
        targets=np.array([0, 0, 1, 1]),
        num_classes=2,
    )

    assert metrics['accuracy'] == pytest.approx(0.75)
    assert metrics['miou'] == pytest.approx(7.0 / 12.0)
    np.testing.assert_allclose(metrics['class_iou'], [2.0 / 3.0, 0.5])
    np.testing.assert_array_equal(metrics['predictions'], [0, 0, 0, 1])
    np.testing.assert_array_equal(metrics['targets'], [0, 0, 1, 1])


def test_metrics_reject_mismatched_or_empty_arrays():
    with pytest.raises(ValueError, match='same shape'):
        evaluation.calculate_metrics([0], [0, 1], num_classes=2)
    with pytest.raises(ValueError, match='cannot be empty'):
        evaluation.calculate_metrics([], [], num_classes=2)


def test_frontend_creates_confusion_matrix_and_report(tmp_path, monkeypatch):
    segmenter = FakeSegmenter([0, 1])
    plot_dir = tmp_path / 'plots'
    report = {}

    monkeypatch.setattr(
        evaluation,
        'collect_labels',
        lambda *args, **kwargs: (np.array([0, 1]), np.array([0, 1])),
    )
    monkeypatch.setattr(evaluation, 'Plotter', PlotRecorder)
    monkeypatch.setattr(
        evaluation,
        'ClassificationReport',
        lambda **kwargs: report.update(kwargs),
    )

    metrics = evaluation.eval_model_front(
        segmenter=segmenter,
        input_path=tmp_path,
        model_path=pathlib.Path('model.pt'),
        plot_dir=plot_dir,
    )

    assert metrics['accuracy'] == 1.0
    assert plot_dir.is_dir()
    assert PlotRecorder.calls == ['confusion']
    assert 'Accuracy:' in report['additional_info']
    assert 'mIoU:' in report['additional_info']
    assert 'Loss:' not in report['additional_info']


def test_parser_validates_model_name_and_optional_raw_input(raw_file):
    args = evaluation.parser([
        '--model_name', 'RandLANet_1',
        '--input_path', str(raw_file),
    ])

    assert args.model_name == 'RandLANet_1'
    assert args.input_path == raw_file
    assert not hasattr(args, 'device')
    assert not hasattr(args, 'mode')

    with pytest.raises(SystemExit):
        evaluation.parser(['--model_name', 'RandLANet_1.pt'])
