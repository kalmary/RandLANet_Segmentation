import pathlib
import sys

import numpy as np
import pytest
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.model_pipeline import EvalSegm_RandLANet as evaluation


class FixedModel(torch.nn.Module):
    def forward(self, points):
        scores = points[..., 0]
        return torch.stack((-scores, scores), dim=1)


class PlotRecorder:
    calls = []

    def __init__(self, class_num, plots_dir):
        self.class_num = class_num
        self.plots_dir = plots_dir

    def cnf_matrix(self, *args, **kwargs):
        self.calls.append("confusion")

    def prc_curve(self, *args, **kwargs):
        self.calls.append("precision_recall")

    def roc_curve(self, *args, **kwargs):
        self.calls.append("roc")


@pytest.fixture(autouse=True)
def reset_plot_recorder():
    PlotRecorder.calls = []


def test_inference_collects_outputs_before_metrics(monkeypatch):
    batches = [
        (
            torch.tensor([[[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 1]]),
        ),
        (
            torch.tensor([[[3.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[1, 0]]),
        ),
    ]
    config = {
        "data_path_test": "unused",
        "num_classes": 2,
        "num_points": 2,
        "batch_size": 1,
        "query_workers": 3,
        "max_seen": 4,
        "device": torch.device("cpu"),
    }
    weights = torch.tensor([0.5, 1.0])
    loader_calls = []

    def make_loader(**kwargs):
        loader_calls.append(kwargs)
        return batches, object()

    monkeypatch.setattr(
        evaluation,
        "compute_pos_weights",
        lambda **kwargs: weights,
    )
    monkeypatch.setattr(evaluation, "make_loader", make_loader)

    outputs, labels, result_weights = evaluation._eval_model(config, FixedModel())

    assert outputs.shape == (2, 2, 2)
    assert torch.equal(labels, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(result_weights, weights)
    assert len(loader_calls) == 1
    loader_args = loader_calls[0]
    np.testing.assert_array_equal(
        loader_args.pop("pos_weights"),
        np.array([0.5, 1.0], dtype=np.float32),
    )
    assert loader_args == {
        "data_dir": "unused",
        "num_points": 2,
        "batch_size": 1,
        "query_workers": 3,
        "shuffle": False,
        "max_seen": 4,
    }


def test_parser_does_not_offer_an_unused_device():
    args = evaluation.parser(["--model_name", "RandLANet_1"])

    assert not hasattr(args, "device")


def test_metrics_use_all_points_and_supplied_weights():
    outputs = torch.tensor([
        [
            [4.0, 4.0, 4.0, -4.0],
            [-4.0, -4.0, -4.0, 4.0],
        ]
    ])
    labels = torch.tensor([[0, 0, 1, 1]])
    weights = torch.tensor([0.5, 1.0])

    metrics = evaluation.calculate_metrics(
        outputs=outputs,
        labels=labels,
        class_weights=weights,
        num_classes=2,
        focal_loss_gamma=1.0,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["weighted_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["miou"] == pytest.approx(7.0 / 12.0)
    np.testing.assert_allclose(metrics["class_iou"], [2.0 / 3.0, 0.5])
    assert metrics["probabilities"].shape == (4, 2)
    np.testing.assert_array_equal(metrics["labels"], [0, 0, 1, 1])
    np.testing.assert_array_equal(metrics["predictions"], [0, 0, 0, 1])


def test_frontend_finalizes_metrics_then_creates_all_outputs(tmp_path, monkeypatch):
    outputs = torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]])
    labels = torch.tensor([[0, 1]])
    weights = torch.tensor([0.5, 1.0])
    config = {
        "num_classes": 2,
        "focal_loss_gamma": 1.0,
    }
    plot_dir = tmp_path / "plots"
    report = {}

    def record_report(**kwargs):
        report.update(kwargs)

    monkeypatch.setattr(
        evaluation,
        "_eval_model",
        lambda config_dict, model: (outputs, labels, weights),
    )
    monkeypatch.setattr(evaluation, "Plotter", PlotRecorder)
    monkeypatch.setattr(evaluation, "ClassificationReport", record_report)

    evaluation.eval_model_front(
        config_dict=config,
        model=FixedModel(),
        paths=[pathlib.Path("model_config.json"), plot_dir],
    )

    assert plot_dir.is_dir()
    assert PlotRecorder.calls == ["confusion", "precision_recall", "roc"]
    assert "Accuracy:" in report["additional_info"]
    assert "Weighted accuracy:" in report["additional_info"]
    assert "mIoU:" in report["additional_info"]
