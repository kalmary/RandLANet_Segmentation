import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


project_root = pathlib.Path(__file__).resolve().parents[1]
model_pipeline_dir = project_root / "src" / "model_pipeline"
sys.path.append(str(project_root))
sys.path.append(str(model_pipeline_dir))

from src.model_pipeline import TrainSegmAutomated as training
from src.model_pipeline import _train_single_case as train_case


class CompilableModel:
    def eval(self):
        return self


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        return self.linear(inputs)


class Progress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values):
        pass


def test_parser_has_no_device_selection():
    args = training.argparser([
        "--model_name", "RandLANet_1",
        "--mode", "2",
    ])

    assert args.model_name == "RandLANet_1"
    assert args.mode == 2
    assert not hasattr(args, "device")


def test_load_config_requires_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        training.load_config(tmp_path, mode=1)


def test_epoch_lengths_have_practical_fixed_defaults():
    assert train_case._configured_epoch_lengths({}) == (500, 100)
    assert train_case._configured_epoch_lengths({
        "train_steps_per_epoch": 12,
        "validation_steps": 3,
    }) == (12, 3)


@pytest.mark.parametrize(
    "config",
    [
        {"train_steps_per_epoch": 0},
        {"validation_steps": -1},
        {"train_steps_per_epoch": True},
    ],
)
def test_epoch_lengths_must_be_positive_integers(config):
    with pytest.raises(ValueError, match="positive integers"):
        train_case._configured_epoch_lengths(config)


def test_unknown_dataset_length_uses_plateau_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    scheduler = train_case._build_scheduler(optimizer, {}, total_t=None)

    assert isinstance(scheduler, ReduceLROnPlateau)


def test_known_dataset_length_uses_one_cycle_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    config = {
        "learning_rate": 0.01,
        "epochs": 2,
        "train_repeat": 3,
        "pc_start": 0.3,
        "div_factor": 10,
        "final_div_factor": 100,
    }

    scheduler = train_case._build_scheduler(optimizer, config, total_t=4)

    assert isinstance(scheduler, OneCycleLR)
    assert scheduler.total_steps == 8


def test_context_only_labels_are_excluded_from_loss_inputs():
    outputs = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]])
    labels = torch.tensor([[0, -1, 1]])

    logits, targets = train_case._labeled_outputs(outputs, labels)

    torch.testing.assert_close(logits, torch.tensor([[2.0, 5.0], [4.0, 7.0]]))
    torch.testing.assert_close(targets, torch.tensor([0, 1]))


def test_class_weights_ignore_context_only_points(tmp_path):
    data = np.zeros((5, 5), dtype=np.float32)
    data[:, 4] = [-1, -1, 0, 0, 1]
    np.save(tmp_path / "tile.npy", data)

    weights = train_case.compute_pos_weights_cloud(tmp_path, 2, power=1.0)

    torch.testing.assert_close(weights, torch.tensor([0.5, 1.0]))


def test_normalized_focal_loss_is_independent_of_global_alpha_scale():
    logits = torch.zeros((4, 2))
    targets = torch.tensor([0, 0, 1, 1])
    first = train_case.NormalizedFocalLoss(
        alpha=torch.tensor([0.02, 1.0]), gamma=1.0, smoothing=0.0
    )
    second = train_case.NormalizedFocalLoss(
        alpha=torch.tensor([0.2, 10.0]), gamma=1.0, smoothing=0.0
    )

    first_loss = first(logits, targets)
    second_loss = second(logits, targets)

    torch.testing.assert_close(first_loss, second_loss)
    assert first_loss.item() == pytest.approx(0.5 * np.log(2))


@pytest.mark.parametrize("use_provided_model", [False, True])
def test_repeats_are_independent_and_use_updated_loader_api(
    monkeypatch,
    use_provided_model,
):
    loader_calls = []
    weight_calls = []
    loss_instances = []
    created_models = []
    optimizers = []
    scheduler_calls = []

    batch = (torch.tensor([[1.0, -1.0]]), torch.tensor([0]))
    train_loader = [batch]
    val_loader = [batch]

    def make_loader(**kwargs):
        loader_calls.append(kwargs)
        loader = train_loader if kwargs["shuffle"] else val_loader
        return loader, object()

    def compute_weights(data_dir, num_classes, power):
        weight_calls.append((data_dir, num_classes, power))
        return torch.tensor([1.0, 2.0])

    class TrackingFocalLoss(torch.nn.Module):
        def __init__(self, alpha, gamma, smoothing, reduction):
            super().__init__()
            self.alpha = alpha.clone()
            loss_instances.append(self)

        def forward(self, outputs, labels):
            return torch.nn.functional.cross_entropy(
                outputs, labels, reduction="none"
            )

    def make_model(model_config, n_classes):
        model = TinyModel()
        created_models.append(model)
        return model

    real_adamw = train_case.optim.AdamW

    def make_optimizer(parameters, **kwargs):
        optimizer = real_adamw(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def make_scheduler(optimizer, training_dict, total_t):
        scheduler = object()
        scheduler_calls.append((optimizer, total_t, scheduler))
        return scheduler

    monkeypatch.setattr(train_case, "make_loader", make_loader)
    monkeypatch.setattr(train_case, "compute_pos_weights_cloud", compute_weights)
    monkeypatch.setattr(train_case, "FocalLoss", TrackingFocalLoss)
    monkeypatch.setattr(train_case, "RandLANet", make_model)
    monkeypatch.setattr(train_case.optim, "AdamW", make_optimizer)
    monkeypatch.setattr(train_case, "_build_scheduler", make_scheduler)
    monkeypatch.setattr(train_case, "tqdm", Progress)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)

    provided_model = TinyModel() if use_provided_model else None
    config = {
        "data_path_train": "train-data",
        "data_path_val": "val-data",
        "num_classes": 2,
        "num_points": 16,
        "batch_size": 1,
        "query_workers": 0,
        "model": provided_model,
        "model_config": {},
        "device": "cpu",
        "focal_loss_gamma": 2.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "epochs": 1,
        "train_repeat": 2,
    }

    results = list(train_case.train_model(config))
    models = [model for model, _ in results]
    histories = [history for _, history in results]

    assert len(results) == 2
    assert models[0] is not models[1]
    if use_provided_model:
        assert created_models == []
        assert all(model is not provided_model for model in models)
    else:
        assert models == created_models

    assert len(optimizers) == 2
    assert optimizers[0] is not optimizers[1]
    assert optimizers[0].param_groups[0]["params"][0] is next(models[0].parameters())
    assert optimizers[1].param_groups[0]["params"][0] is next(models[1].parameters())
    assert [call[:2] for call in scheduler_calls] == [
        (optimizers[0], 500),
        (optimizers[1], 500),
    ]

    assert len(loss_instances) == 4
    assert len({id(loss) for loss in loss_instances}) == 4
    assert all(torch.equal(loss.alpha, torch.tensor([1.0, 2.0])) for loss in loss_instances)
    assert weight_calls == [
        ("train-data", 2, 0.5),
        ("val-data", 2, 0.5),
    ]

    assert loader_calls[0]["data_dir"] == "train-data"
    assert loader_calls[0]["shuffle"] is True
    assert loader_calls[0]["max_seen"] == 3
    assert loader_calls[0]["steps_per_epoch"] == 500
    assert loader_calls[0]["batches_per_file"] == 8
    assert loader_calls[0]["class_aware_fraction"] == 0.5
    assert loader_calls[0]["class_weights"].tolist() == [1.0, 2.0]
    assert loader_calls[1] == {
        "data_dir": "val-data",
        "num_points": 16,
        "batch_size": 1,
        "query_workers": 0,
        "shuffle": False,
        "max_seen": 1,
        "steps_per_epoch": 100,
        "batches_per_file": 8,
        "class_aware_fraction": 0.0,
    }
    assert all(
        history["acc_v_hist"][0] in {0.0, 1.0}
        for history in histories
    )
    assert all(
        history["miou_v_hist"][0] == history["acc_v_hist"][0]
        for history in histories
    )
    assert [len(history["loss_hist"]) for history in histories] == [1, 1]
    assert histories[0]["loss_hist"] is not histories[1]["loss_hist"]


def test_train_model_propagates_missing_dataset_error(tmp_path):
    config = {
        "data_path_train": tmp_path / "missing",
        "num_classes": 2,
    }

    with pytest.raises(FileNotFoundError, match="does not exist"):
        next(training.train_model(config))


def test_test_case_accepts_two_value_training_results(monkeypatch):
    result_hist = {"loss_hist": [1.0]}
    monkeypatch.setattr(
        training,
        "train_model",
        lambda training_dict: iter([(object(), result_hist)]),
    )

    training.test_case({})


def test_test_case_propagates_failed_training(monkeypatch):
    monkeypatch.setattr(
        training,
        "train_model",
        lambda training_dict: iter([(None, {})]),
    )

    with pytest.raises(RuntimeError, match="no model or result history"):
        training.test_case({})


def test_check_models_returns_only_successful_config_paths(monkeypatch):
    paths = [
        pathlib.Path("first.json"),
        pathlib.Path("invalid.json"),
        pathlib.Path("third.json"),
    ]
    monkeypatch.setattr(
        training,
        "load_json",
        lambda path: {"name": path.stem},
    )
    monkeypatch.setattr(training, "convert_str_values", lambda config: config)

    def make_model(config, num_classes):
        if config["name"] == "invalid":
            raise ValueError("invalid model")
        return CompilableModel()

    monkeypatch.setattr(training, "RandLANet", make_model)
    monkeypatch.setattr(
        training,
        "summary",
        lambda *args, **kwargs: SimpleNamespace(
            total_param_bytes=1024,
            total_output_bytes=1024,
        ),
    )

    configs, valid_paths = training.check_models(paths)

    assert [config["name"] for config in configs] == ["first", "third"]
    assert valid_paths == [paths[0], paths[2]]
