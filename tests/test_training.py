import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
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


def test_dataset_length_measurement_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        train_case,
        "get_dataset_len",
        lambda loader: pytest.fail("loader should not be iterated"),
    )

    totals = train_case._dataset_lengths(object(), object(), enabled=False)

    assert totals == (None, None)


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


@pytest.mark.parametrize("use_provided_model", [False, True])
def test_repeats_are_independent_and_use_updated_loader_api(
    monkeypatch,
    use_provided_model,
):
    loader_calls = []
    length_calls = []
    weight_calls = []
    loss_instances = []
    accuracy_calls = []
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

    def get_dataset_len(loader):
        length_calls.append(loader)
        return len(loader)

    def compute_weights(data_dir, num_classes, power):
        weight_calls.append((data_dir, num_classes, power))
        return torch.tensor([1.0, 2.0])

    class TrackingFocalLoss(torch.nn.Module):
        def __init__(self, alpha, gamma, smoothing, reduction):
            super().__init__()
            self.alpha = alpha.clone()
            loss_instances.append(self)

        def forward(self, outputs, labels):
            return torch.nn.functional.cross_entropy(outputs, labels)

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

    def calculate_accuracy(outputs, labels):
        accuracy_calls.append((outputs, labels))
        return 0.75

    monkeypatch.setattr(train_case, "make_loader", make_loader)
    monkeypatch.setattr(train_case, "get_dataset_len", get_dataset_len)
    monkeypatch.setattr(train_case, "compute_pos_weights_prob", compute_weights)
    monkeypatch.setattr(train_case, "FocalLoss", TrackingFocalLoss)
    monkeypatch.setattr(train_case, "RandLANet", make_model)
    monkeypatch.setattr(train_case.optim, "AdamW", make_optimizer)
    monkeypatch.setattr(train_case, "_build_scheduler", make_scheduler)
    monkeypatch.setattr(train_case, "calculate_accuracy", calculate_accuracy)
    monkeypatch.setattr(train_case, "compute_mIoU", lambda *args: (0.25, None))
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
        (optimizers[0], 1),
        (optimizers[1], 1),
    ]

    assert len(loss_instances) == 4
    assert len({id(loss) for loss in loss_instances}) == 4
    assert all(torch.equal(loss.alpha, torch.tensor([1.0, 2.0])) for loss in loss_instances)
    assert weight_calls == [
        ("train-data", 2, 0.5),
        ("val-data", 2, 0.5),
    ]

    assert loader_calls == [
        {
            "data_dir": "train-data",
            "num_points": 16,
            "batch_size": 1,
            "query_workers": 0,
            "shuffle": True,
        },
        {
            "data_dir": "val-data",
            "num_points": 16,
            "batch_size": 1,
            "query_workers": 0,
            "shuffle": False,
        },
    ]
    assert length_calls == [train_loader, val_loader]
    assert len(accuracy_calls) == 2
    assert [history["acc_v_hist"] for history in histories] == [[0.75], [0.75]]
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
