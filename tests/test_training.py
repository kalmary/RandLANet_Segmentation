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

    totals = train_case._dataset_lengths(object(), object(), enabled=True)

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
        "train_repeat": 1,
        "pc_start": 0.3,
        "div_factor": 10,
        "final_div_factor": 100,
    }

    scheduler = train_case._build_scheduler(optimizer, config, total_t=4)

    assert isinstance(scheduler, OneCycleLR)
    assert scheduler.total_steps == 8


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
