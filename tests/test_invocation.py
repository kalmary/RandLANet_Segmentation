import subprocess
import sys
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("entry", [["src/main.py"], ["-m", "src.main"]])
def test_main_help_works_from_project_root(entry):
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model_name" in result.stdout
    assert "--input_path" in result.stdout
    assert "--output_path" in result.stdout


def test_main_parser_preserves_defaults(monkeypatch):
    from src.main import argparser

    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = argparser()

    assert args.device == "cpu"
    assert args.output_path == ""
    assert args.mode == 0


@pytest.mark.parametrize(
    ("entry", "option"),
    [
        (["src/data_processing/downsample_LAZ.py"], "--source_path"),
        (["-m", "src.data_processing.downsample_LAZ"], "--source_path"),
        (["src/model_pipeline/TrainSegmAutomated.py"], "--model_name"),
        (["-m", "src.model_pipeline.TrainSegmAutomated"], "--model_name"),
        (["src/model_pipeline/EvalSegm_RandLANet.py"], "--model_name"),
        (["-m", "src.model_pipeline.EvalSegm_RandLANet"], "--model_name"),
    ],
)
def test_workflow_help_works_in_both_invocation_forms(entry, option, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert option in result.stdout
