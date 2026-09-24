import subprocess
import sys
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
