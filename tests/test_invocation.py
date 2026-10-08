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
    assert "--model-name" in result.stdout
    assert "--input-path" in result.stdout
    assert "--output-path" in result.stdout


def test_main_parser_preserves_defaults():
    code = """
import sys

sys.argv = ['main.py']
from src.main import argparser

args = argparser()
assert args.device == 'cpu'
assert args.output_path == ''
assert args.mode == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_main_help_does_not_import_processing_dependencies():
    code = """
import importlib.abc
import sys

class BlockProcessingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'laspy', 'torch'} or fullname == 'src.array_processing':
            raise ImportError(f'Processing dependency imported: {fullname}')

sys.meta_path.insert(0, BlockProcessingImports())
sys.argv = ['main.py', '--help']
from src.main import main

try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('--help did not exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-name" in result.stdout


def test_training_help_does_not_import_training_dependencies():
    code = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'matplotlib', 'optuna', 'torch', 'torchinfo'}:
            raise ImportError(f'Training dependency imported: {fullname}')

sys.meta_path.insert(0, BlockTrainingImports())
sys.argv = ['train_segm_automated.py', '--help']
from src.model_pipeline.train_segm_automated import main

try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('--help did not exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-name" in result.stdout


def test_evaluation_help_does_not_import_processing_dependencies():
    code = """
import importlib.abc
import sys

class BlockProcessingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'laspy', 'torch'} or fullname == 'src.array_processing':
            raise ImportError(f'Processing dependency imported: {fullname}')

sys.meta_path.insert(0, BlockProcessingImports())
sys.argv = ['eval_segm_randlanet.py', '--help']
from src.model_pipeline.eval_segm_randlanet import main

try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('--help did not exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-name" in result.stdout


@pytest.mark.parametrize(
    ("entry", "option"),
    [
        (["src/data_processing/downsample_laz.py"], "--source-path"),
        (["-m", "src.data_processing.downsample_laz"], "--source-path"),
        (["src/model_pipeline/train_segm_automated.py"], "--model-name"),
        (["-m", "src.model_pipeline.train_segm_automated"], "--model-name"),
        (["src/model_pipeline/eval_segm_randlanet.py"], "--model-name"),
        (["-m", "src.model_pipeline.eval_segm_randlanet"], "--model-name"),
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
