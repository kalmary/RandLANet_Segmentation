import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    ["src.final_files.RandLANet_CB", "src.model_pipeline.RandLANet_CB"],
)
def test_model_import_does_not_require_torchinfo(module):
    code = """
import builtins
import importlib

original_import = builtins.__import__

def import_without_torchinfo(name, *args, **kwargs):
    if name == "torchinfo":
        raise ImportError("torchinfo is not installed")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torchinfo
importlib.import_module("MODULE")
""".replace("MODULE", module)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_segment_class_imports_from_parent_project_root():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.PCDSegmentation.src.array_processing import SegmentClass; "
            "assert SegmentClass.__name__ == 'SegmentClass'",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_array_processing_preserves_model_import_error():
    code = """
import builtins
import importlib

original_import = builtins.__import__

def import_with_model_error(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "final_files.RandLANet_CB" and level == 1:
        raise ImportError("model dependency failed")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_with_model_error
try:
    importlib.import_module("src.array_processing")
except ImportError as error:
    assert str(error) == "model dependency failed", str(error)
else:
    raise AssertionError("model import unexpectedly succeeded")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
