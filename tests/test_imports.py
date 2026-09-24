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
