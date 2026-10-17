import subprocess
import sys
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "loader_name"),
    [
        ("src.data_processing.downsample_laz", "_load_preprocessing_dependencies"),
        ("src.model_pipeline.eval_segm_randlanet", "_load_processing_dependencies"),
        ("src.model_pipeline.train_segm_automated", "_load_training_dependencies"),
    ],
)
def test_dependency_loaders_do_not_modify_import_paths(module_name, loader_name):
    code = f"""
import importlib
import inspect

module = importlib.import_module({module_name!r})
loader_source = inspect.getsource(getattr(module, {loader_name!r}))
assert 'sys.path' not in loader_source
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    'module',
    [
        'array_processing',
        'model_pipeline.train_segm_automated',
        'model_pipeline.eval_segm_randlanet',
        'model_pipeline._train_single_case',
        'model_pipeline._data_loader',
        'data_processing.downsample_laz',
    ],
)
def test_workflows_import_from_parent_with_their_own_utilities(module, tmp_path):
    code = f"""
import importlib

module = importlib.import_module('src.pcd_segmentation.src.{module}')
utilities = importlib.import_module('src.pcd_segmentation.src.utils.nn_utils')
for loader_name in (
    '_load_training_dependencies',
    '_load_processing_dependencies',
    '_load_preprocessing_dependencies',
):
    dependency_loader = getattr(module, loader_name, None)
    if dependency_loader is not None:
        dependency_loader()
for name in ('load_json', 'load_model', 'compute_pos_weights_h5', 'FocalLoss', 'Plotter'):
    if name in vars(module):
        assert getattr(module, name) is getattr(utilities, name), name
"""
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    ["src.final_files.randlanet_cb", "src.model_pipeline.randlanet_cb"],
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
            "from src.pcd_segmentation.src.array_processing import SegmentClass; "
            "assert SegmentClass.__name__ == 'SegmentClass'",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_segment_class_import_does_not_load_training_utilities():
    code = """
import importlib.abc
import sys

class BlockOfflineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'h5py', 'laspy'}:
            raise ImportError(f'Offline dependency imported: {fullname}')

sys.meta_path.insert(0, BlockOfflineImports())
from src.array_processing import SegmentClass

forbidden = {
    'h5py',
    'laspy',
    'matplotlib',
    'optuna',
    'pyvista',
    'seaborn',
    'torchinfo',
    'src.utils.nn_utils.src.accuracy_metrics',
    'src.utils.nn_utils.src.evaluation_plot_tools',
    'src.utils.nn_utils.src.loss_functions',
    'src.utils.nn_utils.src.training_callbacks',
}
loaded = forbidden & sys.modules.keys()
assert not loaded, loaded
assert SegmentClass.__module__ == 'src.array_processing'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_downsample_laz_import_does_not_require_open3d():
    code = """
import builtins

original_import = builtins.__import__

def import_without_open3d(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'open3d':
        raise ImportError('open3d is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_open3d
from src.data_processing import downsample_laz
assert callable(downsample_laz.decimate_chunk_laz)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "arguments",
    [["src/main.py", "--help"], ["-m", "src.main", "--help"]],
)
def test_parent_project_main_imports_segmenter(arguments):
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--segm-model-name" in result.stdout


def test_array_processing_main_uses_explicit_input():
    code = """
import laspy
import runpy
import sys
from pathlib import Path

script = Path("src/array_processing.py").resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), "--input-path", "sample.laz"]

def stop_before_sample_input(path):
    assert path == Path("sample.laz")
    raise RuntimeError("sample input reached")

laspy.read = stop_before_sample_input
try:
    runpy.run_path(str(script), run_name="__main__")
except RuntimeError as error:
    assert str(error) == "sample input reached", str(error)
else:
    raise AssertionError("sample input was not reached")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
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
    if name == "final_files.randlanet_cb" and level == 1:
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
