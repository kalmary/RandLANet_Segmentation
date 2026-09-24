from typing import Optional, Union
import pathlib as pth
import argparse
import logging
import shutil
import sys
from tqdm import tqdm

import laspy
import numpy as np

import torch

if __package__:
    from .array_processing import SegmentClass
else:
    from array_processing import SegmentClass

def iter_files(args_dict):
    """Iterates over files in a directory and processes them using the SegmentClass instance."""
    input_path = pth.Path(args_dict.get('input_path'))
    output_path_value = args_dict.get('output_path')
    output_path = pth.Path(output_path_value) if output_path_value else None
    if output_path is not None:
        output_path.mkdir(exist_ok=True, parents=True)

    device_name = args_dict.get('device')
    device = torch.device(
        'cuda'
        if device_name in ('cuda', 'gpu') and torch.cuda.is_available()
        else 'cpu'
    )

    extension = args_dict.get('pcd_extension')
    if input_path.is_file():
        file_paths = [input_path] if input_path.suffix == extension else []
    elif input_path.is_dir():
        file_paths = list(input_path.rglob(f'*{extension}'))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args_dict.get('verbose'):
        path_generator = tqdm(
            file_paths,
            total=len(file_paths),
            desc="Processing files",
            unit="file",
            leave=False,
        )
    else:
        path_generator = file_paths

    # Create an instance of SegmentClass
    segment_class = SegmentClass(voxel_size_big=100., 
                                 overlap=0.4,
                                 scaled=True,
                                 model_name=args_dict.get('model_name'),
                                 device=device,
                                 verbose=args_dict.get('verbose'))
    
    for file_path in path_generator:
        if args_dict.get('verbose'):
            path_generator.set_postfix_str(f"Processing {file_path.name}")

        if output_path is not None:
            new_path = output_path.joinpath(f"{file_path.stem}_mod{file_path.suffix}")
        else:
            mod_dir = file_path.parent.joinpath('modified')
            mod_dir.mkdir(exist_ok=True, parents=True)
            new_path = mod_dir.joinpath(f"{file_path.stem}_mod{file_path.suffix}")

        shutil.copy(file_path, new_path)

        laz = laspy.read(new_path)
        points = np.vstack([laz.x, laz.y, laz.z]).T
        intensity = np.asarray(laz.intensity)

        labels = segment_class.segment_pcd(points, intensity)

        laz.classification = labels
        laz.write(new_path)

def run_test_mode(args_dict):
    device_name = args_dict.get('device')
    device = torch.device(
        'cuda'
        if device_name in ('cuda', 'gpu') and torch.cuda.is_available()
        else 'cpu'
    )

    # check if Segment Class loads properly:
    segment_class = SegmentClass(
        voxel_size_big=100.,
        overlap=0.4,
        scaled=True,
        model_name=args_dict.get('model_name'),
        device=device,
        verbose=args_dict.get('verbose'),
    )
    assert segment_class is not None, "Segmentation Class didn't load properly."

    input_path = pth.Path(args_dict.get('input_path'))
    assert input_path.exists(), "Input path with .laz files doesn't exist."

    extension = args_dict.get('pcd_extension')
    if input_path.is_file():
        input_paths = [input_path] if input_path.suffix == extension else []
    else:
        input_paths = list(input_path.rglob(f'*{extension}'))
    assert input_paths, "Input path does not contain supported files."

    print("All asserts passed.")

    

        






def argparser():

    parser = argparse.ArgumentParser(
        description="Script for semantic segmentation of point clouds.\n"
        "Supports .LAZ files (default). ",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help=(
            "Base of the model's name.\n"
            "Use full model name without extension suffix (.pt file expected)"
        )
    )

    # Flag definition
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--input_path',
        type=str,
        help=(
            "Path to the directory with raw input files.\n"
            "Supports .LAZ files by default. "
        )
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='',
        help=(
            "Path to the directory with processed output files.\n"
            "Each file is copied to /output_path/{original_file_name}_mod.laz\n"
            "Files with the '_mod' suffix are the ones being processed.\n"
            "If no output_path is given, 'modified' directory is created in every file's parent directory and _mod file is saved there."
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1], # choice limit
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: test\n'
            '1: process_files'
        )
    )

    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verbose mode.\n"
            "If True, the script will print additional information.\n"
        )
    )

    return parser.parse_args()


def test_main_dispatches_processing_arguments(monkeypatch):
    parsed_args = argparse.Namespace(
        model_name="model",
        device="cpu",
        input_path="input",
        output_path="output",
        mode=1,
        verbose=False,
    )
    received = []

    monkeypatch.setattr(sys.modules[__name__], "argparser", lambda: parsed_args)
    monkeypatch.setattr(sys.modules[__name__], "iter_files", received.append)

    main()

    assert received == [
        {
            "model_name": "model",
            "device": "cpu",
            "input_path": "input",
            "output_path": "output",
            "mode": 1,
            "verbose": False,
            "pcd_extension": ".laz",
        }
    ]


def test_iter_files_processes_each_input_once(tmp_path, monkeypatch):
    input_path = tmp_path.joinpath("input")
    input_path.mkdir()
    source_path = input_path.joinpath("cloud.laz")
    source_path.write_bytes(b"point cloud")
    output_path = tmp_path.joinpath("output")
    writes = []

    class FakeLas:
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        z = np.array([5.0, 6.0])
        intensity = np.array([7, 8])
        classification = None

        def write(self, path):
            writes.append((pth.Path(path), self.classification.copy()))

    class FakeSegmenter:
        def __init__(
            self,
            voxel_size_big,
            overlap,
            scaled,
            model_name,
            device,
            verbose,
        ):
            assert voxel_size_big == 100.0
            assert overlap == 0.4
            assert scaled is True
            assert model_name == "model"
            assert device == torch.device("cpu")
            assert verbose is True

        def segment_pcd(self, points, intensity):
            assert points.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
            assert intensity.tolist() == [7, 8]
            return np.array([2, 4], dtype=np.uint8)

    monkeypatch.setattr(sys.modules[__name__], "SegmentClass", FakeSegmenter)
    monkeypatch.setattr(laspy, "read", lambda path: FakeLas())

    iter_files(
        {
            "model_name": "model",
            "device": "cpu",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "pcd_extension": ".laz",
            "verbose": True,
        }
    )

    assert len(writes) == 1
    written_path, written_labels = writes[0]
    assert written_path == output_path.joinpath("cloud_mod.laz")
    np.testing.assert_array_equal(written_labels, np.array([2, 4], dtype=np.uint8))

def main():
    args = argparser()

    # args to dict
    args_dict = vars(args)
    args_dict['pcd_extension'] = '.laz'

    if args.mode == 0:
        run_test_mode(args_dict)
    else:
        iter_files(args_dict)
    
    

    


if __name__ == "__main__":
    main()
