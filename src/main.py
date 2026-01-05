from typing import Optional, Union
import pathlib as pth
import argparse
import logging
import shutil
from tqdm import tqdm

import laspy
import numpy as np

from array_processing import SegmentClass
import torch

def iter_files(**kwargs):
    """Iterates over files in a directory and processes them using the SegmentClass instance."""
    input_path = pth.Path(kwargs.get('input_path'))
    output_path = str(kwargs.get('output_path'))

    no_output = False
    if len(output_path) == 0:
        no_output = True
    else:
        output_path = pth.Path(output_path)

    output_path.mkdir(exist_ok=True, parents=True)

    device = kwargs.get('device')
    device = torch.device(device) if (device != 'cpu' and torch.cuda.is_available is True) else torch.device('cpu')

    model_name = kwargs.get('model_name')



    path_generator = input_path.rglob(f'*{kwargs.get('pcd_extension')}')
    if kwargs.get('verbose'):
        pbar = tqdm(path_generator, total=len(list(path_generator)), desc="Processing files", unit="file", leave=False)
    else:
        pbar = path_generator

    # Create an instance of SegmentClass
    segment_class = SegmentClass(voxel_size_big=np.array([100., 100.],), 
                                 model_name=model_name,
                                 device=device,
                                 pbar_bool = kwargs.get('verbose'))
    
    for file_path in pbar:
        if kwargs.get('verbose'):
            pbar.set_postfix_str(f"Processing {file_path.name}")

        if not no_output:
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

def test(**kwargs):

    device = kwargs.get('device')
    device = torch.device(device) if (device != 'cpu' and torch.cuda.is_available is True) else torch.device('cpu')
    verbose = kwargs.get('verbose')

    # check if Segment Class loads properly:
    segment_class = None
    try:
        segment_class = SegmentClass(voxel_size_big=np.array([100., 100.]),
                                    model_name=kwargs.get('model_name'),
                                    device=device,
                                    pbar_bool=kwargs.get('verbose'))
    except Exception as e:
        print(e)
    
    assert segment_class is None, "Segmentation Class didn't load properly."

    input_path = pth.Path(kwargs.get(input_path))
    assert not input_path.exists(), "Input path with .laz files doesn't exist."
    assert not input_path.is_dir(), "Input path is not a directory."
    
    num_files = 0
    for file_path in input_path.rglob(f'*{kwargs.get('pcd_extension')}'):
        if file_path.is_file():
            num_files+=1
    assert num_files == 0, "Input path directory is empty."

    

        






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
        type=bool,
        default=True,
        help=(
            "Verbose mode.\n"
            "If True, the script will print additional information.\n"
        )
    )

    return parser.parse_args()

def main():
    args = argparser

    # args to dict
    args_dict = vars(args)
    args_dict['pcd_extension'] = '.laz'

    if args.mode == 0:
        test(args_dict)
    else:
        iter_files(args_dict)
    
    

    


if __name__ == "__main__":
    main()