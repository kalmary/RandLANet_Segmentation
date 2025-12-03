from typing import Optional, Union
import pathlib as pth
import argparse
import logging
import tqdm

import laspy
import numpy as np

from array_processing import SegmentClass

def iter_files(**kwargs):
    """Iterates over files in a directory and processes them using the SegmentClass instance."""
    input_path = kwargs.get('input_path')
    output_path = kwargs.get('output_path')
    device = kwargs.get('device')
    model_name = kwargs.get('model_name')

    # Create an instance of SegmentClass
    segment_class = SegmentClass(device=device, model_name=model_name)

    # Iterate over files in the input directory
    for file_path in tqdm.tqdm(input_path.glob('*.laz'), desc="Processing files", unit="file"):
        # Process the file using the SegmentClass instance
        segment_class.process_file(file_path, output_path)

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

    return parser.parse_args()

def main():
    args = argparser

    # args to dict
    args_dict = vars(args)
    

    


if __name__ == "__main__":
    main()