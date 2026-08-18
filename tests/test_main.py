import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import main


class FixedSegmenter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def segment_pcd(self, points, intensity):
        return np.arange(len(points), dtype=np.int8) % 2


def write_las(path: pathlib.Path, point_count: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version='1.2')
    cloud = laspy.LasData(header)
    cloud.x = np.arange(point_count)
    cloud.y = np.arange(point_count) + 1
    cloud.z = np.arange(point_count) + 2
    cloud.intensity = np.arange(point_count, dtype=np.uint16)
    cloud.classification = np.full(point_count, 7, dtype=np.uint8)
    cloud.write(path)


class MainTests(unittest.TestCase):
    def setUp(self):
        FixedSegmenter.instances = []

    def test_parser_accepts_only_processing_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = pathlib.Path(directory) / 'cloud.las'
            write_las(input_file)

            args = main.argparser([
                '--model_name', 'RandLANet_1',
                '--verbose',
                '--device', 'cpu',
                '--input_path', str(input_file),
                '--output_path', str(pathlib.Path(directory) / 'output'),
            ])

        self.assertEqual(args.model_name, 'RandLANet_1')
        self.assertTrue(args.verbose)
        self.assertEqual(args.device, 'cpu')
        self.assertFalse(hasattr(args, 'mode'))

    def test_single_file_creates_modified_copy_beside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = pathlib.Path(directory) / 'cloud.las'
            write_las(input_file)
            args = main.argparser([
                '--model_name', 'RandLANet_1',
                '--input_path', str(input_file),
            ])

            with patch.object(main, 'SegmentClass', FixedSegmenter):
                outputs = main.process_files(args)

            output_file = pathlib.Path(directory) / 'cloud_mod.las'
            self.assertEqual(outputs, [output_file])
            self.assertTrue(output_file.exists())
            np.testing.assert_array_equal(
                laspy.read(input_file).classification,
                [7, 7, 7, 7],
            )
            np.testing.assert_array_equal(
                laspy.read(output_file).classification,
                [1, 2, 1, 2],
            )

    def test_directory_output_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_dir = root / 'input'
            output_dir = root / 'output'
            write_las(input_dir / 'first.las')
            write_las(input_dir / 'nested' / 'second.las')
            write_las(input_dir / 'ignored_mod.las')
            (input_dir / 'notes.txt').write_text('not a point cloud')

            args = main.argparser([
                '--model_name', 'RandLANet_1',
                '--input_path', str(input_dir),
                '--output_path', str(output_dir),
            ])
            with patch.object(main, 'SegmentClass', FixedSegmenter):
                outputs = main.process_files(args)

            self.assertEqual(outputs, [
                output_dir / 'first_mod.las',
                output_dir / 'nested' / 'second_mod.las',
            ])
            self.assertTrue(outputs[0].exists())
            self.assertTrue(outputs[1].exists())
            self.assertEqual(len(FixedSegmenter.instances), 1)

    def test_cuda_request_fails_when_cuda_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = pathlib.Path(directory) / 'cloud.las'
            write_las(input_file)
            args = main.argparser([
                '--model_name', 'RandLANet_1',
                '--device', 'cuda',
                '--input_path', str(input_file),
            ])

            with patch.object(torch.cuda, 'is_available', return_value=False), \
                    patch.object(main, 'SegmentClass', FixedSegmenter):
                with self.assertRaisesRegex(RuntimeError, 'not available'):
                    main.process_files(args)

            self.assertFalse(FixedSegmenter.instances)


if __name__ == '__main__':
    unittest.main()
