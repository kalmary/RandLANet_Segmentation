import pathlib
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import laspy
import numpy as np
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import array_processing
from src.model_pipeline import EvalSegm_Dense as evaluation


class FixedSegmenter:
    n_classes = 2

    def segment_pcd(self, points, intensity):
        return (np.asarray(intensity) > 0).astype(np.int8)


class RecordingSegmenter(FixedSegmenter):
    def __init__(self):
        self.point_counts = []

    def segment_pcd(self, points, intensity):
        self.point_counts.append(len(points))
        return super().segment_pcd(points, intensity)


class PlotRecorder:
    calls = []

    def __init__(self, class_num, plots_dir):
        self.class_num = class_num
        self.plots_dir = plots_dir

    def cnf_matrix(self, file_name, **kwargs):
        self.calls.append((file_name, kwargs))


def write_las(path, classifications, intensity):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version='1.2')
    cloud = laspy.LasData(header)
    point_count = len(classifications)
    cloud.x = np.arange(point_count)
    cloud.y = np.arange(point_count) + 1
    cloud.z = np.arange(point_count) + 2
    cloud.intensity = np.asarray(intensity, dtype=np.uint16)
    cloud.classification = np.asarray(classifications, dtype=np.uint8)
    cloud.write(path)


class DenseEvaluationTests(unittest.TestCase):
    def setUp(self):
        PlotRecorder.calls = []

    def test_collects_one_prediction_per_classified_source_point(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = pathlib.Path(directory)
            write_las(
                input_dir / 'first.las',
                classifications=[1, 2, 0, 1],
                intensity=[0, 1, 1, 0],
            )
            write_las(
                input_dir / 'nested' / 'second.las',
                classifications=[2, 1],
                intensity=[1, 0],
            )
            write_las(
                input_dir / 'ignored_mod.las',
                classifications=[2, 2],
                intensity=[0, 0],
            )

            predictions, targets = evaluation.collect_labels(
                FixedSegmenter(),
                input_dir,
            )

        np.testing.assert_array_equal(predictions, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(targets, [0, 1, 0, 1, 0])
        self.assertEqual(predictions.dtype, np.int8)
        self.assertEqual(targets.dtype, np.int8)

    def test_samples_at_most_max_points_before_segmenting_each_file(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = pathlib.Path(directory)
            write_las(
                input_dir / 'first.las',
                classifications=[1, 2, 0, 1, 2],
                intensity=[0, 1, 1, 0, 1],
            )
            write_las(
                input_dir / 'second.las',
                classifications=[2, 0],
                intensity=[1, 0],
            )
            segmenter = RecordingSegmenter()

            with patch.object(
                evaluation.np.random,
                'choice',
                return_value=np.array([0, 3]),
            ):
                predictions, targets = evaluation.collect_labels(
                    segmenter,
                    input_dir,
                    max_points_per_file=2,
                )

        self.assertEqual(segmenter.point_counts, [2, 1])
        np.testing.assert_array_equal(predictions, [0, 0, 1])
        np.testing.assert_array_equal(targets, [0, 0, 1])

    def test_rejects_negative_max_points_per_file(self):
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            evaluation.collect_labels(
                FixedSegmenter(),
                pathlib.Path('unused'),
                max_points_per_file=-1,
            )

    def test_metrics_use_hard_labels_only(self):
        metrics = evaluation.calculate_metrics(
            predictions=np.array([0, 0, 0, 1]),
            targets=np.array([0, 0, 1, 1]),
            num_classes=2,
        )

        self.assertAlmostEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['miou'], 7.0 / 12.0)
        np.testing.assert_allclose(metrics['class_iou'], [2.0 / 3.0, 0.5])
        self.assertNotIn('probabilities', metrics)
        self.assertNotIn('loss', metrics)

    def test_frontend_saves_confusion_matrix_and_report_with_old_names(self):
        predictions = np.array([0, 1, 0, 1], dtype=np.int8)
        targets = np.array([0, 1, 1, 1], dtype=np.int8)

        with tempfile.TemporaryDirectory() as directory:
            plot_dir = pathlib.Path(directory) / 'plots'
            report = {}

            with patch.object(
                evaluation,
                'collect_labels',
                return_value=(predictions, targets),
            ), patch.object(evaluation, 'Plotter', PlotRecorder), patch.object(
                evaluation,
                'ClassificationReport',
                side_effect=lambda **kwargs: report.update(kwargs),
            ):
                metrics = evaluation.eval_model_front(
                    segmenter=FixedSegmenter(),
                    input_path=pathlib.Path('unused'),
                    model_path=pathlib.Path('RandLANet_1.pt'),
                    plot_dir=plot_dir,
                )

        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertEqual(len(PlotRecorder.calls), 1)
        self.assertEqual(
            PlotRecorder.calls[0][0],
            'confusion_matrix_RandLANet_1.png',
        )
        self.assertEqual(
            report['file_path'].name,
            'classification_report_RandLANet_1.txt',
        )
        self.assertIn('Accuracy:', report['additional_info'])
        self.assertIn('mIoU:', report['additional_info'])
        self.assertNotIn('Loss:', report['additional_info'])

    def test_segmenter_can_load_config_and_model_from_separate_directories(self):
        config = {
            'model_config': {},
            'num_points': 4,
            'batch_size': 1,
            'num_classes': 2,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config_dir = root / 'dict_files'
            model_dir = root / 'models'

            with patch.object(
                array_processing.SegmentClass,
                '_load_config',
                return_value=config,
            ) as load_config, patch.object(
                array_processing.SegmentClass,
                '_load_model',
                return_value=torch.nn.Identity(),
            ) as load_model:
                array_processing.SegmentClass(
                    model_name='RandLANet_1',
                    config_dir=config_dir,
                    model_dir=model_dir,
                )

        load_config.assert_called_once_with(config_dir)
        load_model.assert_called_once_with(model_dir)

    def test_parser_requires_only_model_name(self):
        args = evaluation.parser(['--model_name', 'RandLANet_1'])

        self.assertEqual(args.model_name, 'RandLANet_1')
        self.assertFalse(hasattr(args, 'input_path'))
        self.assertFalse(hasattr(args, 'verbose'))
        self.assertFalse(hasattr(args, 'device'))

    def test_main_uses_configured_input_path_and_enables_progress(self):
        input_path = pathlib.Path('/configured/raw/test')

        class ConfigSegmenter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.config = {'data_path_test': str(input_path)}

        with patch.object(
            evaluation,
            'parser',
            return_value=Namespace(model_name='RandLANet_1'),
        ), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(
            evaluation,
            'SegmentClass',
            ConfigSegmenter,
        ), patch.object(evaluation, 'eval_model_front') as evaluate:
            evaluation.main()

        segmenter = evaluate.call_args.kwargs['segmenter']
        self.assertTrue(segmenter.kwargs['pbar_bool'])
        self.assertEqual(evaluate.call_args.kwargs['input_path'], input_path)
        self.assertTrue(evaluate.call_args.kwargs['verbose'])


if __name__ == '__main__':
    unittest.main()
