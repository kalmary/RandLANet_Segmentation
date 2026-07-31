import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.model_pipeline import EvalSegm_RandLANet as evaluation


class FixedModel(torch.nn.Module):
    def forward(self, points):
        scores = points[..., 0]
        return torch.stack((-scores, scores), dim=1)


class PlotRecorder:
    calls = []

    def __init__(self, class_num, plots_dir):
        self.class_num = class_num
        self.plots_dir = plots_dir

    def cnf_matrix(self, *args, **kwargs):
        self.calls.append('confusion')

    def prc_curve(self, *args, **kwargs):
        self.calls.append('precision_recall')

    def roc_curve(self, *args, **kwargs):
        self.calls.append('roc')


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        PlotRecorder.calls = []

    def test_inference_collects_outputs_before_metrics(self):
        batches = [
            (
                torch.tensor([[[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]),
                torch.tensor([[0, 1]]),
            ),
            (
                torch.tensor([[[3.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]]),
                torch.tensor([[1, 0]]),
            ),
        ]
        config = {
            'data_path_test': 'unused',
            'num_classes': 2,
            'num_points': 2,
            'batch_size': 1,
            'query_workers': 3,
            'max_seen': 4,
            'device': torch.device('cpu'),
        }
        weights = torch.tensor([0.5, 1.0])

        with patch.object(evaluation, 'compute_pos_weights', return_value=weights), \
                patch.object(evaluation, 'make_loader', return_value=(batches, object())) as make_loader:
            outputs, labels, result_weights = evaluation._eval_model(config, FixedModel())

        self.assertEqual(outputs.shape, (2, 2, 2))
        self.assertTrue(torch.equal(labels, torch.tensor([[0, 1], [1, 0]])))
        self.assertTrue(torch.equal(result_weights, weights))
        make_loader.assert_called_once()
        loader_args = make_loader.call_args.kwargs
        np.testing.assert_array_equal(
            loader_args.pop('pos_weights'),
            np.array([0.5, 1.0], dtype=np.float32),
        )
        self.assertEqual(loader_args, {
            'data_dir': 'unused',
            'num_points': 2,
            'batch_size': 1,
            'query_workers': 3,
            'shuffle': False,
            'max_seen': 4,
        })

    def test_parser_does_not_offer_an_unused_device(self):
        args = evaluation.parser(['--model_name', 'RandLANet_1'])

        self.assertFalse(hasattr(args, 'device'))

    def test_metrics_use_all_points_and_supplied_weights(self):
        outputs = torch.tensor([
            [
                [4.0, 4.0, 4.0, -4.0],
                [-4.0, -4.0, -4.0, 4.0],
            ]
        ])
        labels = torch.tensor([[0, 0, 1, 1]])
        weights = torch.tensor([0.5, 1.0])

        metrics = evaluation.calculate_metrics(
            outputs=outputs,
            labels=labels,
            class_weights=weights,
            num_classes=2,
            focal_loss_gamma=1.0,
        )

        self.assertAlmostEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['weighted_accuracy'], 2.0 / 3.0)
        self.assertAlmostEqual(metrics['miou'], 7.0 / 12.0)
        np.testing.assert_allclose(metrics['class_iou'], [2.0 / 3.0, 0.5])
        self.assertEqual(metrics['probabilities'].shape, (4, 2))
        np.testing.assert_array_equal(metrics['labels'], [0, 0, 1, 1])
        np.testing.assert_array_equal(metrics['predictions'], [0, 0, 0, 1])

    def test_frontend_finalizes_metrics_then_creates_all_outputs(self):
        outputs = torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]])
        labels = torch.tensor([[0, 1]])
        weights = torch.tensor([0.5, 1.0])
        config = {
            'num_classes': 2,
            'focal_loss_gamma': 1.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            plot_dir = pathlib.Path(directory) / 'plots'
            report = {}

            def record_report(**kwargs):
                report.update(kwargs)

            with patch.object(
                evaluation,
                '_eval_model',
                return_value=(outputs, labels, weights),
            ), patch.object(evaluation, 'Plotter', PlotRecorder), patch.object(
                evaluation,
                'ClassificationReport',
                side_effect=record_report,
            ):
                evaluation.eval_model_front(
                    config_dict=config,
                    model=FixedModel(),
                    paths=[pathlib.Path('model_config.json'), plot_dir],
                )

            self.assertTrue(plot_dir.is_dir())
            self.assertEqual(
                PlotRecorder.calls,
                ['confusion', 'precision_recall', 'roc'],
            )
            self.assertIn('Accuracy:', report['additional_info'])
            self.assertIn('Weighted accuracy:', report['additional_info'])
            self.assertIn('mIoU:', report['additional_info'])


if __name__ == '__main__':
    unittest.main()
