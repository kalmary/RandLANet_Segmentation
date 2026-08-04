import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
model_pipeline_dir = project_root / 'src' / 'model_pipeline'
sys.path.append(str(project_root))
sys.path.append(str(model_pipeline_dir))

from src.model_pipeline import TrainSegmAutomated as training


class TrainingAutomationTests(unittest.TestCase):
    def test_checkpoint_returns_three_values_on_both_paths(self):
        model = torch.nn.Linear(1, 1)
        config = {
            'device': torch.device('cuda'),
            'num_classes': 2,
        }
        history = {
            'acc_hist': [0.5],
            'loss_hist': [1.0],
            'loss_v_hist': [1.0],
            'acc_v_hist': [0.5],
            'miou_hist': [0.25],
            'miou_v_hist': [0.25],
        }

        with tempfile.TemporaryDirectory() as directory:
            module_path = pathlib.Path(directory) / 'TrainSegmAutomated.py'
            checkpoint = training.Checkpoint(existing_ok=False)

            with patch.object(training, '__file__', str(module_path)), patch.object(
                training,
                'save_model',
                side_effect=lambda path, model, existing_ok: path,
            ), patch.object(training, 'save2json'):
                saved = checkpoint.check_checkpoint(
                    model,
                    'RandLANet_1',
                    0.5,
                    config.copy(),
                    history,
                )
                unchanged = checkpoint.check_checkpoint(
                    model,
                    'RandLANet_1',
                    0.4,
                    config.copy(),
                    history,
                )

        self.assertIsInstance(saved, tuple)
        self.assertIsInstance(unchanged, tuple)
        self.assertEqual(len(saved), 3)
        self.assertEqual(len(unchanged), 3)

    def test_case_training_accepts_checkpoint_contract(self):
        model = torch.nn.Linear(1, 1)
        config = {'case': 'single'}
        history = {
            'acc_v_hist': [0.5],
            'loss_v_hist': [1.0],
        }
        checkpoint = Mock()

        with tempfile.TemporaryDirectory() as directory:
            module_path = pathlib.Path(directory) / 'TrainSegmAutomated.py'
            config_path = pathlib.Path(directory) / 'config.json'
            checkpoint.check_checkpoint.return_value = (
                model,
                config,
                config_path,
            )

            with patch.object(training, '__file__', str(module_path)), patch.object(
                training,
                'Checkpoint',
                return_value=checkpoint,
            ), patch.object(
                training,
                'train_model',
                return_value=iter(((model, history),)),
            ), patch.object(training, 'summary'):
                training.case_based_training([config], 'RandLANet_1')

        checkpoint.check_checkpoint.assert_called_once_with(
            model,
            'RandLANet_1',
            0.5,
            config,
            history,
        )


if __name__ == '__main__':
    unittest.main()
