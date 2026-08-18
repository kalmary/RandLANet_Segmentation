import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from src.model_pipeline import _data_loader


class CustomDatasetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.xyz = rng.random((24, 3), dtype=np.float32)
        self.feats = rng.random((24, 1), dtype=np.float32)
        self.labels = np.repeat(
            np.arange(2, dtype=np.int32),
            12,
        )

    def save_tile(self, directory):
        path = Path(directory) / "scan_source_tile_000_000.npy"
        data = np.concatenate(
            (
                self.xyz,
                self.feats,
                self.labels[:, None].astype(np.float32),
            ),
            axis=1,
        )
        np.save(path, data)

        tree = cKDTree(self.xyz, leafsize=4)
        with path.with_suffix(".pkl").open("wb") as file:
            pickle.dump(
                tree,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return path

    def make_dataset(self, directory, **kwargs):
        return _data_loader.CustomDataset(
            data_dir=directory,
            num_points=8,
            batch_size=3,
            query_workers=2,
            shuffle=False,
            **kwargs,
        )

    def test_class_weights_set_required_views(self):
        with tempfile.TemporaryDirectory() as directory:
            self.save_tile(directory)

            dataset = self.make_dataset(
                directory,
                pos_weights=np.array(
                    [0.4, 0.6, 1.0],
                    dtype=np.float32,
                ),
                max_seen=5,
            )
            np.testing.assert_array_equal(
                dataset.class_targets,
                np.array([2, 3, 5], dtype=np.int8),
            )

            equal_dataset = self.make_dataset(
                directory,
                pos_weights=np.ones(3, dtype=np.float32),
                max_seen=5,
            )
            np.testing.assert_array_equal(
                equal_dataset.class_targets,
                np.array([5, 5, 5], dtype=np.int8),
            )

    def test_sampling_reaches_each_points_required_views(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_tile(directory)
            dataset = self.make_dataset(
                directory,
                pos_weights=np.array([0.1, 1.0], dtype=np.float32),
                max_seen=4,
            )

            iterator = dataset._iter_file(
                path,
                np.random.default_rng(4),
            )
            batches = []

            while True:
                try:
                    batch, labels = next(iterator)
                except StopIteration as result:
                    seen = result.value
                    break

                batches.append((batch, labels))
                self.assertEqual(batch.shape[1:], (8, 4))
                self.assertEqual(labels.shape[1:], (8,))
                np.testing.assert_array_equal(
                    batch[:, 0, :3].numpy(),
                    np.zeros((batch.shape[0], 3), dtype=np.float32),
                )

            self.assertTrue(batches)
            targets = dataset._point_targets(self.labels)
            self.assertTrue(np.all(seen >= targets))
            self.assertEqual(seen.dtype, np.int8)
            self.assertEqual(seen.shape, (len(self.xyz), 1))

    def test_torch_workers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.save_tile(directory)
            dataset = self.make_dataset(directory, max_seen=1)

            with patch(
                "torch.utils.data.get_worker_info",
                return_value=SimpleNamespace(num_workers=2),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "exactly one",
                ):
                    next(iter(dataset))

    def test_make_loader_uses_one_persistent_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.save_tile(directory)

            loader, dataset = _data_loader.make_loader(
                data_dir=directory,
                num_points=8,
                batch_size=3,
                query_workers=2,
                shuffle=False,
                pos_weights=np.ones(2, dtype=np.float32),
                max_seen=1,
            )

            self.assertEqual(loader.num_workers, 1)
            self.assertTrue(loader.persistent_workers)
            self.assertEqual(loader.prefetch_factor, 2)
            self.assertFalse(loader.pin_memory)
            self.assertEqual(dataset.query_workers, 2)

            first_epoch = list(loader)
            second_epoch = list(loader)

            batch, labels = first_epoch[0]
            self.assertEqual(batch.shape, (3, 8, 4))
            self.assertEqual(labels.shape, (3, 8))

            first_centers = np.concatenate(
                [batch[:, 0, 3].numpy() for batch, _ in first_epoch]
            )
            second_centers = np.concatenate(
                [batch[:, 0, 3].numpy() for batch, _ in second_epoch]
            )
            self.assertFalse(
                np.array_equal(first_centers, second_centers)
            )


if __name__ == "__main__":
    unittest.main()
