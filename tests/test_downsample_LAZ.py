import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from src.data_processing import downsample_LAZ


class SaveTilesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.xyz = rng.random((256, 3), dtype=np.float32)
        self.feats = rng.random((256, 1), dtype=np.float32)
        self.labels = rng.integers(0, 10, size=256, dtype=np.int32)

    def save_tile(self, source_path, output_dir):
        tile = (
            self.xyz.copy(),
            self.feats.copy(),
            self.labels.copy(),
            (0, 0),
        )

        with (
            patch.object(
                downsample_LAZ,
                "load_and_normalise",
                return_value=(
                    self.xyz.copy(),
                    self.feats.copy(),
                    self.labels.copy(),
                ),
            ),
            patch.object(
                downsample_LAZ,
                "voxel_subsample_vectorized",
                side_effect=lambda xyz, feats, labels, voxel_size: (
                    xyz,
                    feats,
                    labels,
                ),
            ),
            patch.object(
                downsample_LAZ,
                "iter_tiles",
                return_value=iter((tile,)),
            ),
            patch.object(downsample_LAZ.tqdm, "write"),
        ):
            downsample_LAZ.save_tiles(source_path, output_dir)

    def test_save_tiles_creates_distinct_paired_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "cut"
            first_source = root / "source_a" / "scan.las"
            second_source = root / "source_b" / "scan.las"

            self.save_tile(first_source, output_dir)
            self.save_tile(second_source, output_dir)

            point_files = sorted(output_dir.glob("*.npy"))
            tree_files = sorted(output_dir.glob("*.pkl"))

            self.assertEqual(len(point_files), 2)
            self.assertEqual(len(tree_files), 2)
            self.assertEqual(
                {path.stem for path in point_files},
                {path.stem for path in tree_files},
            )
            self.assertNotEqual(point_files[0].stem, point_files[1].stem)

            expected = np.concatenate(
                [
                    self.xyz,
                    self.feats,
                    self.labels[:, None].astype(np.float32),
                ],
                axis=1,
            )
            saved = np.load(point_files[0])
            np.testing.assert_array_equal(saved, expected)
            self.assertEqual(saved.dtype, np.float32)

            with point_files[0].with_suffix(".pkl").open("rb") as file:
                tree = pickle.load(file)

            self.assertIsInstance(tree, cKDTree)
            _, indices = tree.query(self.xyz[0], k=32)
            self.assertEqual(indices.shape, (32,))
            self.assertIn(0, indices)

    def test_save_tiles_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "cut"
            source_path = root / "source" / "scan.las"

            self.save_tile(source_path, output_dir)

            point_path = next(output_dir.glob("*.npy"))
            tree_path = point_path.with_suffix(".pkl")
            point_data = point_path.read_bytes()
            tree_data = tree_path.read_bytes()

            with self.assertRaises(FileExistsError):
                self.save_tile(source_path, output_dir)

            self.assertEqual(point_path.read_bytes(), point_data)
            self.assertEqual(tree_path.read_bytes(), tree_data)

    def test_split_dataset_keeps_point_clouds_and_trees_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cut_dir = root / "cut"
            output_dir = root / "split"

            self.save_tile(root / "source_a" / "scan.las", cut_dir)
            self.save_tile(root / "source_b" / "scan.las", cut_dir)

            downsample_LAZ.split_dataset(
                cut_dir,
                output_dir,
                train=0.5,
                val=0.0,
                test=0.5,
                seed=42,
            )

            point_files = sorted(output_dir.rglob("*.npy"))
            tree_files = sorted(output_dir.rglob("*.pkl"))

            self.assertEqual(len(point_files), 2)
            self.assertEqual(len(tree_files), 2)
            self.assertEqual(
                {
                    (path.parent.name, path.stem)
                    for path in point_files
                },
                {
                    (path.parent.name, path.stem)
                    for path in tree_files
                },
            )


if __name__ == "__main__":
    unittest.main()
