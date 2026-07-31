import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import array_processing


class IntensityModel(torch.nn.Module):
    def forward(self, inputs):
        intensity = inputs[..., 3]
        return torch.stack((1.0 - intensity, intensity), dim=1) * 8.0


class ArrayProcessingTests(unittest.TestCase):
    @staticmethod
    def make_segmenter():
        segmenter = array_processing.SegmentClass.__new__(
            array_processing.SegmentClass
        )
        segmenter.device = torch.device('cpu')
        segmenter.num_points = 3
        segmenter.batch_size = 2
        segmenter.n_classes = 2
        segmenter.n_seen = 2
        segmenter.query_workers = 1
        segmenter.pbar_bool = False
        segmenter.voxel_size = 0.1
        segmenter.tile_size = 10.0
        segmenter.overlap = 2.0
        segmenter.max_overlaps = 4
        segmenter._rng = np.random.default_rng(4)
        segmenter._model = IntensityModel()
        return segmenter

    def test_hard_vote_uses_mode_and_keeps_first_on_ties(self):
        votes = np.array(
            [
                [1, 2, -1, -1],
                [2, 1, 1, 2],
                [0, 2, 2, -1],
            ],
            dtype=np.int8,
        )

        labels = array_processing.SegmentClass._hard_vote(votes)

        np.testing.assert_array_equal(labels, [1, 2, 2])
        self.assertEqual(labels.dtype, np.int8)

    def test_prediction_accumulation_counts_every_occurrence(self):
        probabilities = np.zeros((3, 2), dtype=np.float32)
        seen = np.zeros((3, 1), dtype=np.int8)
        neighbors = np.array([[0, 1], [1, 2]])
        predictions = np.array(
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.4, 0.6], [0.9, 0.1]],
            ],
            dtype=np.float32,
        )

        array_processing.SegmentClass._add_predictions(
            probabilities,
            seen,
            neighbors,
            predictions,
        )

        np.testing.assert_array_equal(seen[:, 0], [1, 2, 1])
        np.testing.assert_allclose(
            probabilities,
            [[0.8, 0.2], [0.7, 1.3], [0.9, 0.1]],
        )

    def test_part_sampling_classifies_every_point(self):
        segmenter = self.make_segmenter()
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        intensity = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

        labels = segmenter._segment_part(points, intensity)

        np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0])

    def test_overlapping_parts_use_first_label_when_votes_tie(self):
        segmenter = self.make_segmenter()
        segmenter.max_overlaps = 2
        points = np.zeros((3, 3), dtype=np.float32)
        intensity = np.zeros(3, dtype=np.float32)
        parts = [np.array([0, 1]), np.array([1, 2])]
        part_labels = [
            np.array([2, 1], dtype=np.int8),
            np.array([0, 0], dtype=np.int8),
        ]

        with patch.object(segmenter, '_iter_parts', return_value=iter(parts)), \
                patch.object(segmenter, '_segment_part', side_effect=part_labels):
            labels = segmenter._segment_subsampled(points, intensity)

        np.testing.assert_array_equal(labels, [2, 1, 0])

    def test_tile_overlap_bound_covers_every_point(self):
        segmenter = self.make_segmenter()
        coordinates = np.array([0.0, 8.5, 10.0, 11.5, 20.0])
        xx, yy = np.meshgrid(coordinates, coordinates)
        points = np.column_stack(
            (xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size))
        ).astype(np.float32)
        occurrences = np.zeros(len(points), dtype=np.int8)

        for indices in segmenter._iter_parts(points):
            occurrences[indices] += 1

        self.assertTrue(np.all(occurrences >= 1))
        self.assertLessEqual(occurrences.max(), segmenter.max_overlaps)

    def test_segment_pcd_returns_labels_for_original_dense_points(self):
        segmenter = self.make_segmenter()
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_points = points.copy()
        intensity = np.array([0, 1, 10, 20], dtype=np.uint16)

        def subsample(xyz, feats, source_indices, voxel_size):
            selected = np.array([0, 2])
            return xyz[selected], feats[selected], source_indices[selected]

        with patch.object(
            array_processing,
            'voxel_subsample_vectorized',
            side_effect=subsample,
        ), patch.object(
            segmenter,
            '_segment_subsampled',
            return_value=np.array([1, 0], dtype=np.int8),
        ):
            labels = segmenter.segment_pcd(points, intensity)

        np.testing.assert_array_equal(labels, [1, 1, 0, 0])
        np.testing.assert_array_equal(points, original_points)
        self.assertEqual(labels.dtype, np.int8)


if __name__ == '__main__':
    unittest.main()
