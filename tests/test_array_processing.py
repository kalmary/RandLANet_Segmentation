import pathlib
import sys

import numpy as np
import pytest
import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src import array_processing


class IntensityModel(torch.nn.Module):
    def forward(self, inputs):
        intensity = inputs[..., 3]
        return torch.stack((1.0 - intensity, intensity), dim=1) * 8.0


def test_segmenter_loads_config_and_checkpoint_from_separate_directories(
    tmp_path,
    monkeypatch,
):
    config_dir = tmp_path / 'dict_files'
    model_dir = tmp_path / 'models'
    loaded = {}

    def load_config(self, path):
        loaded['config_dir'] = path
        return {
            'model_config': {},
            'num_points': 16,
            'batch_size': 2,
            'num_classes': 3,
        }

    def load_model(self, path):
        loaded['model_dir'] = path
        return IntensityModel()

    monkeypatch.setattr(array_processing.SegmentClass, '_load_config', load_config)
    monkeypatch.setattr(array_processing.SegmentClass, '_load_model', load_model)

    instance = array_processing.SegmentClass(
        model_name='model_1',
        config_dir=config_dir,
        model_dir=model_dir,
    )

    assert loaded == {
        'config_dir': config_dir,
        'model_dir': model_dir,
    }
    assert instance.config['num_classes'] == 3


@pytest.fixture
def segmenter():
    instance = array_processing.SegmentClass.__new__(
        array_processing.SegmentClass
    )
    instance.device = torch.device('cpu')
    instance.num_points = 3
    instance.batch_size = 2
    instance.n_classes = 2
    instance.n_seen = 2
    instance.query_workers = 1
    instance.pbar_bool = False
    instance.voxel_size = 0.1
    instance.tile_size = 10.0
    instance.overlap = 2.0
    instance.max_overlaps = 4
    instance._rng = np.random.default_rng(4)
    instance._model = IntensityModel()
    return instance


def test_hard_vote_uses_mode_and_keeps_first_on_ties():
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
    assert labels.dtype == np.int8


def test_prediction_accumulation_counts_every_occurrence():
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


def test_possibility_updates_use_randlanet_distance_deltas():
    possibilities = np.zeros(3, dtype=np.float64)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    array_processing.SegmentClass._update_possibilities(
        possibilities,
        points,
        center_indices=np.array([0]),
        neighbor_indices=np.array([[0, 1, 2]]),
    )

    np.testing.assert_allclose(possibilities, [1.0, 0.5625, 0.0])


def test_part_sampling_starts_with_lowest_possibility_centers(
    segmenter,
    monkeypatch,
):
    points = np.column_stack(
        (
            np.arange(5, dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
        )
    )
    intensity = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    first_possibilities = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    selected_centers = []
    original_query = segmenter._query_neighbors

    class FixedRng:
        @staticmethod
        def random(size):
            assert size == len(first_possibilities)
            return first_possibilities

    def record_query(tree, xyz, center_indices):
        selected_centers.append(center_indices.copy())
        return original_query(tree, xyz, center_indices)

    segmenter.n_seen = 1
    segmenter._rng = FixedRng()
    monkeypatch.setattr(segmenter, '_query_neighbors', record_query)

    segmenter._segment_part(points, intensity)

    np.testing.assert_array_equal(selected_centers[0], [1, 3])


def test_part_sampling_classifies_every_point(segmenter):
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


def test_overlapping_parts_use_first_label_when_votes_tie(
    segmenter,
    monkeypatch,
):
    segmenter.max_overlaps = 2
    points = np.zeros((3, 3), dtype=np.float32)
    intensity = np.zeros(3, dtype=np.float32)
    parts = [np.array([0, 1]), np.array([1, 2])]
    part_labels = iter(
        [
            np.array([2, 1], dtype=np.int8),
            np.array([0, 0], dtype=np.int8),
        ]
    )
    monkeypatch.setattr(segmenter, '_iter_parts', lambda _: iter(parts))
    monkeypatch.setattr(
        segmenter,
        '_segment_part',
        lambda *_: next(part_labels),
    )

    labels = segmenter._segment_subsampled(points, intensity)

    np.testing.assert_array_equal(labels, [2, 1, 0])


def test_tile_overlap_bound_covers_every_point(segmenter):
    coordinates = np.array([0.0, 8.5, 10.0, 11.5, 20.0])
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack(
        (xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size))
    ).astype(np.float32)
    occurrences = np.zeros(len(points), dtype=np.int8)

    for indices in segmenter._iter_parts(points):
        occurrences[indices] += 1

    assert np.all(occurrences >= 1)
    assert occurrences.max() <= segmenter.max_overlaps


def test_segment_pcd_returns_labels_for_original_dense_points(
    segmenter,
    monkeypatch,
):
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

    monkeypatch.setattr(
        array_processing,
        'voxel_subsample_vectorized',
        subsample,
    )
    captured = {}

    def segment_subsampled(subsampled_points, subsampled_intensity):
        captured['intensity'] = subsampled_intensity.copy()
        return np.array([1, 0], dtype=np.int8)

    monkeypatch.setattr(segmenter, '_segment_subsampled', segment_subsampled)

    labels = segmenter.segment_pcd(points, intensity)

    np.testing.assert_allclose(captured['intensity'], [0.0, 0.5])
    np.testing.assert_array_equal(labels, [1, 1, 0, 0])
    np.testing.assert_array_equal(points, original_points)
    assert labels.dtype == np.int8
