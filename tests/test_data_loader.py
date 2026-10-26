import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from src.model_pipeline import _data_loader


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(12)
    return SimpleNamespace(
        xyz=rng.random((24, 3), dtype=np.float32),
        feats=rng.random((24, 1), dtype=np.float32),
        labels=np.repeat(np.arange(2, dtype=np.int32), 12),
    )


@pytest.fixture
def save_tile(sample_data):
    def _save_tile(directory):
        path = Path(directory) / "scan_source_tile_000_000.npy"
        data = np.concatenate(
            (
                sample_data.xyz,
                sample_data.feats,
                sample_data.labels[:, None].astype(np.float32),
            ),
            axis=1,
        )
        np.save(path, data)

        tree = cKDTree(sample_data.xyz, leafsize=4)
        with path.with_suffix(".pkl").open("wb") as file:
            pickle.dump(
                tree,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return path

    return _save_tile


@pytest.fixture
def make_dataset():
    def _make_dataset(directory, **kwargs):
        return _data_loader.CustomDataset(
            data_dir=directory,
            num_points=8,
            batch_size=3,
            query_workers=2,
            shuffle=False,
            **kwargs,
        )

    return _make_dataset


def test_possibility_updates_use_randlanet_distance_deltas():
    possibilities = np.zeros(3, dtype=np.float64)
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    _data_loader.CustomDataset._update_possibilities(
        possibilities,
        xyz,
        center_indices=np.array([0]),
        neighbor_indices=np.array([[0, 1, 2]]),
    )

    np.testing.assert_allclose(possibilities, [1.0, 0.5625, 0.0])


def test_sampling_covers_every_point_once_or_more(
    tmp_path,
    sample_data,
    save_tile,
    make_dataset,
):
    path = save_tile(tmp_path)
    dataset = make_dataset(tmp_path)

    class FixedRng:
        @staticmethod
        def random(size):
            return np.arange(size, 0, -1, dtype=np.float64)

    iterator = dataset._iter_file(path, FixedRng())
    batches = []

    while True:
        try:
            batch, labels = next(iterator)
        except StopIteration as result:
            covered = result.value
            break

        batches.append((batch, labels))
        assert batch.shape[1:] == (8, 4)
        assert labels.shape[1:] == (8,)
        np.testing.assert_array_equal(
            batch[:, 0, :3].numpy(),
            np.zeros((batch.shape[0], 3), dtype=np.float32),
        )

    assert batches
    assert len(batches) <= int(np.ceil(len(sample_data.xyz) / 3))
    assert covered.dtype == np.bool_
    assert covered.shape == (len(sample_data.xyz),)
    assert np.all(covered)
    np.testing.assert_array_equal(
        batches[0][0][:, 0, 3].numpy(),
        sample_data.feats[[23, 22, 21], 0],
    )


def test_torch_workers_are_rejected(
    tmp_path,
    save_tile,
    make_dataset,
    monkeypatch,
):
    save_tile(tmp_path)
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(
        "torch.utils.data.get_worker_info",
        lambda: SimpleNamespace(num_workers=2),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        next(iter(dataset))


def test_worker_rng_uses_worker_seed(
    tmp_path,
    save_tile,
    make_dataset,
    monkeypatch,
):
    save_tile(tmp_path)
    dataset = make_dataset(tmp_path)
    dataset.shuffle = True
    worker_info = SimpleNamespace(num_workers=1, seed=101)
    monkeypatch.setattr(
        "torch.utils.data.get_worker_info",
        lambda: worker_info,
    )

    def sample_rng(_path, rng):
        yield rng.random()

    monkeypatch.setattr(dataset, "_iter_file", sample_rng)
    first = list(dataset)
    worker_info.seed = 102
    second = list(dataset)

    assert first == [np.random.default_rng(101).random()]
    assert second == [np.random.default_rng(103).random()]
    assert first != second


def test_make_loader_uses_one_persistent_producer(tmp_path, save_tile):
    save_tile(tmp_path)

    loader, dataset = _data_loader.make_loader(
        data_dir=tmp_path,
        num_points=8,
        batch_size=3,
        query_workers=2,
        shuffle=False,
    )

    assert loader.num_workers == 1
    assert loader.persistent_workers
    assert loader.prefetch_factor == 2
    assert not loader.pin_memory
    assert dataset.query_workers == 2
    context = loader.multiprocessing_context
    assert context is not None
    assert context.get_start_method() == "spawn"


def test_class_weights_require_more_coverage_for_rare_points(
    tmp_path,
    sample_data,
    save_tile,
):
    path = save_tile(tmp_path)
    dataset = _data_loader.CustomDataset(
        data_dir=tmp_path,
        num_points=8,
        batch_size=2,
        query_workers=1,
        shuffle=False,
        class_weights=[0.25, 1.0],
        max_seen=4,
    )

    iterator = dataset._iter_file(path, np.random.default_rng(3))
    label_occurrences = np.zeros(2, dtype=np.int64)
    while True:
        try:
            _, labels = next(iterator)
        except StopIteration as result:
            satisfied = result.value
            break
        label_occurrences += np.bincount(
            labels.numpy().reshape(-1),
            minlength=2,
        )

    assert np.all(satisfied)
    assert label_occurrences[1] > label_occurrences[0]


def test_ignored_points_are_context_but_never_crop_centers(
    tmp_path,
    save_tile,
    make_dataset,
):
    path = save_tile(tmp_path)
    data = np.load(path)
    data[:6, 4] = -1
    np.save(path, data)
    dataset = make_dataset(tmp_path)

    batches = list(dataset._iter_file(path, np.random.default_rng(2)))

    assert batches
    assert all(np.all(labels[:, 0].numpy() >= 0) for _, labels in batches)
    assert any(np.any(labels.numpy() == -1) for _, labels in batches)


def test_tiles_smaller_than_crop_are_repeated_at_model_input(tmp_path):
    xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
    data = np.column_stack(
        (xyz, np.ones(5, dtype=np.float32), np.arange(5) % 2)
    ).astype(np.float32)
    path = tmp_path / "small_tile.npy"
    np.save(path, data)
    with path.with_suffix(".pkl").open("wb") as file:
        pickle.dump(cKDTree(xyz), file)
    dataset = _data_loader.CustomDataset(
        tmp_path, num_points=8, batch_size=2, query_workers=1,
        shuffle=False,
    )

    batches = list(dataset._iter_file(path, np.random.default_rng(0)))

    assert batches
    assert all(batch.shape[1:] == (8, 4) for batch, _ in batches)


def test_fixed_step_budget_stops_epoch_without_exhaustive_coverage(
    tmp_path,
    save_tile,
):
    save_tile(tmp_path)
    dataset = _data_loader.CustomDataset(
        tmp_path,
        num_points=8,
        batch_size=2,
        query_workers=1,
        shuffle=True,
        class_weights=[0.25, 1.0],
        max_seen=3,
        steps_per_epoch=5,
        batches_per_file=2,
    )

    batches = list(dataset)

    assert len(batches) == 5


def test_validation_fixed_budget_repeats_identical_crops(tmp_path, save_tile):
    save_tile(tmp_path)
    dataset = _data_loader.CustomDataset(
        tmp_path,
        num_points=8,
        batch_size=2,
        query_workers=1,
        shuffle=False,
        steps_per_epoch=2,
        batches_per_file=2,
    )

    first = list(dataset)
    second = list(dataset)

    assert len(first) == len(second) == 2
    for (first_x, first_y), (second_x, second_y) in zip(first, second):
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)


def test_class_targets_never_exclude_frequent_classes(tmp_path, save_tile):
    path = save_tile(tmp_path)
    dataset = _data_loader.CustomDataset(
        tmp_path,
        num_points=8,
        batch_size=2,
        query_workers=1,
        shuffle=False,
        class_weights=[0.0, 1.0],
        max_seen=3,
        steps_per_epoch=1,
    )

    _, labels = next(dataset._iter_file(path, np.random.default_rng(0), 1))

    # Weight zero still maps to target one, so common class 0 remains eligible
    # as a center and appears in naturally distributed neighborhoods.
    assert 0 in labels.numpy()


@pytest.mark.parametrize("max_seen", [0, 128, True])
def test_max_seen_must_fit_coverage_counter(tmp_path, save_tile, max_seen):
    save_tile(tmp_path)
    with pytest.raises(ValueError, match="max_seen"):
        _data_loader.CustomDataset(tmp_path, max_seen=max_seen)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.inf])
def test_class_aware_fraction_is_a_probability(tmp_path, save_tile, fraction):
    save_tile(tmp_path)
    with pytest.raises(ValueError, match="class_aware_fraction"):
        _data_loader.CustomDataset(
            tmp_path, class_aware_fraction=fraction
        )
    
