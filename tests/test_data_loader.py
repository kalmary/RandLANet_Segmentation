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
    assert second == [np.random.default_rng(102).random()]
    assert first != second


def test_make_loader_uses_one_nonpersistent_producer(tmp_path, save_tile):
    save_tile(tmp_path)

    loader, dataset = _data_loader.make_loader(
        data_dir=tmp_path,
        num_points=8,
        batch_size=3,
        query_workers=2,
        shuffle=False,
    )

    assert loader.num_workers == 1
    assert not loader.persistent_workers
    assert loader.prefetch_factor == 2
    assert not loader.pin_memory
    assert dataset.query_workers == 2
    context = loader.multiprocessing_context
    assert context is not None
    assert context.get_start_method() == "spawn"
    
