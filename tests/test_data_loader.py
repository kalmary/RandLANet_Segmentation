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


def test_class_weights_set_required_views(tmp_path, save_tile, make_dataset):
    save_tile(tmp_path)

    dataset = make_dataset(
        tmp_path,
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

    equal_dataset = make_dataset(
        tmp_path,
        pos_weights=np.ones(3, dtype=np.float32),
        max_seen=5,
    )
    np.testing.assert_array_equal(
        equal_dataset.class_targets,
        np.array([5, 5, 5], dtype=np.int8),
    )


def test_sampling_reaches_each_points_required_views(
    tmp_path,
    sample_data,
    save_tile,
    make_dataset,
):
    path = save_tile(tmp_path)
    dataset = make_dataset(
        tmp_path,
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
        assert batch.shape[1:] == (8, 4)
        assert labels.shape[1:] == (8,)
        np.testing.assert_array_equal(
            batch[:, 0, :3].numpy(),
            np.zeros((batch.shape[0], 3), dtype=np.float32),
        )

    assert batches
    targets = dataset._point_targets(sample_data.labels)
    assert np.all(seen >= targets)
    assert seen.dtype == np.int8
    assert seen.shape == (len(sample_data.xyz), 1)


def test_torch_workers_are_rejected(
    tmp_path,
    save_tile,
    make_dataset,
    monkeypatch,
):
    save_tile(tmp_path)
    dataset = make_dataset(tmp_path, max_seen=1)
    monkeypatch.setattr(
        "torch.utils.data.get_worker_info",
        lambda: SimpleNamespace(num_workers=2),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        next(iter(dataset))


def test_make_loader_uses_one_persistent_producer(tmp_path, save_tile):
    save_tile(tmp_path)

    loader, dataset = _data_loader.make_loader(
        data_dir=tmp_path,
        num_points=8,
        batch_size=3,
        query_workers=2,
        shuffle=False,
        pos_weights=np.ones(2, dtype=np.float32),
        max_seen=1,
    )

    assert loader.num_workers == 1
    assert loader.persistent_workers
    assert loader.prefetch_factor == 2
    assert not loader.pin_memory
    assert dataset.query_workers == 2

    first_epoch = list(loader)
    second_epoch = list(loader)

    batch, labels = first_epoch[0]
    assert batch.shape == (3, 8, 4)
    assert labels.shape == (3, 8)

    first_centers = np.concatenate(
        [batch[:, 0, 3].numpy() for batch, _ in first_epoch]
    )
    second_centers = np.concatenate(
        [batch[:, 0, 3].numpy() for batch, _ in second_epoch]
    )
    assert not np.array_equal(first_centers, second_centers)
