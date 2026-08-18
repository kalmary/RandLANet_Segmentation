import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

from src.data_processing import downsample_LAZ


@pytest.fixture
def point_cloud():
    rng = np.random.default_rng(7)
    xyz = rng.random((256, 3), dtype=np.float32)
    feats = rng.random((256, 1), dtype=np.float32)
    labels = rng.integers(0, 10, size=256, dtype=np.int32)
    return xyz, feats, labels


@pytest.fixture
def save_tile(monkeypatch, point_cloud):
    xyz, feats, labels = point_cloud

    def save(source_path, output_dir):
        tile = (xyz.copy(), feats.copy(), labels.copy(), (0, 0))
        monkeypatch.setattr(
            downsample_LAZ,
            "load_and_normalise",
            lambda _source_path: (xyz.copy(), feats.copy(), labels.copy()),
        )
        monkeypatch.setattr(
            downsample_LAZ,
            "voxel_subsample_vectorized",
            lambda xyz, feats, labels, voxel_size: (xyz, feats, labels),
        )
        monkeypatch.setattr(
            downsample_LAZ,
            "iter_tiles",
            lambda *args, **kwargs: iter((tile,)),
        )
        monkeypatch.setattr(downsample_LAZ.tqdm, "write", lambda *args, **kwargs: None)

        downsample_LAZ.save_tiles(source_path, output_dir)

    return save


def test_save_tiles_creates_distinct_paired_artifacts(tmp_path, save_tile, point_cloud):
    xyz, feats, labels = point_cloud
    output_dir = tmp_path / "cut"
    first_source = tmp_path / "source_a" / "scan.las"
    second_source = tmp_path / "source_b" / "scan.las"

    save_tile(first_source, output_dir)
    save_tile(second_source, output_dir)

    point_files = sorted(output_dir.glob("*.npy"))
    tree_files = sorted(output_dir.glob("*.pkl"))

    assert len(point_files) == 2
    assert len(tree_files) == 2
    assert {path.stem for path in point_files} == {
        path.stem for path in tree_files
    }
    assert point_files[0].stem != point_files[1].stem

    expected = np.concatenate(
        [xyz, feats, labels[:, None].astype(np.float32)],
        axis=1,
    )
    saved = np.load(point_files[0])
    np.testing.assert_array_equal(saved, expected)
    assert saved.dtype == np.float32

    with point_files[0].with_suffix(".pkl").open("rb") as file:
        tree = pickle.load(file)

    assert isinstance(tree, cKDTree)
    _, indices = tree.query(xyz[0], k=32)
    assert indices.shape == (32,)
    assert 0 in indices


def test_save_tiles_does_not_overwrite_existing_artifacts(tmp_path, save_tile):
    output_dir = tmp_path / "cut"
    source_path = tmp_path / "source" / "scan.las"

    save_tile(source_path, output_dir)

    point_path = next(output_dir.glob("*.npy"))
    tree_path = point_path.with_suffix(".pkl")
    point_data = point_path.read_bytes()
    tree_data = tree_path.read_bytes()

    with pytest.raises(FileExistsError):
        save_tile(source_path, output_dir)

    assert point_path.read_bytes() == point_data
    assert tree_path.read_bytes() == tree_data


def test_split_dataset_preserves_existing_dales_splits(tmp_path, save_tile):
    cut_dir = tmp_path / "cut"
    output_dir = tmp_path / "split"

    for split in ("train", "val", "test"):
        save_tile(tmp_path / split / f"{split}.las", cut_dir / split)

    downsample_LAZ.split_dataset(cut_dir, output_dir)

    for split in ("train", "val", "test"):
        point_files = list((output_dir / split).glob("*.npy"))
        tree_files = list((output_dir / split).glob("*.pkl"))
        assert len(point_files) == 1
        assert len(tree_files) == 1
        assert point_files[0].stem == tree_files[0].stem
        assert point_files[0].stem.startswith(split)


def test_split_dataset_keeps_point_clouds_and_trees_paired(tmp_path, save_tile):
    cut_dir = tmp_path / "cut"
    output_dir = tmp_path / "split"

    save_tile(tmp_path / "source_a" / "scan.las", cut_dir)
    save_tile(tmp_path / "source_b" / "scan.las", cut_dir)

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

    assert len(point_files) == 2
    assert len(tree_files) == 2
    assert {
        (path.parent.name, path.stem) for path in point_files
    } == {
        (path.parent.name, path.stem) for path in tree_files
    }
