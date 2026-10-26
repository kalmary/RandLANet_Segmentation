import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, IterableDataset


class CustomDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        num_points: int = 8192,
        batch_size: int = 8,
        shuffle: bool = True,
        query_workers: int = -1,
        epoch: int = 0,
        class_weights=None,
        max_seen: int = 1,
    ):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.query_workers = query_workers
        self._epoch = epoch
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = np.asarray(class_weights, dtype=np.float64)
            if self.class_weights.ndim != 1 or not self.class_weights.size:
                raise ValueError("class_weights must be a non-empty 1-D array")
            if not np.all(np.isfinite(self.class_weights)):
                raise ValueError("class_weights must be finite")
            if np.any((self.class_weights < 0) | (self.class_weights > 1)):
                raise ValueError("class_weights must be within [0, 1]")
        if isinstance(max_seen, (bool, np.bool_)) or not 1 <= max_seen <= 127:
            raise ValueError("max_seen must be an integer within [1, 127]")
        self.max_seen = int(max_seen)

        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if query_workers == 0 or query_workers < -1:
            raise ValueError("query_workers must be -1 or a positive integer")

        missing = [
            path
            for path in self.files
            if not path.with_suffix(".pkl").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing matching .pkl trees for: {missing[:3]}"
            )


    def _load_file(self, path):
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(f"Expected (N, 5) data in {path}, got {data.shape}")

        xyz = np.ascontiguousarray(data[:, :3], dtype=np.float32)
        feats = np.ascontiguousarray(data[:, 3:4], dtype=np.float32)
        labels = data[:, 4].astype(np.int32)
        del data

        with path.with_suffix(".pkl").open("rb") as file:
            tree = pickle.load(file)

        if not isinstance(tree, cKDTree):
            raise TypeError(f"Expected cKDTree in {path.with_suffix('.pkl')}")
        if tree.n != len(xyz):
            raise ValueError(
                f"Point/tree size mismatch for {path.name}: "
                f"{len(xyz)} points, {tree.n} tree entries"
            )
        if len(xyz) < self.num_points:
            raise ValueError(
                f"{path.name} contains {len(xyz)} points, "
                f"but num_points is {self.num_points}"
            )
        if labels.size and labels.min() < -1:
            raise ValueError(f"Labels below ignore label -1 found in {path}")
        if (
            self.class_weights is not None
            and labels.size
            and labels.max() >= len(self.class_weights)
        ):
            raise ValueError(f"Label outside class_weights in {path}")

        return xyz, feats, labels, tree


    def _query_neighbors(self, tree, xyz, center_indices):
        centers = xyz[center_indices]
        _, queried = tree.query(
            centers,
            k=self.num_points,
            workers=self.query_workers,
        )

        queried = np.asarray(queried, dtype=np.int64)
        if queried.ndim == 1:
            queried = queried[None, :]

        neighbors = np.empty_like(queried)
        neighbors[:, 0] = center_indices

        for row, center_idx in enumerate(center_indices):
            other = queried[row][queried[row] != center_idx]
            neighbors[row, 1:] = other[:self.num_points - 1]

        return neighbors

    @staticmethod
    def _update_possibilities(
        possibilities,
        xyz,
        center_indices,
        neighbor_indices,
    ):
        neighborhoods = xyz[neighbor_indices]
        centers = xyz[center_indices, None, :]
        squared_distances = np.sum(
            (neighborhoods - centers) ** 2,
            axis=2,
        )
        max_distances = squared_distances.max(axis=1, keepdims=True)
        normalized = np.divide(
            squared_distances,
            max_distances,
            out=np.zeros_like(squared_distances),
            where=max_distances > 0,
        )
        deltas = (1.0 - normalized) ** 2
        np.add.at(possibilities, neighbor_indices, deltas)

    def _iter_file(self, path, rng):
        xyz, feats, labels, tree = self._load_file(path)
        possibilities = rng.random(len(xyz)) * 1e-3
        valid = labels >= 0
        if not np.any(valid):
            raise ValueError(f"No labeled points in {path}")
        if self.class_weights is None:
            targets = valid.astype(np.int16)
        else:
            class_targets = np.maximum(
                1,
                np.floor(self.class_weights * self.max_seen + 0.5),
            ).astype(np.int16)
            targets = np.zeros(len(xyz), dtype=np.int16)
            targets[valid] = class_targets[labels[valid]]
        seen = np.zeros(len(xyz), dtype=np.int16)

        while np.any(seen < targets):
            remaining = np.flatnonzero(seen < targets)
            center_count = min(self.batch_size, remaining.size)
            normalized_coverage = seen[remaining] / targets[remaining]
            ranked = np.lexsort(
                (possibilities[remaining], normalized_coverage),
            )
            center_indices = remaining[ranked[:center_count]]
            neighbor_indices = self._query_neighbors(
                tree,
                xyz,
                center_indices,
            )
            self._update_possibilities(
                possibilities,
                xyz,
                center_indices,
                neighbor_indices,
            )
            np.add.at(seen, neighbor_indices, 1)
            np.minimum(seen, 127, out=seen)

            centers = xyz[center_indices, None, :]
            batch_xyz = xyz[neighbor_indices] - centers
            batch_feats = feats[neighbor_indices]
            batch = np.concatenate((batch_xyz, batch_feats), axis=2)
            batch_labels = labels[neighbor_indices].astype(
                np.int64,
                copy=False,
            )

            yield (
                torch.from_numpy(batch),
                torch.from_numpy(batch_labels),
            )

        return seen >= targets

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 1:
            raise RuntimeError(
                "CustomDataset requires exactly one DataLoader worker; "
                "query_workers controls parallel cKDTree queries"
            )

        if worker_info is None:
            rng = np.random.default_rng(self._epoch)
            self._epoch += 1
        else:
            rng = np.random.default_rng(worker_info.seed)
        order = np.arange(len(self.files))
        if self.shuffle:
            rng.shuffle(order)

        for file_idx in order:
            yield from self._iter_file(self.files[file_idx], rng)


def make_loader(
    data_dir,
    num_points: int = 8192,
    batch_size: int = 8,
    query_workers: int = -1,
    shuffle: bool = True,
    epoch: int = 0,
    class_weights=None,
    max_seen: int = 1,
) -> tuple[DataLoader, CustomDataset]:
    dataset = CustomDataset(
        data_dir=data_dir,
        num_points=num_points,
        batch_size=batch_size,
        shuffle=shuffle,
        query_workers=query_workers,
        epoch=epoch,
        class_weights=class_weights,
        max_seen=max_seen,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=False,
        multiprocessing_context="spawn",
    )
    return loader, dataset
