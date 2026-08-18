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
        pos_weights: np.ndarray = None,
        max_seen: int = 10,
        query_workers: int = -1,
        epoch: int = 0,
    ):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seen = max_seen
        self.query_workers = query_workers
        self._epoch = epoch
        self.class_targets = self._build_class_targets(pos_weights)

        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 1 <= max_seen <= np.iinfo(np.int8).max:
            raise ValueError("max_seen must fit in int8")
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

    def _build_class_targets(self, pos_weights):
        if pos_weights is None:
            return None

        weights = np.asarray(pos_weights, dtype=np.float64).reshape(-1)
        if weights.size == 0:
            raise ValueError("pos_weights cannot be empty")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.any(weights > 1)
        ):
            raise ValueError("pos_weights must be finite and within [0, 1]")

        positive = weights > 0
        if not np.any(positive):
            raise ValueError("pos_weights must contain a positive value")

        targets = np.ones_like(weights, dtype=np.int8)
        scaled = weights[positive] * self.max_seen
        targets[positive] = np.clip(
            np.floor(scaled + 0.5),
            1,
            self.max_seen,
        ).astype(np.int8)
        return targets

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
        if labels.size and labels.min() < 0:
            raise ValueError(f"Negative labels found in {path}")

        return xyz, feats, labels, tree

    def _point_targets(self, labels):
        if self.class_targets is None:
            return np.full(
                (len(labels), 1),
                self.max_seen,
                dtype=np.int8,
            )

        if labels.size and labels.max() >= len(self.class_targets):
            raise ValueError(
                f"Label {labels.max()} has no corresponding class weight"
            )
        return self.class_targets[labels, None]

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
    def _update_seen(seen, neighbor_indices):
        indices, increments = np.unique(
            neighbor_indices.reshape(-1),
            return_counts=True,
        )
        values = seen[indices, 0].astype(np.int32) + increments
        seen[indices, 0] = np.minimum(
            values,
            np.iinfo(np.int8).max,
        ).astype(np.int8)

    def _iter_file(self, path, rng):
        xyz, feats, labels, tree = self._load_file(path)
        targets = self._point_targets(labels)
        seen = np.zeros((len(xyz), 1), dtype=np.int8)

        while True:
            remaining = np.flatnonzero(seen[:, 0] < targets[:, 0])
            if remaining.size == 0:
                break

            center_count = min(self.batch_size, remaining.size)
            center_indices = rng.choice(
                remaining,
                size=center_count,
                replace=False,
            )
            neighbor_indices = self._query_neighbors(
                tree,
                xyz,
                center_indices,
            )
            self._update_seen(seen, neighbor_indices)

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

        return seen

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 1:
            raise RuntimeError(
                "CustomDataset requires exactly one DataLoader worker; "
                "query_workers controls parallel cKDTree queries"
            )

        rng = np.random.default_rng(self._epoch)
        self._epoch += 1
        order = np.arange(len(self.files))
        if self.shuffle:
            rng.shuffle(order)

        for file_idx in order:
            yield from self._iter_file(self.files[file_idx], rng)


def compute_pos_weights(
    data_dir,
    num_classes: int,
    power: float = 0.25,
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)

    for path in sorted(Path(data_dir).glob("*.npy")):
        labels = np.load(path, mmap_mode="r")[:, 4].astype(np.int32)
        file_counts = np.bincount(labels, minlength=num_classes)
        if len(file_counts) != num_classes:
            raise ValueError(f"Label outside configured classes in {path}")
        counts += file_counts

    weights = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    if weights.max() > 0:
        weights /= weights.max()
    return weights.astype(np.float32)


def make_loader(
    data_dir,
    num_points: int = 8192,
    batch_size: int = 8,
    query_workers: int = -1,
    shuffle: bool = True,
    pos_weights: np.ndarray = None,
    max_seen: int = 10,
    epoch: int = 0,
) -> tuple[DataLoader, CustomDataset]:
    dataset = CustomDataset(
        data_dir=data_dir,
        num_points=num_points,
        batch_size=batch_size,
        shuffle=shuffle,
        pos_weights=pos_weights,
        max_seen=max_seen,
        query_workers=query_workers,
        epoch=epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
        multiprocessing_context="spawn",
    )
    return loader, dataset
