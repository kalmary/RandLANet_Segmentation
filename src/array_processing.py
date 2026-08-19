from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    from .data_processing.downsample_LAZ import voxel_subsample_vectorized
    from .model_pipeline.RandLANet_CB import RandLANet
    from .utils import load_json, load_model
except ImportError:
    from data_processing.downsample_LAZ import voxel_subsample_vectorized
    from model_pipeline.RandLANet_CB import RandLANet
    from utils import load_json, load_model


class SegmentClass:
    def __init__(
        self,
        model_name: str,
        config_dir: Union[str, Path] = "final_files",
        model_dir: Union[str, Path, None] = None,
        device: Union[str, torch.device] = torch.device("cpu"),
        voxel_size: float = 0.10,
        tile_size: float = 40.0,
        overlap: float = 5.0,
        num_votes: int = 3,
        query_workers: int = -1,
        pbar_bool: bool = False,
    ):
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if (
            isinstance(num_votes, (bool, np.bool_))
            or not isinstance(num_votes, (int, np.integer))
            or num_votes < 1
        ):
            raise ValueError("num_votes must be a positive integer")
        if query_workers == 0 or query_workers < -1:
            raise ValueError("query_workers must be -1 or a positive integer")

        self.model_name = model_name.removesuffix(".pt")
        self.device = torch.device(device)
        self.voxel_size = float(voxel_size)
        self.tile_size = float(tile_size)
        self.overlap = float(overlap)
        self.num_votes = int(num_votes)
        self.query_workers = int(query_workers)
        self.pbar_bool = pbar_bool
        self._rng = np.random.default_rng()

        base_path = Path(__file__).parent
        config_dir = Path(config_dir)
        if not config_dir.is_absolute():
            config_dir = base_path / config_dir

        model_dir = config_dir if model_dir is None else Path(model_dir)
        if not model_dir.is_absolute():
            model_dir = base_path / model_dir

        self._config = self._load_config(config_dir)
        self._model_config = self._config["model_config"]
        self.num_points = int(self._config["num_points"])
        self.batch_size = int(self._config["batch_size"])
        self.n_classes = int(self._config["num_classes"])

        if self.n_classes > np.iinfo(np.int8).max + 1:
            raise ValueError("n_classes must fit in int8 labels")

        overlaps_per_axis = int(
            np.ceil((self.tile_size + 2 * self.overlap) / self.tile_size)
        )
        self.max_overlaps = overlaps_per_axis ** 2
        if self.max_overlaps > np.iinfo(np.int8).max:
            raise ValueError("tile overlap count must fit in int8")

        self._model = self._load_model(model_dir)

    def _load_config(self, config_dir: Path) -> dict:
        config_path = config_dir / f"{self.model_name}_config.json"
        return load_json(config_path)

    def _load_model(self, model_dir: Path) -> nn.Module:
        model = RandLANet(
            model_config=self._model_config,
            n_classes=self.n_classes,
        )
        model = load_model(
            file_path=model_dir / f"{self.model_name}.pt",
            model=model,
            device=self.device,
        )
        model.eval()
        return model

    @property
    def config(self) -> dict:
        return self._config

    @property
    def model_config(self) -> dict:
        return self._model_config

    @property
    def model(self) -> nn.Module:
        return self._model

    def _model_predict(self, batch: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            outputs = self._model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.movedim(1, -1).cpu().numpy()

    def _query_neighbors(
        self,
        tree: cKDTree,
        points: np.ndarray,
        center_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        neighbor_count = min(self.num_points, len(points))
        _, queried = tree.query(
            points[center_indices],
            k=neighbor_count,
            workers=self.query_workers,
        )
        queried = np.asarray(queried, dtype=np.int64)
        if neighbor_count == 1:
            queried = queried.reshape(-1, 1)
        elif queried.ndim == 1:
            queried = queried[None, :]

        neighbors = np.empty_like(queried)
        neighbors[:, 0] = center_indices
        for row, center_idx in enumerate(center_indices):
            other = queried[row][queried[row] != center_idx]
            neighbors[row, 1:] = other[:neighbor_count - 1]

        if neighbor_count == self.num_points:
            return neighbors, neighbors

        repeats = int(np.ceil(self.num_points / neighbor_count))
        model_neighbors = np.tile(neighbors, (1, repeats))[:, :self.num_points]
        return neighbors, model_neighbors

    @staticmethod
    def _add_predictions(
        probabilities: np.ndarray,
        seen: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_probabilities: np.ndarray,
    ) -> None:
        indices, increments = np.unique(
            neighbor_indices.reshape(-1),
            return_counts=True,
        )
        new_seen = seen[indices, 0].astype(np.int16) + increments

        np.add.at(
            probabilities,
            neighbor_indices.reshape(-1),
            neighbor_probabilities.reshape(-1, probabilities.shape[1]),
        )
        seen[indices, 0] = np.minimum(
            new_seen,
            np.iinfo(np.int8).max,
        ).astype(np.int8)

    @staticmethod
    def _update_possibilities(
        possibilities: np.ndarray,
        points: np.ndarray,
        center_indices: np.ndarray,
        neighbor_indices: np.ndarray,
    ) -> None:
        neighborhoods = points[neighbor_indices]
        centers = points[center_indices, None, :]
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

    def _segment_part(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
    ) -> np.ndarray:
        tree = cKDTree(points, leafsize=40)
        possibilities = self._rng.random(len(points)) * 1e-3
        seen = np.zeros((len(points), 1), dtype=np.int8)
        probabilities = np.zeros(
            (len(points), self.n_classes),
            dtype=np.float32,
        )

        while possibilities.min() < self.num_votes:
            center_count = min(self.batch_size, len(points))
            center_indices = np.argsort(
                possibilities,
                kind="stable",
            )[:center_count]
            neighbors, model_neighbors = self._query_neighbors(
                tree,
                points,
                center_indices,
            )
            self._update_possibilities(
                possibilities,
                points,
                center_indices,
                neighbors,
            )

            centers = points[center_indices, None, :]
            batch_xyz = points[model_neighbors] - centers
            batch_intensity = intensity[model_neighbors, None]
            batch = np.concatenate((batch_xyz, batch_intensity), axis=2)
            batch_probabilities = self._model_predict(batch)
            expected_shape = (
                center_count,
                self.num_points,
                self.n_classes,
            )
            if batch_probabilities.shape != expected_shape:
                raise ValueError(
                    f"Expected model probabilities {expected_shape}, "
                    f"got {batch_probabilities.shape}"
                )

            self._add_predictions(
                probabilities,
                seen,
                neighbors,
                batch_probabilities[:, :neighbors.shape[1]],
            )

        if np.any(seen == 0):
            raise RuntimeError("possibility sampling left points without predictions")
        probabilities /= seen.astype(np.float32)
        return np.argmax(probabilities, axis=1).astype(np.int8)

    def _tile_starts(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        minimum = points[:, :2].min(axis=0)
        maximum = points[:, :2].max(axis=0)
        x_starts = np.arange(minimum[0], maximum[0], self.tile_size)
        y_starts = np.arange(minimum[1], maximum[1], self.tile_size)
        if x_starts.size == 0:
            x_starts = minimum[0:1]
        if y_starts.size == 0:
            y_starts = minimum[1:2]
        return x_starts, y_starts

    def _iter_parts(self, points: np.ndarray):
        x_starts, y_starts = self._tile_starts(points)
        for x_index, x_start in enumerate(x_starts):
            x_end = x_start + self.tile_size + self.overlap
            x_mask = points[:, 0] >= x_start - self.overlap
            if x_index == len(x_starts) - 1:
                x_mask &= points[:, 0] <= x_end
            else:
                x_mask &= points[:, 0] < x_end

            for y_index, y_start in enumerate(y_starts):
                y_end = y_start + self.tile_size + self.overlap
                mask = x_mask & (points[:, 1] >= y_start - self.overlap)
                if y_index == len(y_starts) - 1:
                    mask &= points[:, 1] <= y_end
                else:
                    mask &= points[:, 1] < y_end

                indices = np.flatnonzero(mask)
                if indices.size:
                    yield indices

    @staticmethod
    def _hard_vote(votes: np.ndarray) -> np.ndarray:
        labels = np.full(len(votes), -1, dtype=np.int8)
        best_counts = np.zeros(len(votes), dtype=np.int8)

        for column in range(votes.shape[1]):
            candidate = votes[:, column]
            valid = candidate >= 0
            candidate_counts = np.zeros(len(votes), dtype=np.int8)
            for other_column in range(votes.shape[1]):
                candidate_counts += (
                    valid & (votes[:, other_column] == candidate)
                ).astype(np.int8)

            replace = valid & (candidate_counts > best_counts)
            labels[replace] = candidate[replace]
            best_counts[replace] = candidate_counts[replace]

        if np.any(labels < 0):
            raise RuntimeError("at least one point has no tile prediction")
        return labels

    def _segment_subsampled(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
    ) -> np.ndarray:
        votes = np.full(
            (len(points), self.max_overlaps),
            -1,
            dtype=np.int8,
        )
        vote_counts = np.zeros(len(points), dtype=np.int8)
        parts = self._iter_parts(points)
        if self.pbar_bool:
            parts = tqdm(parts, desc="Cloud parts", unit="part", leave=False)

        for indices in parts:
            part_labels = self._segment_part(
                points[indices],
                intensity[indices],
            )
            positions = vote_counts[indices]
            if np.any(positions >= self.max_overlaps):
                raise RuntimeError("point belongs to more tiles than expected")
            votes[indices, positions] = part_labels
            vote_counts[indices] += 1

        return self._hard_vote(votes)

    def _upsample_labels(
        self,
        subsampled_points: np.ndarray,
        subsampled_labels: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        tree = cKDTree(subsampled_points, leafsize=40)
        labels = np.empty(len(points), dtype=np.int8)
        chunk_size = 1_000_000

        starts = range(0, len(points), chunk_size)
        if self.pbar_bool:
            starts = tqdm(starts, desc="Label upsampling", unit="chunk", leave=False)

        for start in starts:
            end = min(start + chunk_size, len(points))
            _, indices = tree.query(
                points[start:end],
                k=1,
                workers=self.query_workers,
            )
            labels[start:end] = subsampled_labels[indices]
        return labels

    def segment_pcd(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points)
        intensity = np.asarray(intensity).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
        if len(points) == 0:
            raise ValueError("point cloud cannot be empty")
        if len(intensity) != len(points):
            raise ValueError("points and intensity must have the same length")
        if np.any(intensity < 0):
            raise ValueError("intensity cannot contain negative values")

        normalized_points = points.astype(np.float64, copy=True)
        normalized_points -= normalized_points.mean(axis=0)
        normalized_points = normalized_points.astype(np.float32)

        normalized_intensity = intensity.astype(np.float32, copy=True)
        maximum_intensity = normalized_intensity.max()
        if maximum_intensity > 0:
            normalized_intensity /= maximum_intensity
        else:
            normalized_intensity.fill(0)

        source_indices = np.arange(len(points), dtype=np.int64)
        subsampled_points, subsampled_features, _ = voxel_subsample_vectorized(
            normalized_points,
            normalized_intensity[:, None],
            source_indices,
            self.voxel_size,
        )
        subsampled_intensity = subsampled_features[:, 0]

        subsampled_labels = self._segment_subsampled(
            subsampled_points,
            subsampled_intensity,
        )
        return self._upsample_labels(
            subsampled_points,
            subsampled_labels,
            normalized_points,
        )
