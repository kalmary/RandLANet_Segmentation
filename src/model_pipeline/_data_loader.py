import numpy as np
import pickle
import torch
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader


class CustomDataset(IterableDataset):
    """
    DALES-style iterable dataset over pre-cut .npy tiles.
    Each .npy tile has a paired .pkl KDTree built during preprocessing.

    File format: (N, 5) — xyz(3) + intensity(1) + label(1)

    Output per batch:
        xyz_feats : (B, num_points, 4)  — xyz + intensity
        labels    : (B, num_points)     — class index, long

    Possibility map:
        - spatially aware: initialised from local class distribution on coarse grid
        - common classes start just above coverage_thresh (rarely picked as centers)
        - rare classes start at 0 (picked first, sampled heavily)
        - epsilon controls how often common classes still get picked as centers
        - gaussian decay on update: edge points stay hungry longer than center
    """

    def __init__(self, data_dir,
                 num_points:            int        = 8192,
                 batch_size:            int        = 8,
                 buffer_size:           int        = 64,
                 shuffle:               bool       = True,
                 pos_weights:           np.ndarray = None,
                 coverage_thresh:       float      = 0.6,
                 epsilon:               float      = -0.05,
                 grid_res:              float      = 2.,
                 gaussian_sigma_factor: float      = 4.,
                 epoch:                 int        = 0):
        """
        coverage_thresh:       possibility threshold — point satisfied when poss >= this
        epsilon:               how far above coverage_thresh common classes start
                               larger = common classes sampled less often
        grid_res:              spatial grid cell size (metres) for regional class density
        gaussian_sigma_factor: controls width of gaussian update bell curve
                               larger = narrower bell = more edge revisits
        """
        self.files                 = sorted(Path(data_dir).glob("*.npy"))
        self.num_points            = num_points
        self.batch_size            = batch_size
        self.buffer_size           = buffer_size
        self.shuffle               = shuffle
        self.pos_weights           = pos_weights
        self.coverage_thresh       = coverage_thresh
        self.epsilon               = epsilon
        self.grid_res              = grid_res
        self.gaussian_sigma_factor = gaussian_sigma_factor
        self._epoch                = epoch

        assert len(self.files) > 0, f"No .npy files in {data_dir}"
        assert buffer_size >= batch_size, \
            f"buffer_size ({buffer_size}) must be >= batch_size ({batch_size})"

        missing = [f for f in self.files if not f.with_suffix(".pkl").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing KDTree .pkl for {len(missing)} tiles: {missing[:3]} ..."
            )

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    # ------------------------------------------------------------------
    # possibility map
    # ------------------------------------------------------------------
    def _build_possibility(self, xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
        xyz    = np.asarray(xyz,    dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        n      = len(labels)

        if self.pos_weights is None:
            return np.random.uniform(0, 1, n).astype(np.float32)

        w = self.pos_weights / (self.pos_weights.max() + 1e-6)  # rare=1, common≈0

        # coarse spatial grid — regional class density
        gx = np.floor(xyz[:, 0] / self.grid_res).astype(np.int64)
        gy = np.floor(xyz[:, 1] / self.grid_res).astype(np.int64)

        gy_range = int(gy.max() - gy.min() + 1)
        cell_enc = (gx - gx.min()) * gy_range + (gy - gy.min())

        _, inverse = np.unique(cell_enc, return_inverse=True)
        n_cells    = _.shape[0]

        point_w      = w[labels].astype(np.float64)
        cell_w_sum   = np.bincount(inverse, weights=point_w, minlength=n_cells).astype(np.float32)
        cell_w_count = np.bincount(inverse,                  minlength=n_cells).astype(np.float32)
        cell_w       = cell_w_sum / (cell_w_count + 1e-6)

        point_cell_w = cell_w[inverse]
        point_cls_w  = point_w.astype(np.float32)
        combined_w   = 0.5 * point_cell_w + 0.5 * point_cls_w  # (N,) in [0, 1]

        # rare   (combined_w=1.0) → poss = 0.0
        # common (combined_w=0.0) → poss = coverage_thresh + epsilon  (just above thresh)
        poss  = (self.coverage_thresh + self.epsilon) * (1.0 - combined_w)
        poss += np.random.uniform(0, 0.05, n)   # jitter — common classes occasionally dip below thresh
        return poss.astype(np.float32)

    # ------------------------------------------------------------------
    # core: one file
    # ------------------------------------------------------------------
    def _iter_file(self, npy_path: Path):
        data   = np.load(npy_path)
        xyz    = data[:, :3].astype(np.float32)
        feats  = data[:, 3:4].astype(np.float32)
        labels = data[:, 4].astype(np.int32)
        del data

        with open(npy_path.with_suffix(".pkl"), "rb") as f:
            kdtree = pickle.load(f)

        poss  = self._build_possibility(xyz, labels)
        gap   = poss - self.coverage_thresh   # single uniform threshold

        while gap.min() < 0:
            center_idx   = int(np.argmin(gap))
            center       = xyz[center_idx]

            neighbor_idx = kdtree.query(
                center[None], k=self.num_points, return_distance=False
            )[0].astype(np.int32)

            dists = np.linalg.norm(xyz[neighbor_idx] - center, axis=1)
            sigma = dists.max() / self.gaussian_sigma_factor
            delta = np.exp(-(dists ** 2) / (2 * sigma ** 2 + 1e-6))

            poss[neighbor_idx] += delta
            gap[neighbor_idx]  += delta

            xyz_feats = np.concatenate(
                [xyz[neighbor_idx] - center, feats[neighbor_idx]], axis=1
            )
            yield xyz_feats, labels[neighbor_idx].astype(np.int64)

        del xyz, feats, labels, kdtree, poss, gap

    # ------------------------------------------------------------------
    # buffer → batches
    # ------------------------------------------------------------------
    def _iter_batched(self, file_order, rng):
        buffer = []

        def drain(flush=False):
            if self.shuffle:
                rng.shuffle(buffer)

            while len(buffer) >= self.batch_size:
                batch       = buffer[:self.batch_size]
                del buffer[:self.batch_size]
                xyz_feats_b = np.stack([s[0] for s in batch])
                labels_b    = np.stack([s[1] for s in batch])

                if self.shuffle:
                    for b in range(self.batch_size):
                        perm           = rng.permutation(self.num_points)
                        xyz_feats_b[b] = xyz_feats_b[b][perm]
                        labels_b[b]    = labels_b[b][perm]

                yield (
                    torch.from_numpy(xyz_feats_b),           # (B, N, 4)
                    torch.from_numpy(labels_b).long(),        # (B, N)
                )

            if flush and buffer:
                batch = buffer[:]
                del buffer[:]
                yield (
                    torch.from_numpy(np.stack([s[0] for s in batch])),
                    torch.from_numpy(np.stack([s[1] for s in batch])).long(),
                )

        for file_idx in file_order:
            for sample in self._iter_file(self.files[file_idx]):
                buffer.append(sample)
                if len(buffer) >= self.buffer_size:
                    yield from drain()

        yield from drain(flush=True)

    # ------------------------------------------------------------------
    # __iter__
    # ------------------------------------------------------------------
    def __iter__(self):
        rng   = np.random.default_rng(self._epoch)
        order = list(range(len(self.files)))

        if self.shuffle:
            order = rng.permutation(order).tolist()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            order = order[worker_info.id :: worker_info.num_workers]

        yield from self._iter_batched(order, rng)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def compute_pos_weights(data_dir, num_classes: int,
                        power: float = 0.25) -> np.ndarray:
    """
    Inverse frequency weights with power dampening.
    power=1.0 → raw inverse freq
    power=0.5 → sqrt dampening
    power=0.25 → fourth root (default, mild compression)
    power=0.0 → uniform
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in sorted(Path(data_dir).glob("*.npy")):
        labels  = np.load(path)[:, 4].astype(np.int32)
        counts += np.bincount(labels, minlength=num_classes)

    weights              = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    weights              = (weights / weights.max()).astype(np.float32)
    return weights


def make_loader(data_dir,
                num_points:            int        = 8192,
                batch_size:            int        = 8,
                buffer_size:           int        = 64,
                num_workers:           int        = 4,
                shuffle:               bool       = True,
                pos_weights:           np.ndarray = None,
                coverage_thresh:       float      = 0.5,
                epsilon:               float      = 0.05,
                grid_res:              float      = 5.0,
                gaussian_sigma_factor: float      = 3.0,
                epoch:                 int        = 0) -> tuple[DataLoader, CustomDataset]:

    ds = CustomDataset(
        data_dir               = data_dir,
        num_points             = num_points,
        batch_size             = batch_size,
        buffer_size            = buffer_size,
        shuffle                = shuffle,
        pos_weights            = pos_weights,
        coverage_thresh        = coverage_thresh,
        epsilon                = epsilon,
        grid_res               = grid_res,
        gaussian_sigma_factor  = gaussian_sigma_factor,
        epoch                  = epoch,
    )
    loader = DataLoader(
        ds,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = False,
    )
    return loader, ds


# ------------------------------------------------------------------
# usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    TRAIN_DIR    = "data/split/train"
    NUM_CLASSES  = 10
    WEIGHTS_PATH = Path("pos_weights.npy")

    if not WEIGHTS_PATH.exists():
        loader, ds = make_loader(TRAIN_DIR, shuffle=False)
        for epoch in range(3):
            ds.set_epoch(epoch)
            for xyz_feats, labels in loader:
                pass

        pos_weights = compute_pos_weights(TRAIN_DIR, NUM_CLASSES)
        np.save(WEIGHTS_PATH, pos_weights)
        print("pos_weights:", pos_weights)

    pos_weights = np.load(WEIGHTS_PATH)
    loader, ds  = make_loader(
        TRAIN_DIR,
        pos_weights           = pos_weights,
        shuffle               = True,
        epsilon               = 0.05,
        grid_res              = 5.0,
        gaussian_sigma_factor = 3.0,
    )

    for epoch in range(100):
        ds.set_epoch(epoch)
        for xyz_feats, labels in loader:
            # xyz_feats : (B, num_points, 4)
            # labels    : (B, num_points)
            pass