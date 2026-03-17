import numpy as np
import laspy
import pickle
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
import shutil


# ------------------------------------------------------------------
# 1. Load + normalise intensity
# ------------------------------------------------------------------
def load_and_normalise(las_path):
    las       = laspy.read(las_path)
    xyz       = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
    intensity = np.array(las.intensity, dtype=np.float32)
    intensity = intensity / (intensity.max() + 1e-6)
    feats     = intensity[:, None]
    labels    = np.array(las.classification, dtype=np.int32)
    del las
    return xyz, feats, labels


# ------------------------------------------------------------------
# 2. Voxel subsampling — closest point to cell center, one pass
# ------------------------------------------------------------------
def voxel_subsample(xyz, feats, labels, voxel_size=0.10):
    keys     = np.floor(xyz / voxel_size).astype(np.int32)
    centers  = (keys + 0.5) * voxel_size
    dists_sq = np.sum((xyz - centers) ** 2, axis=1)

    best_dist = {}
    best_idx  = {}

    with tqdm(total=len(xyz), desc="  Voxel subsample", unit="pt",
              leave=False) as pbar:
        for i in range(len(xyz)):
            k = (keys[i, 0], keys[i, 1], keys[i, 2])
            if k not in best_dist or dists_sq[i] < best_dist[k]:
                best_dist[k] = dists_sq[i]
                best_idx[k]  = i
            pbar.update(1)

    chosen = np.array(list(best_idx.values()), dtype=np.int32)
    return xyz[chosen], feats[chosen], labels[chosen]


# ------------------------------------------------------------------
# 3. Tile splitting
# ------------------------------------------------------------------
SKIP_CLASSES = {1, 7}

def iter_tiles(xyz, feats, labels, tile_size=40.0):
    mins     = xyz[:, :2].min(0)
    maxs     = xyz[:, :2].max(0)
    x_starts = np.arange(mins[0], maxs[0], tile_size)
    y_starts = np.arange(mins[1], maxs[1], tile_size)

    total = len(x_starts) * len(y_starts)
    with tqdm(total=total, desc="  Tiling", unit="cell", leave=False) as pbar:
        for i, x0 in enumerate(x_starts):
            for j, y0 in enumerate(y_starts):
                pbar.update(1)
                mask = (
                    (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + tile_size) &
                    (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + tile_size)
                )
                if mask.sum() < 512:
                    pbar.set_postfix_str("skip — too few pts")
                    continue

                tile_labels = labels[mask]
                if set(tile_labels.tolist()).issubset(SKIP_CLASSES):
                    pbar.set_postfix_str("skip — only ground/tree")
                    continue

                tile_xyz        = xyz[mask].copy()
                tile_xyz[:, 0] -= x0 + tile_size / 2
                tile_xyz[:, 1] -= y0 + tile_size / 2

                pbar.set_postfix_str(f"{mask.sum():,} pts")
                yield tile_xyz, feats[mask].copy(), tile_labels, (i, j)


# ------------------------------------------------------------------
# 4. Save tiles as .npy + precomputed KDTree as .pkl
# ------------------------------------------------------------------
def save_tiles(las_path, cut_dir, voxel_size=0.10, tile_size=40.0):
    cut_dir = Path(cut_dir)
    cut_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(las_path).stem
    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"Processing: {las_path}")

    tqdm.write("  Loading + normalising intensity...")
    xyz, feats, labels = load_and_normalise(las_path)
    tqdm.write(f"  raw: {len(xyz):,} pts")

    xyz, feats, labels = voxel_subsample(xyz, feats, labels, voxel_size)
    tqdm.write(f"  after voxel subsample ({voxel_size}m): {len(xyz):,} pts")

    saved   = 0
    skipped = 0
    for tile_xyz, tile_feats, tile_labels, (ti, tj) in iter_tiles(
        xyz, feats, labels, tile_size
    ):
        name = f"{stem}_tile_{ti:03d}_{tj:03d}"

        arr = np.concatenate(
            [tile_xyz, tile_feats, tile_labels[:, None].astype(np.float32)],
            axis=1
        )
        np.save(cut_dir / f"{name}.npy", arr)

        tree = KDTree(tile_xyz)
        with open(cut_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(tree, f)

        saved += 1

    tqdm.write(f"  saved {saved} tiles  |  skipped {skipped} tiles")


# ------------------------------------------------------------------
# 5. Train / val / test split
# ------------------------------------------------------------------
def split_dataset(cut_dir, out_dir, train=0.7, val=0.15, test=0.15, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6

    cut_dir = Path(cut_dir)
    out_dir = Path(out_dir)

    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    tiles      = sorted(cut_dir.glob("*.npy"))
    scan_names = sorted(set("_".join(t.stem.split("_")[:-3]) for t in tiles))

    rng     = np.random.default_rng(seed)
    order   = rng.permutation(len(scan_names))
    n_train = int(len(scan_names) * train)
    n_val   = int(len(scan_names) * val)

    train_scans = set(scan_names[i] for i in order[:n_train])
    val_scans   = set(scan_names[i] for i in order[n_train:n_train + n_val])

    counts = {"train": 0, "val": 0, "test": 0}
    for npy_path in tqdm(tiles, desc="Splitting", unit="tile"):
        scan     = "_".join(npy_path.stem.split("_")[:-3])
        pkl_path = npy_path.with_suffix(".pkl")

        split = (
            "train" if scan in train_scans else
            "val"   if scan in val_scans   else
            "test"
        )
        shutil.copy(npy_path, out_dir / split / npy_path.name)
        shutil.copy(pkl_path, out_dir / split / pkl_path.name)
        counts[split] += 1

    print(f"\nSplit complete: {counts}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    LAS_FILES  = sorted(Path("/home/kalmary/Dokumenty/tree_data/FULL_LAZ").glob("*.laz"))
    CUT_DIR    = Path("/home/kalmary/Dokumenty/tree_data/CUT")
    SPLIT_DIR  = Path("/home/kalmary/Dokumenty/tree_data/SPLIT")
    VOXEL_SIZE = 0.10
    TILE_SIZE  = 40.0

    for las_path in tqdm(LAS_FILES, desc="Files", unit="file"):
        save_tiles(las_path, CUT_DIR, VOXEL_SIZE, TILE_SIZE)

    split_dataset(CUT_DIR, SPLIT_DIR)