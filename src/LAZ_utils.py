import numpy as np
import open3d as o3d
import laspy
import contextlib
import logging
import os
import pathlib as pth
import sys


TREE_IDS_DIM = 'tree_ids'
TREE_SPECIES_DIM = 'tree_species'
LEGACY_SPECIES_DIM = 'species'


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_stderr():
    try:
        sys.stderr.flush()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def read_las(file_path: pth.Path) -> laspy.LasData:
    try:
        with _quiet_stderr():
            return laspy.read(file_path)
    except Exception:
        logger.exception("Failed to read LAZ file: %s", file_path)
        raise


def as_las_classification(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if np.any((labels < 0) | (labels > np.iinfo(np.uint8).max)):
        raise ValueError("LAS classification labels must be in the uint8 range [0, 255].")
    return labels.astype(np.uint8, copy=False)


def as_tree_species(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    info = np.iinfo(np.int8)
    if np.any((labels < info.min) | (labels > info.max)):
        raise ValueError(f"tree_species labels must be in the int8 range [{info.min}, {info.max}].")
    return labels.astype(np.int8, copy=False)


def _extra_dim_names(las_or_header) -> set[str]:
    return set(las_or_header.point_format.extra_dimension_names)


def _dimension_names(las: laspy.LasData) -> set[str]:
    return set(las.point_format.dimension_names)


def _make_light_header(
    las: laspy.LasData,
    *,
    include_tree_ids: bool = True,
    include_tree_species: bool = True,
) -> laspy.LasHeader:
    new_header = las.header.copy()
    extra_names = list(new_header.point_format.extra_dimension_names)
    if extra_names:
        new_header.remove_extra_dims(extra_names)

    if include_tree_ids:
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=TREE_IDS_DIM, type="int32", description=TREE_IDS_DIM))
    if include_tree_species:
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=TREE_SPECIES_DIM, type="int8", description=TREE_SPECIES_DIM))

    return new_header


def _copy_dimension(source: laspy.LasData, target: laspy.LasData, name: str, indices=None) -> None:
    if name not in _dimension_names(source) or name not in _dimension_names(target):
        return

    values = np.asarray(source[name])
    if indices is not None:
        values = values[indices]

    if name == 'classification':
        values = as_las_classification(values)
    elif name == TREE_SPECIES_DIM:
        values = as_tree_species(values)

    target[name] = values


def _copy_standard_dimensions(source: laspy.LasData, target: laspy.LasData, indices=None) -> None:
    for name in target.point_format.standard_dimension_names:
        if name in {'X', 'Y', 'Z'}:
            continue
        _copy_dimension(source, target, name, indices)


def _source_tree_species(source: laspy.LasData, indices=None) -> np.ndarray | None:
    for name in (TREE_SPECIES_DIM, LEGACY_SPECIES_DIM):
        if name not in _extra_dim_names(source):
            continue
        values = np.asarray(source[name])
        if indices is not None:
            values = values[indices]
        return as_tree_species(values)
    return None


def create_new_las(file_path: pth.Path, new_file_path: pth.Path, exist_ok=False, light_format=True) -> pth.Path:
    
    if new_file_path.exists() and not exist_ok:
        new_file_path.unlink()
    elif new_file_path.exists() and exist_ok:
        return new_file_path


    las = read_las(file_path)
    new_header = _make_light_header(las) if light_format else las.header.copy()
    new_header.scales = las.header.scales
    new_header.offsets = las.header.offsets


    # zero_array_f = np.full(int(new_header.point_count), -1, dtype= np.float32)
    zero_array_i = np.full(int(new_header.point_count), -1, dtype= np.int32)
    zero_array_u8 = np.zeros(int(new_header.point_count), dtype=np.uint8)
    zero_array_i8 = np.full(int(new_header.point_count), -1, dtype=np.int8)

    existing_dims = _extra_dim_names(new_header)
    if TREE_IDS_DIM not in existing_dims:
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=TREE_IDS_DIM, type="int32", description=TREE_IDS_DIM))
    if TREE_SPECIES_DIM not in existing_dims:
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=TREE_SPECIES_DIM, type="int8", description=TREE_SPECIES_DIM))
    new_las = laspy.LasData(new_header)



    new_las.X = las.X
    new_las.Y = las.Y
    new_las.Z = las.Z

    _copy_standard_dimensions(las, new_las)

    new_las.classification = zero_array_u8

    new_las[TREE_IDS_DIM] = zero_array_i
    new_las[TREE_SPECIES_DIM] = zero_array_i8

    new_las.write(new_file_path)

    return new_file_path


def create_decimated_las(las, idxs, light_format=True) -> laspy.LasData:
    representative_indices = [idx[0] for idx in idxs if len(idx) > 0]

    source_extra_names = _extra_dim_names(las)
    has_tree_ids = TREE_IDS_DIM in source_extra_names
    has_tree_species = TREE_SPECIES_DIM in source_extra_names or LEGACY_SPECIES_DIM in source_extra_names

    if light_format:
        new_header = _make_light_header(
            las,
            include_tree_ids=has_tree_ids,
            include_tree_species=has_tree_species,
        )
    else:
        new_header = las.header.copy()
    new_header.point_count = 0
    new_header.scales = las.header.scales
    new_header.offsets = las.header.offsets
    new_las = laspy.LasData(header=new_header)


    new_las.X = np.asarray(las.X)[representative_indices]
    new_las.Y = np.asarray(las.Y)[representative_indices]
    new_las.Z = np.asarray(las.Z)[representative_indices]

    _copy_standard_dimensions(las, new_las, representative_indices)
    new_las.classification = as_las_classification(np.asarray(las.classification)[representative_indices])

    if has_tree_ids:
        new_las[TREE_IDS_DIM] = np.asarray(las[TREE_IDS_DIM], dtype=np.int32)[representative_indices]

    tree_species = _source_tree_species(las, representative_indices)
    if tree_species is not None:
        new_las[TREE_SPECIES_DIM] = tree_species


    new_las.update_header()

    return new_las

def downsample_laz_multiple(work_dir: pth.Path):
    for i, path in enumerate(work_dir.rglob('*.laz')):
        name = path.name
        if '.laz' in name and 'sqlite' not in name and '_mod' not in name and '_dec' not in name:
            las = read_las(path)
            points = np.vstack([las.x, las.y, las.z]).transpose()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            voxel_size = 0.1055
            pcd_down, _, point_indices = pcd.voxel_down_sample_and_trace(
                voxel_size=voxel_size,
                min_bound=points.min(0) - voxel_size * 0.5,
                max_bound=points.max(0) + voxel_size * 0.5
            )
            las = create_decimated_las(las, point_indices)

            parent_path = path.parent
            goal_folder = parent_path.joinpath('output_data')
            goal_folder.mkdir(exist_ok=True, parents=True)

            goal_path = goal_folder.joinpath(path.stem + '_dec.laz')

            las.write(goal_path)

def downsample_laz_single(file_path: pth.Path, output_dir: pth.Path) -> pth.Path:

    # parent_path = file_path.parent
    goal_folder = output_dir
    goal_folder.mkdir(exist_ok=True, parents=True)

    goal_path = goal_folder.joinpath(file_path.stem + '_dec.laz')

    if goal_path.exists() and goal_path.is_file(): # if given file was already decimated skip this step
        return goal_path

    las = read_las(file_path)
    points = np.vstack([las.x, las.y, las.z]).transpose()




    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_size = 0.105495
    pcd_down, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=points.min(0) - voxel_size * 0.5,
        max_bound=points.max(0) + voxel_size * 0.5
    )
    las = create_decimated_las(las, point_indices)

    del points, pcd, point_indices

    las.write(goal_path)

    return goal_path
