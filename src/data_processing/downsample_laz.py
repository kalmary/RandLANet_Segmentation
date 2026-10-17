import argparse
import pathlib as pth
import random
import runpy
import shutil
import sys
from collections.abc import Sequence

_preprocessing_dependencies_loaded = False


def _load_preprocessing_dependencies():
    global _preprocessing_dependencies_loaded
    global MinMaxScaler, convert_str_values, h5py, laspy, load_json, np
    global save_to_json, tqdm, train_test_split, voxel_grid_fragmentation

    if _preprocessing_dependencies_loaded:
        return

    import h5py as h5py_module
    import laspy as laspy_module
    import numpy as numpy_module
    from sklearn.model_selection import train_test_split as split_dataset
    from sklearn.preprocessing import MinMaxScaler as min_max_scaler
    from tqdm import tqdm as progress_bar

    from ..utils.pcd_manipulation import (
        voxel_grid_fragmentation as fragment_voxel_grid,
    )
    from ..utils.nn_utils import (
        convert_str_values as convert_values,
        load_json as load_config,
        save_to_json as save_config,
    )

    h5py = h5py_module
    laspy = laspy_module
    np = numpy_module
    train_test_split = split_dataset
    MinMaxScaler = min_max_scaler
    tqdm = progress_bar
    voxel_grid_fragmentation = fragment_voxel_grid
    convert_str_values = convert_values
    load_json = load_config
    save_to_json = save_config
    _preprocessing_dependencies_loaded = True

def decimate_chunk_laz(work_dir: pth.Path, goal_dir: pth.Path, folder_split: dict) -> None:
    _load_preprocessing_dependencies()
    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)

    goal_dir.mkdir(parents=True, exist_ok=True)

    train_pth = goal_dir.joinpath('train')
    train_pth.mkdir(exist_ok=True, parents=True)

    test_pth = goal_dir.joinpath('test')
    test_pth.mkdir(exist_ok=True, parents=True)

    val_pth = goal_dir.joinpath('val')
    val_pth.mkdir(exist_ok=True, parents=True)

    all_paths = list(work_dir.rglob('*.las')) # todo change to desired format later
    random.shuffle(all_paths)

    train_paths, test_paths = train_test_split(all_paths, 
                                               train_size=folder_split['train_ratio'],
                                               random_state=42, 
                                               shuffle=True)

    test_paths, val_paths = train_test_split(test_paths,
                                             train_size=folder_split['test_ratio'] /
                                                                    (folder_split['test_ratio'] + folder_split['val_ratio']),
                                             random_state=42, 
                                             shuffle=True)



    # random.shuffle(all_paths)
    progress_train = tqdm(enumerate(train_paths), desc = f'Decimation of training data in folder: {work_dir}', total=len(train_paths))
    progress_test = tqdm(enumerate(test_paths), desc=f'Decimation of testing data in folder: {work_dir}',
                          total=len(test_paths))
    progress_val = tqdm(enumerate(val_paths), desc=f'Decimation of validation data in folder: {work_dir}',
                          total=len(val_paths))

    scaler = MinMaxScaler(feature_range=(0, 10.))

    def decimate_folder(generator, goal):
        cut_label = 0
        for _, path in generator:
            
                n = 0
                chunk_num = 0


                try:
                    las = laspy.read(path)
                    total_points = len(las.points)
                except Exception as e:
                    print(e)
                    continue

                points = np.vstack(
                    (
                        np.asarray(las.x),
                        np.asarray(las.y),
                        np.asarray(las.z),
                    )
                ).transpose()
                points = points.astype(np.float32, copy=False)

                points = points - np.mean(points, axis =0)


                classification = np.asarray(las.classification, dtype=np.uint8)
                intensity = np.asarray(las.intensity, dtype=np.float32)
                

                valid_mask = classification > cut_label
                points = points[valid_mask]
                intensity = intensity[valid_mask]
                intensity = scaler.fit_transform(intensity.reshape(-1, 1))
                intensity = intensity.flatten().astype(np.float32, copy=False)
                
                classification = classification[valid_mask]
                classification = (classification - 1).astype(np.uint8, copy=False)

                for i_0, (sampled_idx_0, noise_0) in enumerate(voxel_grid_fragmentation(points,
                                                                                      voxel_size=np.array([200., 200.]),
                                                                                      overlap_ratio=0.,
                                                                                      num_points=0,
                                                                                      shuffle=True)):
                    if noise_0:
                        continue

                    classification_chunk_0 = classification[sampled_idx_0]
                    if np.unique(classification_chunk_0).flatten().shape[0] < 3:
                        continue

                    points_chunk_0 = points[sampled_idx_0]
                    intensity_chunk_0 = intensity[sampled_idx_0]



                    for i, (sampled_idx, noise) in enumerate(voxel_grid_fragmentation(points_chunk_0,
                                                                                    voxel_size=np.array([20., 20.]), #TODO check if it works, update in other places
                                                                                    overlap_ratio=0.25,
                                                                                    num_points=2*8192,
                                                                                    shuffle=True)):
                        if noise:
                            continue

                        points_chunk = points_chunk_0[sampled_idx]
                        points_chunk -= np.mean(points_chunk, axis = 0)

                        intensity_chunk = intensity_chunk_0[sampled_idx]
                        classification_chunk = classification_chunk_0[sampled_idx]

                        if np.unique(classification_chunk).flatten().shape[0] < 3: # TODO a way to avoid imbalance of dataset with huge number of ground points.
                            continue

                        chunk = np.concatenate([points_chunk.astype(np.float32, copy=False),
                                                intensity_chunk.reshape(-1, 1),
                                                classification_chunk.reshape(-1, 1)],
                                                axis = 1).astype(np.float32, copy=False)
                        
                        n_org = points_chunk.shape[0]
                        chunk_num += 1

                        generator.set_postfix({
                            'Points': f"{n}/ {total_points}, ({n_org} -> {points_chunk.shape[0]})",
                            'Partitioning': f"{i}"
                        })

                        file_name = goal.joinpath(path.stem+f'_{chunk_num}_{i}.npy')
                        np.save(file_name, chunk)

                        n+=n_org

    decimate_folder(progress_train, train_pth)
    decimate_folder(progress_test, test_pth)
    decimate_folder(progress_val, val_pth)




def convert_dataset(work_dir: pth.Path, goal_dir: pth.Path) -> tuple[pth.Path, pth.Path, pth.Path]:
    _load_preprocessing_dependencies()
    work_train = work_dir.joinpath('train')
    work_test = work_dir.joinpath('test')
    work_val = work_dir.joinpath('val')

    chunk_num_point = 2*8192
    chunk_h5_shape = 30

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)
    if not goal_dir.exists():
        raise ValueError('Incorrect path:', goal_dir)

    train_paths = list(work_train.rglob('*.npy'))
    test_paths = list(work_test.rglob('*.npy'))
    validation_paths = list(work_val.rglob('*.npy'))

    def convert2_h5(path_list: Sequence[str | pth.Path], mode: int):
        available_modes = {
            0: 'train.h5',
            1: 'test.h5',
            2: 'validation.h5',
        }
        if mode not in available_modes:
            raise ValueError(f'Incorrect mode: {mode}\nAvailable modes: {available_modes}')

        goal_file = goal_dir.joinpath(available_modes[mode])
        source_dir = pth.Path(path_list[0]).parent if path_list else work_dir
        
        chunk2save = np.zeros((0, chunk_num_point, 5), dtype=np.float32)
        chunk_num = 0

        with h5py.File(goal_file, 'w') as h5_file:
            for path in tqdm(
                path_list,
                desc=f'Training folder, copying data {source_dir} ---> {goal_file.name}',
                total=len(path_list),
            ):
                points = np.load(path)
                points = np.expand_dims(points, axis=0)


                chunk2save = np.concatenate([chunk2save, points], axis=0)
                if chunk2save.shape[0] >= chunk_h5_shape:
                    h5_file.create_dataset(str(chunk_num), data=chunk2save)
                    chunk2save = np.zeros((0, chunk_num_point, 5), dtype=np.float32)
                    chunk_num += 1

            if chunk2save.shape[0] > 0:
                h5_file.create_dataset(str(chunk_num), data=chunk2save)

    convert2_h5(train_paths, 0)
    convert2_h5(test_paths, 1)
    convert2_h5(validation_paths, 2)

    return goal_dir.joinpath('train.h5'), goal_dir.joinpath('test.h5'), goal_dir.joinpath('validation.h5')




def rebalance_dataset(work_dir: pth.Path, folder_split: dict, tolerance=0.03):
    _load_preprocessing_dependencies()
    work_train = work_dir.joinpath('train')
    work_test = work_dir.joinpath('test')
    work_val = work_dir.joinpath('val')

    work_pths = [work_train, work_test, work_val]
    split_keys = ['train_ratio', 'test_ratio', 'val_ratio']
    folder_ratio = [folder_split[k] for k in split_keys]

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)


    # Count .npy files
    counts = np.array([len(list(p.rglob('*.npy'))) for p in work_pths])
    total = counts.sum()

    print(f"File counts before balancing: train={counts[0]}, test={counts[1]}, val={counts[2]}")

    if total == 0:
        raise RuntimeError("No .npy files found in dataset folders.")

    current_ratio = counts / total
    desired_ratio = np.array(folder_ratio)

    if np.all(np.abs(current_ratio - desired_ratio) <= tolerance):
        print("No rebalancing needed. Folders are within desired ratio.")
        return

    # Compute target file counts per folder
    target_counts = np.round(desired_ratio * total).astype(int)
    diffs = counts - target_counts

    surplus = np.where(diffs > 0)[0]
    deficit = np.where(diffs < 0)[0]

    move_log = {}

    for s in surplus:
        files = list(work_pths[s].rglob('*.npy'))
        np.random.shuffle(files)  # pyright: ignore[reportArgumentType]
        for d in deficit:
            move_n = min(diffs[s], -diffs[d])
            if move_n <= 0:
                continue

            to_move = files[:move_n]
            for f in to_move:
                dest = work_pths[d] / f.name
                if dest.exists():
                    raise FileExistsError(f"Destination file already exists: {dest}")
                shutil.move(str(f), str(dest))

            move_log[f"{split_keys[s]} -> {split_keys[d]}"] = move_n
            diffs[s] -= move_n
            diffs[d] += move_n
            files = files[move_n:]

            if np.all(diffs == 0):
                break

    new_counts = [len(list(p.rglob('*.npy'))) for p in work_pths]
    print(f"File counts after balancing: train={new_counts[0]}, test={new_counts[1]}, val={new_counts[2]}")
    print("Move summary:", move_log)

def argparser(argv=None):
        
    """
    Parse command-line arguments for automated point cloud data processing for semantic segmentation.
    Returns parsed arguments: source_path, decimated_path, converted_path
    """

    parser = argparse.ArgumentParser(
        description="Script for preprocessing .LAZ files. Each point cloud is fragmented into voxels, decimated and stored in:\n" \
        "1. .npy files - checkpoint part, files cut, but not converted to faster format\n" \
        "2. .hdf5 files - files used during training/ validation/ testing. Fast format and chunked data work well with HDD disks and low RAM.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--source-path',
        type=str,
        help=(
            "Dir path with raw, labelled .LAZ files to process."
        )
    )

    parser.add_argument(
        '--decimated-path',
        type=str,
        help=(
            "Checkpoint path with cut, distributed but non-converted files."
        )
    )

    parser.add_argument(
        '--converted-path',
        type=str,
        help=(
            "Final path with files meant for further computations with model pipeline."
        )
    )

    parser.add_argument(
        '--folder-split',
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        help=(
            "Folder split ratios for train, test, validation."
        )
    )

    return parser.parse_args(argv)


def update_paths_config(path2train: pth.Path, path2test: pth.Path, path2val: pth.Path):
    _load_preprocessing_dependencies()


    def _update_path(path2dataset: str | pth.Path, dataset_name: str):

        config_dir = pth.Path(__file__).parent.parent.joinpath('model_pipeline/training_configs')
        
        path2config_single = config_dir.joinpath('config_train_single.json')
        path2config = config_dir.joinpath('config_train.json')

        config_single = load_json(path2config_single)
        config = load_json(path2config)

        config_single[dataset_name] = str(path2dataset)
        config[dataset_name] = str(path2dataset)

        save_to_json(config_single, path2config_single)
        save_to_json(config, path2config)

    _update_path(path2train, 'data_path_train')
    _update_path(path2test, 'data_path_test')
    _update_path(path2val, 'data_path_val')


def test_argparser_parses_explicit_folder_split():
    args = argparser(
        [
            '--folder-split',
            '0.6',
            '0.25',
            '0.15',
        ]
    )

    assert args.folder_split == [0.6, 0.25, 0.15]


def test_convert_dataset_creates_empty_split_files(tmp_path):
    work_dir = tmp_path / 'work'
    goal_dir = tmp_path / 'converted'
    for split in ('train', 'test', 'val'):
        work_dir.joinpath(split).mkdir(parents=True)
    goal_dir.mkdir()

    output_paths = convert_dataset(work_dir, goal_dir)

    assert output_paths == (
        goal_dir / 'train.h5',
        goal_dir / 'test.h5',
        goal_dir / 'validation.h5',
    )
    for output_path in output_paths:
        with h5py.File(output_path, 'r') as output_file:
            assert list(output_file.keys()) == []




def main():
    parser = argparser()
    _load_preprocessing_dependencies()

    source = parser.source_path
    source = pth.Path(source)

    decimated = parser.decimated_path
    decimated = pth.Path(decimated)

    converted = parser.converted_path
    converted = pth.Path(converted)

    folder_split = {
        'train_ratio': parser.folder_split[0],
        'test_ratio': parser.folder_split[1],
        'val_ratio': parser.folder_split[2]
    }
    folder_split = convert_str_values(folder_split)

    decimate_chunk_laz(source, decimated, folder_split)
    rebalance_dataset(decimated, folder_split)

    path2train, path2test, path2val = convert_dataset(decimated, converted)

    update_paths_config(path2train, path2test, path2val)

    



def _run_direct_entry_point() -> None:
    project_root = str(pth.Path(__file__).resolve().parents[2])
    sys.path.insert(0, project_root)
    try:
        runpy.run_module("src.data_processing.downsample_laz", run_name="__main__")
    finally:
        sys.path.remove(project_root)


if __name__ == '__main__':
    if __package__:
        main()
    else:
        _run_direct_entry_point()
