import argparse
from typing import Union
import pathlib as pth
import h5py
import shutil
from tqdm import tqdm
import sys

import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import laspy
import numpy as np

import sys
main_dir = pth.Path(__file__).parent.parent
sys.path.append(str(main_dir))

from utils.pcd_manipulation import voxelGridFragmentation
from utils import convert_str_values, load_json, save2json

def decimate_chunk_laz(work_dir: pth.Path, goal_dir: pth.Path, folder_split: dict) -> None:
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

                with laspy.open(path) as f:
                    total_points = f.header.point_count


                try:
                    las = laspy.read(path)
                except Exception as e:
                    print(e)

                points = np.vstack((las.x, las.y, las.z)).transpose()
                points = points - np.mean(points, axis =0)


                classification = np.asarray(las.classification, dtype=np.int32)

                points = points[classification>cut_label]

                intensity = np.asarray(las.intensity, dtype=np.float32)
                intensity = scaler.fit_transform(intensity.reshape(-1, 1))
                # intensity = intensity / (2 ** 16 - 1)
                intensity = intensity.flatten()

                intensity = intensity[classification>cut_label]

                classification = classification[classification >cut_label]
                classification -= 1 # TODO remove cut cabel part for official repo use


                for i, (sampled_idx, noise) in enumerate(voxelGridFragmentation(points,
                                                                                voxel_size=np.array([20., 20.]), #TODO check if it works, update in other places
                                                                                num_points=2*8192,
                                                                                shuffle=True)):
                    if noise:
                        continue

                    points_chunk = points[sampled_idx]
                    points_chunk -= np.mean(points, axis = 0)

                    intensity_chunk = intensity[sampled_idx]
                    classification_chunk = classification[sampled_idx]

                    if np.unique(classification_chunk).flatten().shape[0] < 3: # TODO a way to avoid imbalance of dataset with huge number of ground points.
                        continue

                    chunk = np.concatenate([points_chunk,
                                            intensity_chunk.reshape(-1, 1),
                                            classification_chunk.reshape(-1, 1)],
                                            axis = 1)
                    
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

    def convert2_h5(path_list: list[Union[str, pth.Path]], mode: int):
        available_modes = {0: 'train', 1: 'test', 2: 'val'}
        if mode not in [0, 1, 2]:
            raise ValueError(f'Incorrect mode: {mode}\nAvailable modes: {available_modes}')
        
        match mode:
            case 0:
                goal_file = goal_dir.joinpath('train.h5')
                h5_file = h5py.File(goal_file, 'w')
            case 1:
                goal_file = goal_dir.joinpath('test.h5')
                h5_file = h5py.File(goal_file, 'w')
            case 2:
                goal_file = goal_dir.joinpath('validation.h5')
                h5_file = h5py.File(goal_file, 'w')
        
        chunk2save = np.zeros((0, chunk_num_point, 5))
        chunk_num = 0

        for path in tqdm(path_list, desc=f'Training folder, copying data {pth.Path(path_list[0]).parent} ---> {goal_file.name}',
                        total=len(path_list)):
            points = np.load(path)
            points = np.expand_dims(points, axis=0)


            chunk2save = np.concatenate([chunk2save, points], axis=0)
            if chunk2save.shape[0] >= chunk_h5_shape:
                h5_file.create_dataset(str(chunk_num), data=chunk2save)
                chunk2save = np.zeros((0, chunk_num_point, 5))
                chunk_num += 1

        if chunk2save.shape[0] > 0:
            h5_file.create_dataset(str(chunk_num), data=chunk2save)
        h5_file.close()

    convert2_h5(train_paths, 0)
    convert2_h5(test_paths, 1)
    convert2_h5(validation_paths, 2)

    return goal_dir.joinpath('train.h5'), goal_dir.joinpath('test.h5'), goal_dir.joinpath('validation.h5')




def rebalance_dataset(work_dir: pth.Path, folder_split: dict, tolerance=0.03):
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
        np.random.shuffle(files)
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

def argparser():
        
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
        '--source_path',
        type=str,
        help=(
            "Dir path with raw, labelled .LAZ files to process."
        )
    )

    parser.add_argument(
        '--decimated_path',
        type=str,
        help=(
            "Checkpoint path with cut, distributed but non-converted files."
        )
    )

    parser.add_argument(
        '--converted_path',
        type=str,
        help=(
            "Final path with files meant for further computations with model pipeline."
        )
    )

    parser.add_argument(
        '--folder_split',
        type=Union[list[int], list[str]],
        default=[0.7, 0.2, 0.1],
        help=(
            "Folder split ratios for train, test, validation."
        )
    )

    return parser.parse_args()


def update_paths_config(path2train: pth.Path, path2test: pth.Path, path2val: pth.Path):


    def _update_path(path2dataset: Union[str, pth.Path], dataset_name: str):

        config_dir = pth.Path(__file__).parent.parent.joinpath('model_pipeline/training_configs')
        
        path2config_single = config_dir.joinpath(f'config_train_single.json')
        path2config = config_dir.joinpath(f'config_train.json')

        config_single = load_json(path2config_single)
        config = load_json(path2config)

        config_single[dataset_name] = str(path2dataset)
        config[dataset_name] = str(path2dataset)

        save2json(config_single, path2config_single)
        save2json(config, path2config)

    _update_path(path2train, 'data_path_train')
    _update_path(path2test, 'data_path_test')
    _update_path(path2val, 'data_path_val')




def main():
    parser = argparser()

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

    



if __name__ == '__main__':
    main()
