from typing import Union, Optional, Any, Tuple, Dict, List
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import pathlib as pth
import os
import sys
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn


from model_pipeline.RandLANet_CB import RandLANet
from utils import load_json, load_model, pcd_manipulation

class SegmentClass:
    def __init__(self,
                 voxel_size_big: float = 200.,
                 overlap: float = 0.4,
                 model_name: str = None,
                 config_dir: Union[str, pth.Path] = "final_files",
                 device: torch.device = torch.device('cpu'),
                 pbar_bool: bool = False):
        
        self.voxel_size_small = None
        self.voxel_size_big = voxel_size_big
        self.overlap = overlap
        if model_name is None:
            raise ValueError("model_name cannot be None")
        self.model_name = model_name + '.pt'

        if isinstance(device, str):
            self.device = torch.device(device)
        self.device = device

        self.pbar_bool = pbar_bool

        self._scaler = None
        
        self.base_path = pth.Path(__file__).parent
        config_dir = self.base_path.joinpath(config_dir)
        self._model_config = self._load_config(config_dir)
        self._model = self._load_segmModel(config_dir)
        self._scaler = self._init_scaler()


    # TODO adjust model loading 
    def _load_config(self, config_dir: Optional[Union[pth.Path, str]] = None) -> dict:

        config_path = pth.Path(config_dir).joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        self._model_config: dict = config_dict['model_config']
        self.voxel_size_small: float = self.model_config['max_voxel_dim']

        return config_dict

    def _load_segmModel(self, model_dir: Union[pth.Path, str] = "./final_files") -> nn.Module:

        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = RandLANet(self._model_config["model_config"], self._model_config['num_classes'])
        self._model: nn.Module = load_model(file_path=path2model,
                                            model=model,
                                            device=self.device)
        self._model.eval()
    

        return model
    
    def _init_scaler(self, feature_range: Tuple[int] = (0, 10)) -> MinMaxScaler:
        self._scaler = MinMaxScaler(feature_range)
        return self._scaler
    
    @property
    def model_config(self) -> dict:
        return self._model_config
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def scaler(self) -> MinMaxScaler:
        return self._scaler
    
    @staticmethod
    def _worker_task(chunk_data: Tuple[int, int],
                     voxel_probs_all: np.ndarray,
                     points: np.ndarray,
                     shm_info: Dict[str, Any], # Nowy argument
                     tree: KDTree,
                     k_neighbors: int,
                     distance_sigma: float):
    
        # Add to output array SHM
        shm_out = shared_memory.SharedMemory(name=shm_info['name'])
        points_probs_view = np.ndarray(
            shm_info['shape'], 
            dtype=shm_info['dtype'], 
            buffer=shm_out.buf
        )
        
        start_idx, end_idx = chunk_data
        points_chunk = points[start_idx:end_idx] 

        # kd tree (shared) query
        dists, indices = tree.query(points_chunk, k=k_neighbors) 
        
        for local_i in range(points_chunk.shape[0]):
            global_i = start_idx + local_i
            
            neighbor_probs = voxel_probs_all[indices[local_i]]
            neighbor_dists = dists[local_i]

            weights = np.exp(- (neighbor_dists ** 2) / (2 * distance_sigma ** 2))
            weights_sum = weights.sum()

            if weights_sum > 0:
                weights /= weights_sum
                points_probs_view[global_i] = np.sum(neighbor_probs * weights[:, np.newaxis], axis=0)
            else:
                points_probs_view[global_i] = 0.0
            
        # Close local reference to SHM
        shm_out.close()
        
        return None

    def _upsample_labeled_chunk_parallel(self, 
                                        voxel_all: np.array,
                                        voxel_probs_all: np.array,
                                        points: np.array,
                                        k_neighbors_upsampling: int = 14,
                                        distance_sigma: float = 0.35,
                                        num_workers: int = -1,
                                        pbar: bool = None) -> np.array:
        
        shm_points_probs: Optional[shared_memory.SharedMemory] = None
        
        available_workers = os.cpu_count()
        if num_workers == -1 or num_workers == 0 or available_workers < num_workers:
            num_workers = available_workers
        else:
            num_workers = available_workers

        
        try:
            # 1. build kdtree
            # voxel_all is shared as read only in joblib
            tree = KDTree(voxel_all, leaf_size=7)
            
            # 2. Ręczna alokacja macierzy WYJŚCIOWEJ (points_probs) w SHM
            num_points = points.shape[0]
            num_classes = voxel_probs_all.shape[1]
            points_probs_shape = (num_points, num_classes)
            points_probs_dtype = np.float32
            
            size_points_probs = np.dtype(points_probs_dtype).itemsize * num_points * num_classes
            
            shm_points_probs = shared_memory.SharedMemory(create=True, size=size_points_probs)
            shared_points_probs_view = np.ndarray(points_probs_shape, dtype=points_probs_dtype, buffer=shm_points_probs.buf)
            shared_points_probs_view[:] = 0 # start with 0s
            
            # input array metadata
            shm_info_out = {
                'name': shm_points_probs.name,
                'shape': points_probs_shape,
                'dtype': points_probs_dtype
            }
            
            # 3. work division 
            num_workers = os.cpu_count()
            chunk_size = num_points // num_workers
            chunk_data_list: List[Tuple[int, int]] = []
            
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_workers - 1 else num_points
                if start_idx < end_idx:
                    chunk_data_list.append((start_idx, end_idx))
        

            task_iterator = (
                delayed(self._worker_task)(
                    chunk,
                    voxel_probs_all,
                    points,
                    shm_info_out, 
                    tree,
                    k_neighbors_upsampling,
                    distance_sigma
                )
                for chunk in chunk_data_list
            )

            Parallel(n_jobs=num_workers, prefer='processes')(task_iterator)
            
            points_labels = np.argmax(shared_points_probs_view, axis=1)
            return points_labels.flatten()

        finally:
            if shm_points_probs is not None:
                try:
                    shm_points_probs.close()
                    import time
                    time.sleep(0.1)  # Give workers time to finish cleanup
                    shm_points_probs.unlink()
                except FileNotFoundError:
                    pass
    
    def _model_predict(self, voxel: torch.Tensor) -> torch.Tensor:

        voxel = torch.from_numpy(voxel).float().to(self.device)
        voxel = voxel.unsqueeze(dim = 0)
        with torch.no_grad():
            voxel_probs = self._model(voxel)
        voxel_probs = voxel_probs.permute(0, 2, 1).squeeze(dim = 0).cpu().numpy()

        return voxel_probs

    def _segment_voxel_base(self,
                             points: np.ndarray,
                             intensity: np.ndarray):


        voxel_all = np.full((points.shape[0], 3), np.nan, dtype = np.float32)
        voxel_probs_all = np.full((points.shape[0], self._model_config['num_classes']), np.nan, dtype=np.float32)

        checksum = 0
        generator = pcd_manipulation.voxelGridFragmentation(points,
                                                            voxel_size = np.array([self.voxel_size_small, self.voxel_size_small]),
                                                            num_points = self.model_config['num_points'],
                                                            overlap_ratio=0.4)
        if self.pbar_bool:
            pbar0 = tqdm(generator, desc="Points classification", unit=" voxel", leave=False)
        else:
            pbar0 = generator

        for (voxel_idx, noise) in pbar0:

            if voxel_idx.shape[0] == 0:
                continue

            voxel = points[voxel_idx]
            voxel0 = voxel.copy()

            voxel -= voxel.mean(axis= 0)

            intensity_voxel = intensity[voxel_idx]

            global_idx, voxel_idx = np.unique(voxel_idx, return_index=True) # global - unique z points, voxel - unique z voxel
            global_idx = np.sort(global_idx)

            voxel_idx = np.sort(voxel_idx)

            checksum += voxel_idx.shape[0]
            if self.pbar_bool:
                pbar0.update(1)
                pbar0.set_postfix({"Number of processed points": checksum})

            voxel = np.concatenate([voxel, intensity_voxel.reshape(-1, 1)], axis = 1)

            if not noise:

                voxel_probs = self._model_predict(voxel)

                assert voxel_probs.shape[0] == voxel.shape[0]

            else:
                voxel_probs = np.zeros((voxel.shape[0], self._model_config['num_classes']), dtype = np.float32)
                voxel_probs[:, 0] = 1 # highest prob for class 0


            voxel = voxel0[voxel_idx] # remove redundant points and overwrite centered voxel
            voxel_probs = voxel_probs[voxel_idx]

            voxel_all[global_idx] = voxel

            voxel_probs_all[global_idx] = voxel_probs
            voxel_all[global_idx] = voxel

        mask0 = np.isnan(voxel_all).any(axis = 1)
        mask1 = np.isnan(voxel_probs_all).any(axis=1)

        voxel_all = voxel_all[~mask0]
        voxel_probs_all = voxel_probs_all[~mask1]

        del voxel_probs, voxel, voxel0, voxel_idx

        return voxel_all, voxel_probs_all
    

    def _segment_small_voxel(self, points: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    
        voxel_all, voxel_probs_all = self._segment_voxel_base(points, intensity)
        labels = self._upsample_labeled_chunk_parallel(voxel_all, voxel_probs_all, points)

        return labels
    


    def _segment_big_voxel(self, points: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        labels = np.zeros(intensity.shape, dtype=np.int32)

        for indices in pcd_manipulation.voxelGridFragmentation(data=points,
                                                               num_points=0,
                                                               voxel_size=np.array([self.voxel_size_big, self.voxel_size_big]),
                                                               overlap_ratio=0,
                                                               shuffle=False):
            if indices.shape[0] == 0:
                continue

            points_chunk = points[indices]
            points_chunk -= points_chunk.mean(axis = 0)

            intensity_chunk = intensity[indices]

            labels_chunk = self._segment_small_voxel(points_chunk, intensity_chunk)
            labels[indices] = labels_chunk

        return labels

    def segment_pcd(self, points: np.ndarray, intensity: np.ndarray, fragment_pcd_threshold: int = 20*10e6) -> np.ndarray:

        intensity = self._scaler.fit_transform(intensity.reshape(1, -1))
        intensity = intensity.flatten()

        points -= points.mean(axis = 0)

        num_points = points.shape[0]
        if num_points < fragment_pcd_threshold:
            labels = self._segment_small_voxel(points, intensity)
        else:
            labels = self._segment_big_voxel(points, intensity)

        return labels

        
def test_segm():
    path2laz = "/home/michal-siniarski/Dokumenty/Z32/LAS/TREE_SEGM/automatically_segmented_data/O1W.laz"

    import laspy
    import pathlib as pth

    path2laz = pth.Path(path2laz)
    las = laspy.read(path2laz)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    intensity = np.asarray(las.intensity)

    segmenter = SegmentClass(model_name="RandLANetV3_2",
                             device = torch.device('cuda'),
                             pbar_bool = True)
    labels = segmenter.segment_pcd(points=points,
                          intensity=intensity)




if __name__ == '__main__':
    test_segm()


