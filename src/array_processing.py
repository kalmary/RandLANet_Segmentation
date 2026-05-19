from typing import Union, Optional, Any, Tuple, Dict, List
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import pathlib as pth
import os
import sys
import gc
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn

try:
    from .final_files.RandLANet_CB import RandLANet
    from .utils import load_json, load_model, pcd_manipulation
except ImportError:
    try:
        from final_files.RandLANet_CB import RandLANet
        from utils import load_json, load_model, pcd_manipulation
    except ImportError:
        from PCDSegmentation.src.final_files.RandLANet_CB import RandLANet
        from PCDSegmentation.src.utils import load_json, load_model, pcd_manipulation

class SegmentClass:
    def __init__(self,
                 voxel_size_big: float = 200.,
                 overlap: float = 0.4,
                 scaled: bool = False,
                 model_name: str = None,
                 config_dir: Union[str, pth.Path] = "final_files",
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False):
        
        self.voxel_size_small = None
        self.voxel_size_big = voxel_size_big
        self.overlap = overlap
        self.scaled = scaled
        if model_name is None:
            raise ValueError("model_name cannot be None")
        self.model_name = model_name + '.pt'

        if isinstance(device, str):
            self.device = torch.device(device)
        self.device = device

        self.verbose = verbose

        self._scaler = None
        if scaled is True:
            self._scaler = self._init_scaler(feature_range=(0, 1))
        
        self.base_path = pth.Path(__file__).parent
        config_dir = self.base_path.joinpath(config_dir)
        self._config = None
        self._model_config = None
        self._load_config(config_dir)
        self._model = self._load_segmModel(config_dir)


    # TODO adjust model loading 
    def _load_config(self, config_dir: Optional[Union[pth.Path, str]] = None) -> dict:

        config_path = pth.Path(config_dir).joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        self._config = config_dict
        self._model_config: dict = config_dict['model_config']
        self.voxel_size_small: float = self._model_config['max_voxel_dim']

        return config_dict

    def _load_segmModel(self, model_dir: Union[pth.Path, str] = "./final_files") -> nn.Module:

        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = RandLANet(self._config["model_config"], self._config['num_classes'])
        self._model: nn.Module = load_model(file_path=path2model,
                                            model=model,
                                            device=self.device)
        self._model.to(self.device)
        self._model.eval()
    

        return self._model
    
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
    def _worker_task(chunk_data, info_probs, info_points, info_voxel, info_out,
                    k_neighbors, distance_sigma):
        shm_probs = shared_memory.SharedMemory(name=info_probs['name'])
        shm_points = shared_memory.SharedMemory(name=info_points['name'])
        shm_voxel = shared_memory.SharedMemory(name=info_voxel['name'])
        shm_out = shared_memory.SharedMemory(name=info_out['name'])
        try:
            voxel_probs_all = np.ndarray(info_probs['shape'], dtype=info_probs['dtype'], buffer=shm_probs.buf)
            points = np.ndarray(info_points['shape'], dtype=info_points['dtype'], buffer=shm_points.buf)
            voxel_all = np.ndarray(info_voxel['shape'], dtype=info_voxel['dtype'], buffer=shm_voxel.buf)
            out = np.ndarray(info_out['shape'], dtype=info_out['dtype'], buffer=shm_out.buf)

            tree = KDTree(voxel_all, leaf_size=7)

            start_idx, end_idx = chunk_data
            points_chunk = points[start_idx:end_idx]
            dists, indices = tree.query(points_chunk, k=k_neighbors)

            for local_i in range(points_chunk.shape[0]):
                global_i = start_idx + local_i
                neighbor_probs = voxel_probs_all[indices[local_i]]
                neighbor_dists = dists[local_i]
                weights = np.exp(-(neighbor_dists ** 2) / (2 * distance_sigma ** 2))
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights_sum
                    out[global_i] = np.sum(neighbor_probs * weights[:, np.newaxis], axis=0)
                else:
                    out[global_i] = 0.0
        finally:
            for shm in (shm_probs, shm_points, shm_voxel, shm_out):
                shm.close()


    def _upsample_labeled_chunk_parallel(self, voxel_all, voxel_probs_all, points,
                                        k_neighbors_upsampling=14, distance_sigma=0.35,
                                        num_workers=-1, verbose=False):
        if voxel_all.shape[0] == 0:
            return np.zeros(points.shape[0], dtype=np.int32)

        n_workers = os.cpu_count() if num_workers <= 0 else min(num_workers, os.cpu_count())
        k_neighbors_upsampling = min(k_neighbors_upsampling, voxel_all.shape[0])

        def make_shm(arr):
            arr = np.ascontiguousarray(arr, dtype=arr.dtype)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            view[:] = arr
            return shm, {'name': shm.name, 'shape': arr.shape, 'dtype': arr.dtype}

        num_points = points.shape[0]
        num_classes = voxel_probs_all.shape[1]

        shm_points, info_points = make_shm(points.astype(np.float32))
        shm_probs, info_probs = make_shm(voxel_probs_all.astype(np.float32))
        shm_voxel, info_voxel = make_shm(voxel_all.astype(np.float32))
        shm_out, info_out = make_shm(np.zeros((num_points, num_classes), dtype=np.float32))

        chunk_size = max(1, num_points // n_workers)
        chunks = [
            (i * chunk_size, min((i + 1) * chunk_size, num_points))
            for i in range(n_workers)
            if i * chunk_size < num_points
        ]

        try:
            Parallel(n_jobs=n_workers, backend='multiprocessing')(
                delayed(self._worker_task)(
                    chunk, info_probs, info_points, info_voxel, info_out,
                    k_neighbors_upsampling, distance_sigma
                )
                for chunk in chunks
            )
            out_view = np.ndarray((num_points, num_classes), dtype=np.float32, buffer=shm_out.buf)
            result = np.argmax(out_view, axis=1).flatten()
            return result
        finally:
            for shm in (shm_points, shm_probs, shm_voxel, shm_out):
                try:
                    shm.close()
                    shm.unlink()
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


        voxel_all = np.full((points.shape[0], 3), 0.0, dtype = np.float32)
        voxel_probs_all = np.full((points.shape[0], self._model_config['num_classes']), 0.0, dtype=np.float32)

        checksum = 0
        generator = pcd_manipulation.voxelGridFragmentation(points,
                                                            voxel_size = np.array([self.voxel_size_small, self.voxel_size_small]),
                                                            num_points = self._config['num_points'],
                                                            overlap_ratio=0.4)
        if self.verbose:
            pbar0 = tqdm(generator, desc="Points classification", unit=" voxel", leave=False, position=2)
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
            if self.verbose:
                pbar0.update(1)
                pbar0.set_postfix({"Number of processed points": checksum})

            voxel = np.concatenate([voxel, intensity_voxel.reshape(-1, 1)], axis = 1)

            if not noise:

                voxel_probs = self._model_predict(voxel)

                assert voxel_probs.shape[0] == voxel.shape[0]

            else:
                voxel_probs = np.full((voxel.shape[0], self._model_config['num_classes']), 0.0, dtype = np.float32)
                # voxel_probs[:, 0] = 0.9 # highest prob for class 0


            voxel = voxel0[voxel_idx] # remove redundant points and overwrite centered voxel
            voxel_probs = voxel_probs[voxel_idx]

            voxel_all[global_idx] = voxel
            voxel_probs_all[global_idx] = voxel_probs
        
        try:
            pbar0.close()
        except Exception:
            pass

        # mask0 = np.isnan(voxel_all).any(axis = 1)
        # mask1 = np.isnan(voxel_probs_all).any(axis=1)

        # voxel_all = voxel_all[~mask0]
        # voxel_probs_all = voxel_probs_all[~mask1]

        del voxel_probs, voxel, voxel0, voxel_idx

        return voxel_all, voxel_probs_all
    

    def _segment_small_voxel(self, points: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    
        voxel_all, voxel_probs_all = self._segment_voxel_base(points, intensity)
        labels = self._upsample_labeled_chunk_parallel(voxel_all, voxel_probs_all, points)
        del voxel_all, voxel_probs_all
        return labels
    


    def _segment_big_voxel(self, points: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        labels = np.zeros(intensity.shape, dtype=np.int32)

        for indices, _ in pcd_manipulation.voxelGridFragmentation(data=points,
                                                               num_points=0,
                                                               voxel_size=np.array([self.voxel_size_big, self.voxel_size_big]),
                                                               overlap_ratio=0,
                                                               shuffle=False,
                                                               verbose=self.verbose):
            if indices.shape[0] == 0:
                continue

            points_chunk = points[indices]
            points_chunk -= points_chunk.mean(axis = 0)

            intensity_chunk = intensity[indices]

            labels_chunk = self._segment_small_voxel(points_chunk, intensity_chunk)
            labels[indices] = labels_chunk
            
            del points_chunk, intensity_chunk, labels_chunk
            gc.collect()

        return labels

    def segment_pcd(self, points: np.ndarray, intensity: np.ndarray, fragment_pcd_threshold: int = 7e6) -> np.ndarray:

        if self.scaled: # TODO enable it if necessary

            intensity = self._scaler.fit_transform(intensity.reshape(1, -1))
        intensity = intensity.flatten()

        points = points - points.mean(axis = 0)
        points = points.astype(np.float32)

        num_points = points.shape[0]
        if num_points < fragment_pcd_threshold:
            labels = self._segment_small_voxel(points, intensity)
        else:
            labels = self._segment_big_voxel(points, intensity)

        return labels

        
def test_segm():
    path2laz = "/mnt/SSD_EXT4_1TB/DATA/GRAJEWO/Grajewo_michal_mod.laz"

    import laspy
    import pathlib as pth

    path2laz = pth.Path(path2laz)
    las = laspy.read(path2laz)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    intensity = np.asarray(las.intensity)

    segmenter = SegmentClass(
        voxel_size_big=200.,
        overlap=0.4,
        scaled=True,
        model_name="RandLANetV8_2",
        config_dir="final_files",
        device = torch.device('cuda'),
        verbose = True)
    
    labels = segmenter.segment_pcd(points=points,
                          intensity=intensity)
    
    from utils import plot_cloud
    plot_cloud(points, labels)




if __name__ == '__main__':
    test_segm()