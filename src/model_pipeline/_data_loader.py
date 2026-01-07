import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union, OrderedDict

import torch
from torch.utils.data import IterableDataset, get_worker_info

import fpsample
import random

import sys
import os

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import rotate_points, tilt_points, transform_points, add_gaussian_noise


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Dataset(IterableDataset):

    def __init__(self, base_dir: Union[str, pth.Path],
                 mode: int = 0,
                 num_points: int = 4096,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 weights: Optional[torch.Tensor] = None,
                 oversample_power: float = 1.5,
                 device: Optional[torch.device] = torch.device('cpu')):

        super(Dataset).__init__()

        self.path = pth.Path(base_dir)
        self.device = device
        self.mode = mode

        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights

        # Process weights for sampling if provided
        if weights is not None and oversample_power != 1.0:
            # Apply power to make oversampling more/less aggressive
            sampling_weights = torch.pow(weights, oversample_power)
            # Re-normalize
            sampling_weights = sampling_weights / sampling_weights.sum()
            self.sampling_weights = sampling_weights
        else:
            self.sampling_weights = weights

    def _key_streamer(self):
        """
        Generator over all keys. Each worker processes all keys,
        but only its assigned chunks within each key.
        """
        with h5py.File(self.path, 'r') as h_file:
            keys = list(h_file.keys())
            
            if self.shuffle:
                random.shuffle(keys)

            worker_info = get_worker_info()
            if worker_info is None:
                iter_keys = keys
            else:
                total_workers = worker_info.num_workers
                worker_id = worker_info.id
                iter_keys = keys[worker_id::total_workers]

            for key in iter_keys:
                data = h_file[key][:]
                if self.shuffle:
                    indices = list(range(data.shape[0]))
                    random.shuffle(indices)
                    data = data[indices]

                yield from data



    def _balance_point_cloud(self, cloud_tensor, features_tensor, labels_tensor):
            """
            Resample points within a point cloud to balance classes using provided weights.
            
            Returns:
                Resampled cloud_tensor, features_tensor, labels_tensor (all same length)
            """
            if self.sampling_weights is None:
                return cloud_tensor, features_tensor, labels_tensor
            

            self.weights.to(labels_tensor.device)
            # Get per-point sampling weights based on their class
            point_weights = self.sampling_weights[labels_tensor.long()]
            
            # Normalize to valid probabilities
            point_weights = point_weights / (point_weights.sum() + 1e-10)
            
            # Sample with replacement according to weights
            try:
                num_samples = min(self.num_points, len(labels_tensor))
                indices = torch.multinomial(
                    point_weights, 
                    num_samples, 
                    replacement=True
                ).to(labels_tensor.device)
                
                # Apply same indices to all tensors
                cloud_tensor = cloud_tensor[indices]
                features_tensor = features_tensor[indices]
                labels_tensor = labels_tensor[indices]
                
            except RuntimeError as e:
                # Fallback if multinomial fails (rare edge case)
                print(f"Warning: Point cloud resampling failed: {e}")
                pass
            
            return cloud_tensor, features_tensor, labels_tensor

    def _process_cloud(self):


        for cloud in self._key_streamer():
            cloud_tensor = cloud[:, :3]

            cloud_tensor = torch.from_numpy(cloud_tensor[:, :3]).float()
            cloud_tensor -= cloud_tensor.mean(dim=0)


            labels_tensor = torch.from_numpy(cloud[:, -1]).long()


            features_tensor = torch.from_numpy(cloud[:, 3]).reshape(-1, 1).float()

            if cloud_tensor.shape[0] > self.num_points:
                idx = fpsample.bucket_fps_kdline_sampling(cloud_tensor.cpu().numpy(), self.num_points, h=7)
                cloud_tensor = cloud_tensor[idx]
                features_tensor = features_tensor[idx]
                labels_tensor = labels_tensor[idx]

            if self.weights is not None and self.shuffle:
                self.weights = self.weights[labels_tensor] # 
                cloud_tensor, features_tensor, labels_tensor = self._balance_point_cloud(cloud_tensor, features_tensor, labels_tensor)

            if self.shuffle:
                cloud_tensor = cloud_tensor.to(self.device)
                cloud_tensor = add_gaussian_noise(cloud_tensor, std=0.05)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor,
                                           max_x_tilt_degrees=5,
                                           max_y_tilt_degrees=5)
                cloud_tensor = transform_points(cloud_tensor,
                                                min_scale=0.95,
                                                max_scale=1.05,
                                                device=self.device)
                cloud_tensor = cloud_tensor.cpu()

            cloud_tensor -= cloud_tensor.mean(dim=0)



           
            if features_tensor is not None:
                cloud_tensor = torch.cat([cloud_tensor, features_tensor], dim=1)

            yield cloud_tensor, labels_tensor


    def __iter__(self):
        stream = self._process_cloud()
        batch_data = []
        batch_labels = []

        for cloud_tensor, labels_tensor in stream:

            batch_data.append(cloud_tensor)
            batch_labels.append(labels_tensor)

            if len(batch_data) == self.batch_size:

                batch_data_tensor = torch.stack(batch_data)
                batch_labels_tensor = torch.stack(batch_labels)

                if self.shuffle:
                    idx = torch.randperm(batch_labels_tensor.shape[0])

                    batch_data_tensor = batch_data_tensor[idx]
                    batch_labels_tensor = batch_labels_tensor[idx]

                yield batch_data_tensor, batch_labels_tensor

                batch_data = []
                batch_labels = []

        # Yield any remaining samples
        if len(batch_data) > 0:

            batch_data_tensor = torch.stack(batch_data)
            batch_labels_tensor = torch.stack(batch_labels)

            if self.shuffle:
                idx = torch.randperm(batch_labels_tensor.shape[0])

                batch_data_tensor = batch_data_tensor[idx]
                batch_labels_tensor = batch_labels_tensor[idx]

            yield batch_data_tensor, batch_labels_tensor
