import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union, OrderedDict

import torch
from torch.utils.data import IterableDataset, get_worker_info

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.pcd_manipulation import rotate_points, tilt_points, transform_points


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import fpsample
import random

class Dataset(IterableDataset):

    def __init__(self, base_dir: Union[str, pth.Path],
                 mode: int = 0,
                 num_points: int = 4096,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 weights: torch.Tensor = Optional[torch.Tensor],
                 device: torch.device = Optional[torch.device('cpu')]):

        super(Dataset).__init__()

        self.path = pth.Path(base_dir)
        self.device = device
        self.mode = mode

        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights

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

    def _add_gaussian_noise(self, cloud, std=0.01):
        noise = torch.randn_like(cloud) * std
        return cloud + noise

    def _process_cloud(self):


        retry_count = 0

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



            if self.shuffle:
                cloud_tensor = self._add_gaussian_noise(cloud_tensor, std=0.015)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor,
                                           max_x_tilt_degrees=5,
                                           max_y_tilt_degrees=5)
                cloud_tensor = transform_points(cloud_tensor,
                                                min_scale=0.95,
                                                max_scale=1.05,
                                                device=self.device)

            cloud_tensor -= cloud_tensor.mean(dim=0)

            cloud_tensor = cloud_tensor.cpu()

           
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
