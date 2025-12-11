import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union
from collections import Counter

import torch
from torch.utils.data import IterableDataset, get_worker_info

import fpsample

import sys
import os

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import rotate_points, tilt_points, transform_points

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
                 device: Optional[torch.device] = torch.device('cpu')):

        super(Dataset).__init__()

        self.path = pth.Path(base_dir)
        self.device = device if device is not None else torch.device('cpu')
        self.mode = mode

        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights
        
        # Class balancing parameters
        self.oversample_factor = oversample_factor
        self.augment_minority = augment_minority
        self.minority_threshold = minority_threshold
        
        # Will be computed on first iteration if class_balanced_sampling=True
        self.class_sample_weights = None
        self.minority_classes = None

        self.verbose = verbose

    def _compute_class_statistics(self):
        """
        Compute class distribution (streaming - doesn't load all data).
        Only called once if class_balanced_sampling is enabled.
        """
        class_counts = Counter()
        total_points = 0
        
        if self.verbose:
            print("Computing class statistics for balanced sampling...")
        
        with h5py.File(self.path, 'r') as h_file:
            keys = list(h_file.keys())
            
            for key in keys:
                data = h_file[key][:]
                labels = data[:, :, -1].flatten()  # Assuming last column is labels
                
                class_counts.update(labels.astype(int))
                total_points += len(labels)
        
        # Compute sampling weights per class
        num_classes = max(class_counts.keys()) + 1
        class_frequencies = np.zeros(num_classes)
        
        for cls, count in class_counts.items():
            class_frequencies[int(cls)] = count / total_points
        
        # Inverse frequency weights (capped for stability)
        weights = np.where(class_frequencies > 0, 
                          1.0 / (class_frequencies + 1e-6), 
                          0.0)
        weights = weights / weights.sum() * num_classes
        
        # Apply oversample factor
        weights = np.power(weights, self.oversample_factor)
        weights = weights / weights.sum()
        
        self.class_sample_weights = torch.from_numpy(weights).float()
        
        # Identify minority classes
        self.minority_classes = set(np.where(class_frequencies < self.minority_threshold)[0])
        
        if self.verbose:
            print(f"\nClass distribution:")
        for cls in range(num_classes):
            if cls in class_counts:
                freq = class_frequencies[cls]
                
                if self.verbose:
                    is_minority = "MINORITY" if cls in self.minority_classes else ""
                    print(f"  Class {cls}: {class_counts[cls]:8d} points ({freq*100:5.2f}%) "
                        f"weight={weights[cls]:.4f} {is_minority}")

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
        noise = noise.to(cloud.device)
        return cloud + noise
    
    def _augment_minority_class_cloud(self, cloud_tensor, labels_tensor):
        """
        Apply stronger augmentation to clouds with many minority class points.
        """
        if self.minority_classes is None:
            return cloud_tensor
        
        # Check if this cloud has minority classes
        unique_labels = set(labels_tensor.numpy())
        has_minority = bool(unique_labels & self.minority_classes)
        
        if not has_minority:
            return cloud_tensor
        
        # Apply stronger augmentation
        cloud_tensor = cloud_tensor.to(self.device)
        
        # Stronger rotation (up to 45 degrees instead of random)
        angle = (torch.rand(1) - 0.5) * np.pi / 2  # ±45 degrees
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_z = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=self.device, dtype=cloud_tensor.dtype)
        
        cloud_tensor[:, :3] = torch.matmul(cloud_tensor[:, :3], rotation_z.T)
        
        # Stronger noise
        cloud_tensor[:, :3] = self._add_gaussian_noise(cloud_tensor[:, :3], std=0.025)
        
        # Random scaling
        scale = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
        cloud_tensor[:, :3] *= scale
        
        return cloud_tensor.cpu()

    def _balance_point_cloud(self, cloud_tensor, labels_tensor):
        """
        Resample points within a point cloud to balance classes.
        This is done per-cloud, maintaining streaming property.
        """
        if self.class_sample_weights is None:
            return cloud_tensor, labels_tensor
        
        # Compute per-point sampling weights based on their class
        point_weights = self.class_sample_weights[labels_tensor.long()]
        
        # Normalize to valid probabilities
        point_weights = point_weights / point_weights.sum()
        
        # Sample with replacement according to weights
        try:
            indices = torch.multinomial(point_weights, 
                                       self.num_points, 
                                       replacement=True)
            
            cloud_tensor = cloud_tensor[indices]
            labels_tensor = labels_tensor[indices]
        except RuntimeError:
            # Fallback if multinomial fails (shouldn't happen but safety)
            pass
        
        return cloud_tensor, labels_tensor

    def _balance_point_cloud(self, cloud_tensor, labels_tensor):
        """
        Resample points within a point cloud to balance classes.
        This is done per-cloud, maintaining streaming property.
        """
        
        # Compute per-point sampling weights based on their class
        point_weights = self.class_sample_weights[labels_tensor.long()]
        
        # Normalize to valid probabilities
        point_weights = point_weights / point_weights.sum()
        
        # Sample with replacement according to weights
        try:
            indices = torch.multinomial(point_weights, 
                                       self.num_points, 
                                       replacement=True)
            
            cloud_tensor = cloud_tensor[indices]
            labels_tensor = labels_tensor[indices]
        except RuntimeError:
            # Fallback if multinomial fails (shouldn't happen but safety)
            pass
        
        return cloud_tensor, labels_tensor

    def _process_cloud(self):
        """Process clouds one at a time (streaming)."""
        
        # Compute class statistics once if needed
        if self.class_balanced_sampling and self.class_sample_weights is None:
            self._compute_class_statistics()
        
        for cloud in self._key_streamer():
            cloud_tensor = torch.from_numpy(cloud[:, :3]).float()
            cloud_tensor -= cloud_tensor.mean(dim=0)

            labels_tensor = torch.from_numpy(cloud[:, -1]).long()
            features_tensor = torch.from_numpy(cloud[:, 3]).reshape(-1, 1).float()

            # Downsample if needed
            if cloud_tensor.shape[0] > self.num_points:
                idx = fpsample.bucket_fps_kdline_sampling(
                    cloud_tensor.cpu().numpy(), 
                    self.num_points, 
                    h=7
                )
                cloud_tensor = cloud_tensor[idx]
                features_tensor = features_tensor[idx]
                labels_tensor = labels_tensor[idx]

            if self.weights is not None and self.shuffle:
                weights_tensor = self.weights[labels_tensor] # 
                weights_tensor = weights_tensor.cpu()
                cloud_tensor, labels_tensor = self._balance_point_cloud(cloud_tensor, labels_tensor)

            if self.shuffle:
                cloud_tensor = cloud_tensor.to(self.device)
                cloud_tensor = self._add_gaussian_noise(cloud_tensor, std=0.015)
                cloud_tensor = rotate_points(cloud_tensor, device=self.device)
                cloud_tensor = tilt_points(cloud_tensor,
                                          max_x_tilt_degrees=5,
                                          max_y_tilt_degrees=5)
                cloud_tensor = transform_points(cloud_tensor,
                                               min_scale=0.95,
                                               max_scale=1.05,
                                               device=self.device)
                cloud_tensor = cloud_tensor.cpu()
                
                # Extra augmentation for minority classes
                if self.augment_minority:
                    cloud_tensor = self._augment_minority_class_cloud(
                        cloud_tensor, 
                        labels_tensor
                    )

            # Re-center after augmentations
            cloud_tensor -= cloud_tensor.mean(dim=0)

            # Concatenate features
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

        # Yield remaining samples
        if len(batch_data) > 0:
            batch_data_tensor = torch.stack(batch_data)
            batch_labels_tensor = torch.stack(batch_labels)

            if self.shuffle:
                idx = torch.randperm(batch_labels_tensor.shape[0])
                batch_data_tensor = batch_data_tensor[idx]
                batch_labels_tensor = batch_labels_tensor[idx]

            yield batch_data_tensor, batch_labels_tensor