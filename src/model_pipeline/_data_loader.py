import argparse
import pathlib as pth
import numpy as np
import random
import h5py
from typing import Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info

import fpsample

class Dataset(IterableDataset):

    def __init__(self, base_dir: Union[str, pth.Path],
                 mode: int = 0,
                 num_points: int = 4096,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 device: Optional[torch.device] = torch.device('cpu')):

        super(Dataset).__init__()

        self.path = pth.Path(base_dir)
        self.device = device
        self.mode = mode

        self.num_points = num_points
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _key_streamer(self):
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

        if len(batch_data) > 0:
            batch_data_tensor = torch.stack(batch_data)
            batch_labels_tensor = torch.stack(batch_labels)

            if self.shuffle:
                idx = torch.randperm(batch_labels_tensor.shape[0])
                batch_data_tensor = batch_data_tensor[idx]
                batch_labels_tensor = batch_labels_tensor[idx]

            yield batch_data_tensor, batch_labels_tensor

def inspect_dataset(input_path, mode=0, num_points=8192, batch_size=1,
                    shuffle=False, device="cpu"):
    dataset = Dataset(
        input_path,
        mode=mode,
        num_points=num_points,
        batch_size=batch_size,
        shuffle=shuffle,
        device=torch.device(device),
    )
    for data, labels in dataset:
        data_avg = data.squeeze(0)
        print(data_avg.min(axis = 0).values, data_avg.max(axis = 0).values)


def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Inspect batches from an HDF5 point-cloud dataset.")
    parser.add_argument("--input-path", type=pth.Path, required=True)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)
    inspect_dataset(
        args.input_path,
        mode=args.mode,
        num_points=args.num_points,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        device=args.device,
    )


if __name__ == '__main__':
    main()
