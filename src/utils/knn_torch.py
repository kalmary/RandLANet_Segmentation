import torch
import torch.nn as nn
from typing import Tuple, Optional


def knn_me(src, tgt, k):
    B, N_tgt, D = tgt.size()

    # Process in chunks to avoid large distance matrices
    chunk_size = min(1024, N_tgt)  # Adjust based on available memory

    all_idx = []
    all_dist = []

    for i in range(0, N_tgt, chunk_size):
        end_idx = min(i + chunk_size, N_tgt)
        tgt_chunk = tgt[:, i:end_idx]

        # Only compute distances for this chunk
        distances = torch.cdist(tgt_chunk, src)
        # dist, idx = torch.topk(distances, k=k, dim=-1, largest=False)

        # all_idx.append(idx)
        all_dist.append(distances)

    return torch.cat(all_dist, dim=1)


class KNNCache:
    def __init__(self):
            
        self.distances = None
        self.pcd = None

    
    def build(self, pcd): # batched knn for whole point cloud, gpu and cdist
        """Memory-efficient KNN using chunked processing"""
        self.pcd = pcd.contiguous()
        _, N_tgt, _ = self.pcd.size()

        # Process in chunks to avoid large distance matrices
        chunk_size = min(1024, N_tgt)  # Adjust based on available memory

        all_dist = []

        for i in range(0, N_tgt, chunk_size):
            end_idx = min(i + chunk_size, N_tgt)
            tgt_chunk = self.pcd[:, i:end_idx]

            # Only compute distances for this chunk
            distances = torch.cdist(tgt_chunk, self.pcd)
            # dist, idx = torch.topk(distances, k=k, dim=-1, largest=False)

            # all_idx.append(idx)
            all_dist.append(distances)

        self.distances = torch.cat(all_dist, dim=1)
    
    def query(self, src_idx: torch.Tensor, tgt_idx: torch.Tensor, num_neighbors) -> Tuple[torch.Tensor, torch.Tensor]:
        
        assert self.distances is not None or self.pcd is not None, "KNN not initialized"
            
        distances = self.distances[src_idx, tgt_idx] # todo may be improved further by making distances array smaller and smaller

        dist, idx = torch.topk(distances, k=num_neighbors, dim=-1, largest=False) #idx always for original pcd

        return dist, idx
    
    def get_coords(self, idx) -> torch.Tensor:
        return self.pcd[:, idx]
    
    def clean(self) -> None:
        self.distances = None
        self.pcd = None
