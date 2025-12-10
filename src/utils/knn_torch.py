import torch
import torch.nn as nn
from typing import Tuple, Optional


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

        N_tgt = len(tgt_idx)
        
        # For memory efficiency, process in chunks if needed
        chunk_size = min(1024, N_tgt)
        
        all_dist = []
        all_idx = []
        
        for i in range(0, N_tgt, chunk_size):
            end_idx = min(i + chunk_size, N_tgt)
            tgt_chunk = tgt_idx[i:end_idx]
            
            # Extract only the needed distances for this chunk
            # Shape: (B, chunk_size, N_src)
            chunk_distances = self.distances[:, tgt_chunk][:, :, src_idx]
            
            # Find k nearest neighbors
            dist, local_idx = torch.topk(chunk_distances, k=num_neighbors, dim=-1, largest=False)
            
            # Map local indices back to original point cloud indices
            idx = src_idx[local_idx]
            
            all_dist.append(dist)
            all_idx.append(idx)

            del dist, local_idx, idx, chunk_distances
        
        return torch.cat(all_idx, dim=1), torch.cat(all_dist, dim=1)
    
    def get_coords(self, idx) -> torch.Tensor:
        return self.pcd[:, idx]
    
    def clear(self) -> None:
        self.distances = None
        self.pcd = None
