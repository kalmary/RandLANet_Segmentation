import torch
import torch.nn as nn
from typing import Tuple, Optional


class KNNCache:
    def __init__(self):

        self.distances = None
        self.pcd = None

    
    def build(self, pcd): # batched knn for whole point cloud, gpu and cdist
        """Build the full distance cache while allocating one chunk at a time."""
        self.pcd = pcd.contiguous()
        batch_size, num_points, _ = self.pcd.size()
        chunk_size = min(1024, num_points)
        self.distances = torch.empty(
            (batch_size, num_points, num_points),
            device=self.pcd.device,
            dtype=self.pcd.dtype,
        )

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk_distances = torch.cdist(
                self.pcd[:, start:end],
                self.pcd,
            )
            self.distances[:, start:end].copy_(chunk_distances)
            del chunk_distances
    
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
