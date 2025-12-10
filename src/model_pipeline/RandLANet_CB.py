import torch
import torch.nn as nn
from torchinfo import summary

import pathlib as pth
import sys
src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import KNNCache

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional


def knn_me(src, tgt, k):
    """Memory-efficient KNN using chunked processing"""
    B, N_src, D = src.size()
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
        dist, idx = torch.topk(distances, k=k, dim=-1, largest=False)

        all_idx.append(idx)
        all_dist.append(dist)

    return torch.cat(all_idx, dim=1), torch.cat(all_dist, dim=1)

def input_norm(input: torch.Tensor, max_voxel_dim = 20.):
    max_voxel_dim /= 2 # voxels should be in range (-10, 10), intensity (0, 10) -> both moved into (-1, 1) range

    coords = input[..., :3]
    coords.sub_(coords.mean(dim=1, keepdim=True)) # center
    coords.div_(max_voxel_dim) # scale

    input[..., -1].div_(5.).sub_(1.) # intensity scale + center
    
    input[..., :3] = coords
    return input




class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        idx, dist = knn_output
        B, N, K = idx.size()

        # Ensure idx is long type for indexing
        idx = idx.long()
        
        # More memory-efficient approach
        coords_t = coords.transpose(-2, -1)  # (B, 3, N) - done once

        # Efficient neighbor gathering using advanced indexing
        # IMPORTANT: All index tensors must be long type
        batch_indices = torch.arange(B, device=coords.device, dtype=torch.long).view(B, 1, 1, 1)
        coord_indices = torch.arange(3, device=coords.device, dtype=torch.long).view(1, 3, 1, 1)
        neighbors = coords_t[batch_indices, coord_indices, idx.unsqueeze(1)]  # (B, 3, N, K)

        # Create center coordinates efficiently
        center_coords = coords_t.unsqueeze(-1)  # (B, 3, N, 1)

        # Compute spatial encodings directly without intermediate expansions
        relative_pos = center_coords - neighbors  # Broadcasting handles expansion

        # Concatenate all spatial features at once
        concat = torch.cat([
            center_coords.expand(-1, -1, -1, K),  # center coordinates
            neighbors,  # neighbor coordinates
            relative_pos,  # relative positions
            dist.unsqueeze(1)  # distances
        ], dim=1)  # Shape: (B, 10, N, K)

        # Process through MLP
        encoded = self.mlp(concat)

        # Concatenate with expanded features
        return torch.cat([
            encoded,
            features.expand(-1, -1, -1, K)
        ], dim=1)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # Modified to match state dict structure
        self.score_mlp = SharedMLP(in_channels, in_channels, bn=False, activation_fn=None)
        self.output_mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_mlp(x)
        scores = torch.softmax(scores, dim=-1)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        del scores

        return self.output_mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU(inplace=True)


    def forward(self, coords_idx, knn_query: KNNCache, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        # knn_output = knn_me(tgt_idx, src_idx, self.num_neighbors)
        knn_output = knn_query.query(coords_idx, coords_idx, self.num_neighbors)
        coords = knn_query.get_coords(coords_idx)


        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)

        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, model_config: dict, num_classes: int):
        super(RandLANet, self).__init__()

        # Extract parameters from config
        d_in = model_config.get('d_in')
        self.num_neighbors = model_config.get('num_neighbors')
        self._num_neighbors_upsample = 3
        self.decimation = model_config.get('decimation')

        self.KNN = KNNCache()

        self.max_voxel_dim = model_config.get('max_voxel_dim')
        
        # Get encoder and decoder layer configurations
        encoder_layers = model_config.get('encoder_layers')
        decoder_layers = model_config.get('decoder_layers')
        
        # Get fc_start and fc_end configurations
        fc_start_config = model_config.get('fc_start', {'d_out': 8})
        fc_end_config = model_config.get('fc_end', {'layers': [64, 32], 'dropout': 0.5})

        # fc_start: configurable output dimension
        fc_start_d_out = fc_start_config.get('d_out', 8)
        self.fc_start = nn.Linear(d_in, fc_start_d_out)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(fc_start_d_out, eps=1e-6, momentum=0.99),
            nn.Tanh()
        )

        # encoding layers - build from config
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(layer['d_in'], layer['d_out'], self.num_neighbors)
            for layer in encoder_layers
        ])

        # MLP dimension is 2 * last encoder output
        mlp_dim = 2 * encoder_layers[-1]['d_out']
        self.mlp = SharedMLP(mlp_dim, mlp_dim, activation_fn=nn.ReLU())

        # decoding layers - build from config
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(layer['d_in'], layer['d_out'], **decoder_kwargs)
            for layer in decoder_layers
        ])

        # fc_end: configurable layers and dropout
        fc_end_layers = fc_end_config.get('layers', [64, 32])
        fc_end_dropout = fc_end_config.get('dropout', 0.5)
        
        final_d_in = decoder_layers[-1]['d_out']
        
        # Build fc_end dynamically
        fc_end_modules = []
        current_d = final_d_in
        
        for d_out in fc_end_layers:
            fc_end_modules.append(SharedMLP(current_d, d_out, bn=True, activation_fn=nn.ReLU()))
            current_d = d_out
        
        if fc_end_dropout > 0:
            fc_end_modules.append(nn.Dropout(fc_end_dropout))
        
        fc_end_modules.append(SharedMLP(current_d, num_classes))
        
        self.fc_end = nn.Sequential(*fc_end_modules)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], num_classes: int):
        """
        Load model from a JSON config file
        
        Parameters
        ----------
        config_path : str or Path
            Path to JSON config file
        num_classes : int
            Number of output classes
            
        Returns
        -------
        RandLANet
            Model instance
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(model_config=config, num_classes=num_classes)

    def forward(self, input):
        d = self.decimation
        N = input.shape[1]



        input = input_norm(input, max_voxel_dim=self.max_voxel_dim)

        if self.training:
            permutation = torch.randperm(N, device=input.device) 
            input = input[:,permutation]
        

        coords = input[..., :3]
        self.KNN.build(coords)

        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        for i, lfa in enumerate(self.encoder):

            # at iteration i, x.shape = (B, N//(d**i), d_in)
            current_indices = torch.arange(N//decimation_ratio, device=coords.device)
            x = lfa(current_indices, self.KNN, x)
            x_stack.append(x)
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]



        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for i, mlp in enumerate(self.decoder):
            down_indices = torch.arange(N//decimation_ratio, device=coords.device)
            up_indices = torch.arange(d*N//decimation_ratio, device=coords.device)
            neighbors, distances = self.KNN.query(
                down_indices,
                up_indices,
                self._num_neighbors_upsample
            )

            _, C, _, _ = x.size()
            # N_up = neighbors.size(1)
            
            # Ensure neighbors is long type for gather
            neighbors = neighbors.long()
            
            # Inverse distance weighting for interpolation
            distances.add_(1e-8)
            torch.reciprocal_(distances)
            weights = distances / distances.sum(dim=-1, keepdim=True)  # (B, N_up, 3)
            del distances

            # Reshape for gathering: (B, C, N_down, 1) -> gather -> (B, C, N_up, K)
            # We need to gather from the spatial dimension (dim=2)
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N_up, 3)
            del neighbors
            
            # Gather neighbors features
            x_neighbors = torch.gather(x.expand(-1, -1, -1, self._num_neighbors_upsample), 2, extended_neighbors)  # (B, C, N_up, 3)
            del extended_neighbors
            
            # Apply weights and sum: (B, C, N_up, 3) * (B, 1, N_up, 3) -> sum -> (B, C, N_up, 1)
            x_neighbors.mul_(weights.unsqueeze(1))
            del weights

            x = torch.cat((x_neighbors.sum(dim=-1, keepdim=True), x_stack.pop()), dim=1)
            del x_neighbors

            x = mlp(x)

            decimation_ratio //= d

        del x_stack
        # >>>>>>>>>> DECODER
        
        self.KNN.clear()

        scores = self.fc_end(x)
        if self.training:
            scores = scores[:, : ,torch.argsort(permutation)]

        return scores.squeeze(-1)
    


def test_model():
    model_config = {
        'd_in': 4,
        'num_neighbors': 16,
        'decimation': 2,
        'encoder_layers': [
            {'d_in': 8, 'd_out': 32},
            {'d_in': 64, 'd_out': 128},
            {'d_in': 256, 'd_out': 256},
            {'d_in': 512, 'd_out': 512},
            {'d_in': 1024, 'd_out': 1024}
        ],
        'decoder_layers': [
            {'d_in': 4096, 'd_out': 1024},
            {'d_in': 2048, 'd_out': 512},
            {'d_in': 1024, 'd_out': 256},
            {'d_in': 512, 'd_out': 128},
            {'d_in': 192, 'd_out': 8}
        ],
        'fc_start': {
            'd_out': 8
        },
        'fc_end': {
            'layers': [64, 32],
            'dropout': 0.5
        },
        'max_voxel_dim': 20.
    }
    
    batch_size = 10
    num_points = 8192
    num_classes = 10

    # random test input
    dummy_input = torch.randn(batch_size, num_points, model_config['d_in']).to(torch.device('cuda'))

    model = RandLANet(model_config=model_config, num_classes=num_classes).to(torch.device('cuda'))
    # summary(model, input_size=dummy_input.shape)

    # test output
    output = model(dummy_input)

    # shape check
    expected_output_shape = (batch_size, num_classes, num_points)
    assert output.shape == expected_output_shape, f"Expected shape: {expected_output_shape}, received: {output.shape}"

    print(f"Success! Output shape: {output.shape}, expected: {expected_output_shape}")

if __name__ == '__main__':
    test_model()