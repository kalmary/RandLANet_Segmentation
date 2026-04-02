import torch
import torch.nn as nn

import pathlib as pth
import sys
src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import KNNCache

import json
from pathlib import Path
from typing import Union


def input_norm(input: torch.Tensor) -> torch.Tensor:
    """
    input: (B, N, 4) — xyz relative to sphere center + intensity [0,1]
    xyz:       divide by per-sample max abs → [-1, 1]
    intensity: subtract 0.5             → [-0.5, 0.5]
    """
    input = input.clone()
    r = input[..., :3].abs().amax(dim=1, keepdim=True)  # (B, 1, 3)
    input[..., :3] = input[..., :3] / (r + 1e-6)
    input[..., 3] = input[..., 3] - 0.5
    return input


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 transpose=False, padding_mode='zeros', bn=False, activation_fn=None):
        super().__init__()
        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_fn(in_channels, out_channels, kernel_size,
                            stride=stride, padding=0, padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        idx, dist = knn_output
        B, N, K = idx.size()
        idx = idx.long()

        coords_t      = coords.transpose(-2, -1)                                         # (B, 3, N)
        batch_idx     = torch.arange(B, device=coords.device, dtype=torch.long).view(B,1,1,1)
        coord_idx     = torch.arange(3, device=coords.device, dtype=torch.long).view(1,3,1,1)
        neighbors     = coords_t[batch_idx, coord_idx, idx.unsqueeze(1)]                 # (B, 3, N, K)
        center_coords = coords_t.unsqueeze(-1)                                           # (B, 3, N, 1)
        relative_pos  = center_coords - neighbors

        concat = torch.cat([
            center_coords.expand(-1, -1, -1, K),
            relative_pos,
            neighbors,
            dist.unsqueeze(1)
        ], dim=1)

        encoded = self.mlp(concat)
        return torch.cat([encoded, features.expand(-1, -1, -1, K)], dim=1)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_mlp  = SharedMLP(in_channels, in_channels, bn=False, activation_fn=None)
        self.output_mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        scores   = torch.softmax(self.score_mlp(x), dim=-1)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        del scores
        return self.output_mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp1     = SharedMLP(d_in,   d_out//2, activation_fn=nn.ReLU())
        self.mlp2     = SharedMLP(d_out,  2*d_out)
        self.shortcut = SharedMLP(d_in,   2*d_out,  bn=True, activation_fn=nn.ReLU())
        self.lse1     = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2     = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.pool1    = AttentivePooling(d_out, d_out//2)
        self.pool2    = AttentivePooling(d_out, d_out)
        self.relu     = nn.ReLU()

    def forward(self, coords_idx, knn_query: KNNCache, features):
        knn_output = knn_query.query(coords_idx, coords_idx, self.num_neighbors)
        coords     = knn_query.get_coords(coords_idx)
        x = self.mlp1(features)
        x = self.pool1(self.lse1(coords, x, knn_output))
        x = self.pool2(self.lse2(coords, x, knn_output))
        return self.relu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, model_config: dict, n_classes: int):
        super().__init__()

        d_in                         = model_config['d_in']
        self.num_neighbors           = model_config['num_neighbors']
        self._num_neighbors_upsample = 3
        self.decimation              = model_config['decimation']
        self.KNN                     = KNNCache()

        encoder_layers  = model_config['encoder_layers']
        decoder_layers  = model_config['decoder_layers']
        fc_start_config = model_config.get('fc_start', {'d_out': 8})
        fc_end_config   = model_config.get('fc_end',   {'layers': [64, 32], 'dropout': 0.5})

        fc_start_d_out = fc_start_config.get('d_out', 8)
        self.fc_start  = SharedMLP(d_in, fc_start_d_out, bn=True, activation_fn=nn.ReLU())

        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(l['d_in'], l['d_out'], self.num_neighbors)
            for l in encoder_layers
        ])

        mlp_dim  = 2 * encoder_layers[-1]['d_out']
        self.mlp = SharedMLP(mlp_dim, mlp_dim, activation_fn=nn.ReLU())

        self.decoder = nn.ModuleList([
            SharedMLP(l['d_in'], l['d_out'], transpose=True, bn=True, activation_fn=nn.ReLU())
            for l in decoder_layers
        ])

        self.relu = nn.ReLU()

        # fc_end → n_classes
        fc_end_layers  = fc_end_config.get('layers',  [64, 32])
        fc_end_dropout = fc_end_config.get('dropout', 0.5)
        current_d      = decoder_layers[-1]['d_out']
        fc_end_modules = []

        for d_out in fc_end_layers:
            fc_end_modules.append(SharedMLP(current_d, d_out, bn=True, activation_fn=nn.ReLU()))
            current_d = d_out

        if fc_end_dropout > 0:
            fc_end_modules.append(nn.Dropout(fc_end_dropout))

        fc_end_modules.append(SharedMLP(current_d, n_classes))
        self.fc_end = nn.Sequential(*fc_end_modules)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], n_classes: int):
        with open(config_path) as f:
            config = json.load(f)
        return cls(model_config=config, n_classes=n_classes)

    def forward(self, input):
        """
        input:  (B, N, d_in)
        output: (B, N, n_classes)
        """
        d = self.decimation
        N = input.shape[1]

        input = input_norm(input)

        if self.training:
            permutation = torch.randperm(N, device=input.device)
            input = input[:, permutation]

        coords = input[..., :3]
        self.KNN.build(coords)

        x = self.fc_start(input.transpose(-2, -1).unsqueeze(-1))  # (B, d, N, 1)

        decimation_ratio = 1
        x_stack = []

        # encoder
        for lfa in self.encoder:
            current_indices = torch.arange(N // decimation_ratio, device=coords.device)
            x = lfa(current_indices, self.KNN, x)
            x_stack.append(x)
            decimation_ratio *= d
            x = x[:, :, :N // decimation_ratio]

        x = self.mlp(x)

        # decoder
        for mlp in self.decoder:
            down_indices = torch.arange(N // decimation_ratio,     device=coords.device)
            up_indices   = torch.arange(d * N // decimation_ratio, device=coords.device)

            neighbors, distances = self.KNN.query(down_indices, up_indices,
                                                   self._num_neighbors_upsample)
            _, C, _, _ = x.size()
            neighbors  = neighbors.long()

            distances.add_(1e-8)
            torch.reciprocal_(distances)
            weights = distances / distances.sum(dim=-1, keepdim=True)
            del distances

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, C, -1, -1)
            del neighbors

            x_expanded = x.squeeze(-1).unsqueeze(-1).expand(-1, -1, -1, self._num_neighbors_upsample)
            x_neighbors = torch.gather(x_expanded, 2, extended_neighbors)
            del extended_neighbors, x_expanded

            x_neighbors.mul_(weights.unsqueeze(1))
            del weights

            x = mlp(self.relu(torch.cat(
                (x_neighbors.sum(dim=-1, keepdim=True), x_stack.pop()), dim=1
            )))
            del x_neighbors

            decimation_ratio //= d

        del x_stack
        self.KNN.clear()

        out = self.fc_end(x).squeeze(-1)  # (B, n_classes, N)

        if self.training:
            out = out[:, :, torch.argsort(permutation)]

        return out


def test_model():
    model_config = {
        "d_in": 4,
        "num_neighbors": 32,
        "decimation": 2,
        "encoder_layers": [
            {"d_in": 8,    "d_out": 32},
            {"d_in": 64,   "d_out": 128},
            {"d_in": 256,  "d_out": 256},
            {"d_in": 512,  "d_out": 512}
        ],
        "decoder_layers": [
            {"d_in": 2048, "d_out": 512},
            {"d_in": 1024, "d_out": 256},
            {"d_in": 512,  "d_out": 128},
            {"d_in": 192,  "d_out": 32}
        ],
        "fc_start": {"d_out": 8},
        "fc_end":   {"layers": [32, 16], "dropout": 0.5}
    }

    B, N, n_classes = 4, 8192, 10
    dummy = torch.randn(B, N, model_config['d_in']).cuda()
    model = RandLANet(model_config, n_classes=n_classes).cuda()

    out = model(dummy)
    expected = (B, n_classes, N)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print(f"Success! Output: {out.shape}")


if __name__ == '__main__':
    test_model()