import pathlib
import sys

import pytest
import torch
import torch.nn as nn


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.model_pipeline.RandLANet_CB import (
    LocalFeatureAggregation,
    LocalSpatialEncoding,
    RandLANet,
    input_norm,
)
from src.model_pipeline.RandLANet_CB_pooled import RandLANet as PooledRandLANet


class ZeroSpatialEncoding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.channels, x.shape[2], x.shape[3]))


def test_local_spatial_encoding_gathers_neighbor_features_from_knn_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    features = torch.tensor(
        [[[[10.0], [20.0], [30.0]], [[1.0], [2.0], [3.0]]]]
    )
    neighbor_idx = torch.tensor([[[1, 2], [2, 0], [0, 1]]])
    distances = torch.zeros(1, 3, 2)

    encoding = LocalSpatialEncoding(d=2, num_neighbors=2)
    monkeypatch.setattr(encoding, "mlp", ZeroSpatialEncoding(channels=2))
    output = encoding(coords, features, (neighbor_idx, distances))

    expected_neighbors = torch.tensor(
        [[[[20.0, 30.0], [30.0, 10.0], [10.0, 20.0]],
          [[2.0, 3.0], [3.0, 1.0], [1.0, 2.0]]]]
    )
    assert output.shape == (1, 4, 3, 2)
    torch.testing.assert_close(output[:, 2:], expected_neighbors)


def test_input_norm_preserves_metric_xyz_and_centers_intensity():
    inputs = torch.tensor(
        [
            [[3.0, 0.0, 0.0, 1.0], [0.0, 4.0, 0.0, 0.0]],
            [[1.0, 2.0, 2.0, 0.75], [0.0, 0.0, 0.0, 0.25]],
        ]
    )

    normalized = input_norm(inputs)

    torch.testing.assert_close(normalized[..., :3], inputs[..., :3])
    torch.testing.assert_close(
        normalized[..., 3],
        torch.tensor([[0.5, -0.5], [0.25, -0.25]]),
    )


def test_residual_block_matches_randlanet_normalization_and_activation():
    block = LocalFeatureAggregation(d_in=8, d_out=16, num_neighbors=4)

    assert block.mlp1.batch_norm is not None
    assert block.mlp2.batch_norm is not None
    assert block.shortcut.batch_norm is not None
    assert block.shortcut.activation_fn is None
    assert isinstance(block.activation, nn.LeakyReLU)
    assert block.activation.negative_slope == pytest.approx(0.2)


def test_pooled_module_uses_canonical_model():
    assert PooledRandLANet is RandLANet


def test_neighborhood_max_pooling_and_model_output_shapes():
    features = torch.tensor(
        [[[[1.0], [5.0], [3.0], [2.0]], [[4.0], [0.0], [7.0], [6.0]]]]
    )
    neighbor_idx = torch.tensor([[[0, 2], [1, 3]]])

    pooled = RandLANet.random_sample(features, neighbor_idx)

    assert pooled.shape == (1, 2, 2, 1)
    torch.testing.assert_close(
        pooled,
        torch.tensor([[[[3.0], [5.0]], [[7.0], [6.0]]]]),
    )

    model_config = {
        "d_in": 4,
        "num_neighbors": 2,
        "decimation": 2,
        "encoder_layers": [
            {"d_in": 4, "d_out": 4},
            {"d_in": 8, "d_out": 8},
        ],
        "decoder_layers": [
            {"d_in": 32, "d_out": 8},
            {"d_in": 16, "d_out": 4},
        ],
        "fc_start": {"d_out": 4},
        "fc_end": {"layers": [], "dropout": 0.0},
    }
    model = RandLANet(model_config, n_classes=3).eval()
    model_input = torch.rand(2, 16, 4)

    with torch.no_grad():
        output = model(model_input)

    assert output.shape == (2, 3, 16)
