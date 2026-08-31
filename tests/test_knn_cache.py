import pathlib
import sys

import torch


project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils import knn_torch


def test_build_preallocates_full_cache_without_concatenating_chunks(monkeypatch):
    points = torch.rand(1, 1025, 3)
    expected = torch.cdist(points, points)
    real_cdist = torch.cdist
    processed_chunk_sizes = []

    def tracked_cdist(first, second):
        processed_chunk_sizes.append(first.shape[1])
        return real_cdist(first, second)

    def reject_cat(*args, **kwargs):
        raise AssertionError('KNNCache.build must not concatenate distance chunks')

    monkeypatch.setattr(knn_torch.torch, 'cdist', tracked_cdist)
    monkeypatch.setattr(knn_torch.torch, 'cat', reject_cat)

    cache = knn_torch.KNNCache()
    cache.build(points)

    assert processed_chunk_sizes == [1024, 1]
    assert cache.distances.shape == (1, 1025, 1025)
    torch.testing.assert_close(cache.distances, expected)
