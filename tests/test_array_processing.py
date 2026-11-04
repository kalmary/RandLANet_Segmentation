import numpy as np
import pytest

from src import array_processing


@pytest.mark.parametrize("window_size", [12.0, 20.0])
def test_inference_windows_bound_xyz_with_quarter_overlap(window_size, monkeypatch):
    segmenter = array_processing.SegmentClass.__new__(array_processing.SegmentClass)
    segmenter.voxel_size_small = window_size
    segmenter.verbose = False
    segmenter._config = {"num_points": 0}
    segmenter._model_config = {"num_classes": 2}

    coordinates = np.array([0.0, 0.75, 1.0, 1.5, 2.0]) * window_size
    points = np.stack(np.meshgrid(coordinates, coordinates, coordinates), axis=-1)
    points = points.reshape(-1, 3).astype(np.float32)
    windows = []
    fragment = array_processing.pcd_manipulation.voxelGridFragmentation

    def capture_windows(data, **kwargs):
        for indices, noise in fragment(data, **kwargs):
            windows.append(data[indices].copy())
            yield indices, noise

    def predict(voxel):
        assert np.all(np.ptp(voxel[:, :3], axis=0) <= window_size)
        return np.tile(np.array([0.25, 0.75]), (len(voxel), 1))

    monkeypatch.setattr(
        array_processing.pcd_manipulation, "voxelGridFragmentation", capture_windows
    )
    monkeypatch.setattr(segmenter, "_model_predict", predict)

    reconstructed, probabilities = segmenter._segment_voxel_base(
        points, np.ones(len(points), dtype=np.float32)
    )

    assert len(windows) == 27
    for axis in range(3):
        starts = np.unique([window[:, axis].min() for window in windows])
        np.testing.assert_allclose(starts, [0, 0.75 * window_size, 1.5 * window_size])
    overlap_point = np.full(3, 0.75 * window_size)
    assert sum(np.any(np.all(window == overlap_point, axis=1)) for window in windows) == 8
    np.testing.assert_array_equal(reconstructed, points)
    np.testing.assert_allclose(probabilities, np.tile([0.25, 0.75], (len(points), 1)))
