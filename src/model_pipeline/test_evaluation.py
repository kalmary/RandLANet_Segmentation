from types import SimpleNamespace

import numpy as np
import pytest

from src.model_pipeline import EvalSegm_RandLANet as evaluation


def test_stratified_indices_are_exact_reproducible_and_keep_rare_classes():
    targets = np.array([0] * 8 + [1] * 3 + [2])

    first = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )
    second = evaluation._stratified_indices(
        targets, 6, np.random.default_rng(0)
    )

    assert len(first) == 6
    assert len(np.unique(first)) == 6
    np.testing.assert_array_equal(first, second)
    assert np.bincount(targets[first], minlength=3).tolist() == [3, 2, 1]


def test_stratified_indices_return_all_points_below_cap_and_validate_cap():
    indices = evaluation._stratified_indices(
        np.array([1, 0, 1]), 4, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(indices, [0, 1, 2])

    with pytest.raises(ValueError, match="positive"):
        evaluation._stratified_indices(
            np.array([0]), 0, np.random.default_rng(0)
        )


def test_metrics_report_accuracy_and_ordered_class_iou():
    metrics = evaluation.calculate_metrics(
        predictions=np.array([0, 0, 0, 1]),
        targets=np.array([0, 0, 1, 1]),
        num_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["miou"] == pytest.approx(7 / 12)
    np.testing.assert_allclose(metrics["class_iou"], [2 / 3, 1 / 2])


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([0], [0, 1], "same shape"),
        ([], [], "empty"),
        ([-1], [0], "predictions"),
        ([0], [2], "targets"),
    ],
)
def test_metrics_reject_invalid_arrays(predictions, targets, message):
    with pytest.raises(ValueError, match=message):
        evaluation.calculate_metrics(predictions, targets, num_classes=2)
