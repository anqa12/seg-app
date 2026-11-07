import pytest
import numpy as np
from utils.display import _compute_metrics, class_percentages


# perfect match with pred mask
def test_compute_metrics_perfect_match():
    pred = np.ones((4, 4), dtype=np.uint8)
    gt = np.ones((4, 4), dtype=np.uint8)

    metrics = _compute_metrics(pred, gt)

    # IoU & F1 needs to be exactly 1
    assert metrics["iou"][1] == 1.0
    assert metrics["f1"][1] == 1.0

    # for other classes it needs to be nan (to not influence on the results because the class do not occur in GT)
    assert np.isnan(metrics["iou"][0])
    assert np.isnan(metrics["f1"][2])


# partial match with pred mask
def test_compute_metrics_partial_match():
    # gt includes 1st and 2nd class
    gt = np.array([[1, 1],
                   [0, 0]], dtype=np.uint8)

    pred = np.array([[1, 0],
                     [0, 0]], dtype=np.uint8)

    metrics = _compute_metrics(pred, gt)

    # class 1:
    # TP = 1, FP = 0, FN = 1
    # IoU = TP / (TP + FP + FN) = 1 / (1 + 0 + 1) = 0.5
    assert np.isclose(metrics["iou"][1], 0.5)

    # class 0:
    # TP = 2, FP = 1, FN = 0
    # IoU = 2 / (2 + 1 + 0) = 2/3
    assert np.isclose(metrics["iou"][0], 2 / 3)


# pred and GT mask do not match at all
def test_compute_metrics_mismatch():
    gt = np.ones((2, 2), dtype=np.uint8) * 2  # 2nd class
    pred = np.ones((2, 2), dtype=np.uint8) * 3  # 3rd class

    metrics = _compute_metrics(pred, gt)

    assert metrics["iou"][2] == 0.0
    assert metrics["iou"][3] == 0.0


# correct way of calculating pixels on the image
def test_class_percentages():
    mask = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.uint8)

    pct = class_percentages(mask)

    assert pct["Tło"] == 50.0
    assert pct["Zabudowa"] == 30.0
    assert pct["Lasy"] == 20.0
    assert pct["Woda"] == 0.0