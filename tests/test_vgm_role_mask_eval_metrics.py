from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.eval_vgm_bridge_stage1 import role_mask_metrics


def test_binary_role_mask_metrics_include_common_foreground_metrics() -> None:
    gt = torch.tensor([[[[[1.0, 1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0, 1.0]]]]])
    pred = torch.tensor([[[[[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 1.0]]]]])

    metrics = role_mask_metrics(gt, pred, "binary")

    assert metrics["role_mask_foreground_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["role_mask_foreground_f1"] == pytest.approx(0.8)
    assert metrics["role_mask_iou"] == metrics["role_mask_foreground_iou"]


def test_color_role_mask_metrics_include_common_foreground_metrics() -> None:
    colors = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        4: [0, 0, 255],
    }
    gt = torch.tensor([colors[1], colors[2], colors[0], colors[4]], dtype=torch.float32)
    pred = torch.tensor([colors[1], colors[0], colors[0], colors[4]], dtype=torch.float32)
    gt = (gt / 255.0).T.reshape(1, 1, 3, 1, 4)
    pred = (pred / 255.0).T.reshape(1, 1, 3, 1, 4)

    metrics = role_mask_metrics(gt, pred, "role_color", role_mask_palette=colors)

    assert metrics["role_mask_foreground_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["role_mask_foreground_f1"] == pytest.approx(0.8)
    assert metrics["role_mask_macro_iou"] == pytest.approx(2.0 / 3.0)
