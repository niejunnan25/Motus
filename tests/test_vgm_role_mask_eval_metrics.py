from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.eval_vgm_bridge_stage1 import role_mask_metrics


def test_binary_role_mask_metrics_include_common_foreground_metrics() -> None:
    gt_frame = torch.tensor([[[1.0, 1.0, 0.0, 1.0]]]).repeat(3, 1, 1)
    pred_middle = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]]).repeat(3, 1, 1)
    gt = gt_frame.reshape(1, 1, 3, 1, 4).repeat(1, 3, 1, 1, 1)
    pred = torch.stack([gt_frame, pred_middle, gt_frame]).unsqueeze(0)

    metrics = role_mask_metrics(gt, pred, "binary", tail_condition_frames=1)

    assert metrics["role_mask_foreground_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["role_mask_foreground_f1"] == pytest.approx(0.8)
    assert metrics["role_mask_iou"] == metrics["role_mask_foreground_iou"]
    assert metrics["role_mask_full_foreground_iou"] == pytest.approx(8.0 / 9.0)


def test_color_role_mask_metrics_include_common_foreground_metrics() -> None:
    colors = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        4: [0, 0, 255],
    }
    gt_frame = torch.tensor([colors[1], colors[2], colors[0], colors[4]], dtype=torch.float32)
    pred_middle = torch.tensor([colors[1], colors[0], colors[0], colors[4]], dtype=torch.float32)
    gt_frame = (gt_frame / 255.0).T.reshape(3, 1, 4)
    pred_middle = (pred_middle / 255.0).T.reshape(3, 1, 4)
    gt = gt_frame.reshape(1, 1, 3, 1, 4).repeat(1, 3, 1, 1, 1)
    pred = torch.stack([gt_frame, pred_middle, gt_frame]).unsqueeze(0)

    metrics = role_mask_metrics(
        gt,
        pred,
        "role_color",
        role_mask_palette=colors,
        tail_condition_frames=1,
    )

    assert metrics["role_mask_foreground_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["role_mask_foreground_f1"] == pytest.approx(0.8)
    assert metrics["role_mask_macro_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["role_mask_full_foreground_iou"] == pytest.approx(8.0 / 9.0)
