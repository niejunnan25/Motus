from pathlib import Path
import inspect
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vgm_bridge_stage1 import VGMBridgeStage1, VGMBridgeStage1Config
from train.eval_vgm_bridge_stage1 import pixel_metrics, role_mask_metrics


def _tail_conditioning_helper(tail_condition_frames: int) -> VGMBridgeStage1:
    model = object.__new__(VGMBridgeStage1)
    torch.nn.Module.__init__(model)
    model.config = VGMBridgeStage1Config(
        num_video_frames=6,
        tail_condition_frames=tail_condition_frames,
    )
    model.condition_on_last_frame = tail_condition_frames > 0
    return model


def test_tail4_condition_video_supports_real_tail_and_repeat_goal() -> None:
    model = _tail_conditioning_helper(4)
    first = torch.full((1, 3, 2, 2), 0.25)
    tail = torch.stack(
        [torch.full((1, 3, 2, 2), value) for value in (0.4, 0.5, 0.6, 0.7)],
        dim=1,
    )

    condition = model._make_condition_video(first_frame=first, tail_frames=tail)
    assert condition.shape == (1, 3, 7, 2, 2)
    torch.testing.assert_close(condition[:, :, 0], first * 2.0 - 1.0)
    torch.testing.assert_close(
        condition[:, :, -4:],
        (tail * 2.0 - 1.0).permute(0, 2, 1, 3, 4),
    )
    assert torch.count_nonzero(condition[:, :, 1:3]) == 0
    torch.testing.assert_close(
        model._frame_condition_mask(1, torch.device("cpu"), torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]),
    )

    goal = torch.full((1, 3, 2, 2), 0.9)
    repeated = model._make_condition_video(first_frame=first, last_frame=goal)
    expected_goal = (goal * 2.0 - 1.0).unsqueeze(2).expand(-1, -1, 4, -1, -1)
    torch.testing.assert_close(repeated[:, :, -4:], expected_goal)

    with pytest.raises(ValueError, match="Expected tail_frames with 4 frames"):
        model._make_condition_video(first_frame=first, tail_frames=tail[:, :3])

    signature = inspect.signature(VGMBridgeStage1.sample_bridge)
    assert "tail_frames" in signature.parameters
    assert "tail_role_masks" in signature.parameters


def test_tail4_pixel_metrics_exclude_all_conditioned_tail_frames() -> None:
    gt = torch.zeros(1, 6, 3, 1, 1)
    pred = gt.clone()
    pred[:, -4:] = 1.0

    metrics = pixel_metrics(gt, pred, tail_condition_frames=4)

    assert metrics["generated_mse"] == pytest.approx(0.0)
    assert metrics["full_mse"] > 0.0


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
