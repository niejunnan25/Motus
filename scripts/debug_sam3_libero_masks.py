#!/usr/bin/env python3
"""Generate SAM3 mask definition debug sheets for fixed LIBERO bridge windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


DEFAULT_PROMPTS: Dict[int, List[str]] = {
    0: ["white mug", "yellow and white mug", "left plate", "right plate", "robot arm"],
    1: ["white mug", "yellow and white mug", "left plate", "right plate", "robot arm"],
    2: ["white mug", "chocolate pudding", "pudding cup", "plate", "robot arm"],
    3: ["white mug", "yellow and white mug", "left plate", "right plate", "robot arm"],
    4: ["plate", "stove", "robot arm"],
    5: ["middle drawer", "drawer handle", "cabinet", "robot arm"],
    6: ["bowl", "cabinet", "robot arm"],
    7: ["wine bottle", "green bottle", "bottle", "cabinet", "robot arm"],
    8: ["chocolate pudding", "pudding cup", "food cup", "basket", "robot arm"],
    9: ["alphabet soup", "soup can", "can", "basket", "robot arm"],
    10: ["orange juice", "basket", "robot arm"],
    11: ["orange juice", "basket", "robot arm"],
    12: ["ketchup", "ketchup bottle", "red bottle", "basket", "robot arm"],
    13: ["black bowl", "top drawer", "wooden cabinet", "plate", "robot arm"],
    14: ["black bowl", "top drawer", "wooden cabinet", "plate", "robot arm"],
    15: ["black bowl", "cookie box", "plate", "robot arm"],
}

PROMPT_COLORS: Sequence[Tuple[int, int, int]] = (
    (255, 75, 75),
    (45, 180, 255),
    (80, 220, 120),
    (255, 190, 55),
    (190, 120, 255),
    (255, 95, 200),
)

ROLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "robot": (180, 105, 255),
    "object": (255, 80, 80),
    "target": (45, 180, 255),
    "context": (120, 150, 160),
    "final": (220, 35, 35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows_json",
        default=(
            "/mnt/workspace1/users/niejunnan/codebase/Motus/eval_outputs/"
            "vgm_bridge_v2_50k_step50004_s50_16samples/v2_a_base50k/"
            "vgm_bridge_v2_a_roi_lerobot_video_448x224_lang_17f_2gpu_gbs8_50k/"
            "step_50004/windows.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/workspace1/users/niejunnan/codebase/Motus/eval_outputs/v3_sam3_mask_debug",
    )
    parser.add_argument("--sample_ids", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Override prompts for all samples. Can be passed multiple times.",
    )
    parser.add_argument("--video_height", type=int, default=448)
    parser.add_argument("--video_width", type=int, default=224)
    parser.add_argument("--view_layout", choices=["single", "vertical", "horizontal"], default="vertical")
    parser.add_argument("--confidence_threshold", type=float, default=0.30)
    parser.add_argument("--motion_percentile", type=float, default=90.0)
    parser.add_argument("--motion_min_diff", type=float, default=18.0)
    parser.add_argument("--motion_dilate", type=int, default=5)
    parser.add_argument("--interaction_dilate", type=int, default=31)
    parser.add_argument("--selection_motion_dilate", type=int, default=17)
    parser.add_argument("--target_local_dilate", type=int, default=29)
    parser.add_argument("--final_motion_dilate", type=int, default=21)
    parser.add_argument("--view_edge_margin", type=int, default=2)
    parser.add_argument(
        "--manual_overrides",
        default=None,
        help=(
            "Optional JSON with manual box seeds. Coordinates are on the composed "
            "debug frame. Supports xyxy_abs, xyxy_norm, or per-frame boxes."
        ),
    )
    parser.add_argument("--manual_box_pad_ratio", type=float, default=0.18)
    parser.add_argument("--manual_box_pad_pixels", type=int, default=4)
    parser.add_argument("--sam_resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Use FP32 weights. bfloat16 means BF16 autocast, matching SAM3's inference path.",
    )
    parser.add_argument("--max_prompts", type=int, default=5)
    return parser.parse_args()


def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    src_h, src_w = frame.shape[:2]
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source frame shape: {frame.shape}")
    scale = min(target_h / src_h, target_w / src_w)
    new_h = max(1, int(src_h * scale))
    new_w = max(1, int(src_w * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def read_video_frames(video_path: str, frame_indices: Sequence[int]) -> List[np.ndarray]:
    wanted = set(int(idx) for idx in frame_indices)
    max_index = max(wanted)
    decoded: Dict[int, np.ndarray] = {}

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                decoded[frame_idx] = frame.to_ndarray(format="rgb24")
            if frame_idx >= max_index:
                break
    finally:
        container.close()

    missing = [idx for idx in frame_indices if idx not in decoded]
    if missing:
        raise RuntimeError(f"Failed to decode frames {missing} from {video_path}")
    return [decoded[int(idx)] for idx in frame_indices]


def compose_views(view_frames: Sequence[np.ndarray], target_size: Tuple[int, int], view_layout: str) -> np.ndarray:
    if not view_frames:
        raise ValueError("Expected at least one view frame")
    if len(view_frames) == 1 or view_layout == "single":
        frame = view_frames[0]
    else:
        if view_layout not in ("vertical", "horizontal"):
            raise ValueError(f"Unsupported view_layout={view_layout!r}")
        target_h = max(frame.shape[0] for frame in view_frames)
        target_w = max(frame.shape[1] for frame in view_frames)
        aligned = [
            frame
            if frame.shape[:2] == (target_h, target_w)
            else resize_with_padding(frame, (target_h, target_w))
            for frame in view_frames
        ]
        axis = 0 if view_layout == "vertical" else 1
        frame = np.concatenate(aligned, axis=axis)
    if frame.shape[:2] != target_size:
        frame = resize_with_padding(frame, target_size)
    return frame


def load_window_frames(window: Dict, target_size: Tuple[int, int], view_layout: str) -> List[np.ndarray]:
    frame_indices = window["frame_indices"]
    video_paths = window.get("video_path") or window.get("video_paths")
    if video_paths is None:
        raise KeyError("Window must contain video_path or video_paths")
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    if not video_paths:
        raise ValueError("Window video path list is empty")
    view_batches = [read_video_frames(path, frame_indices) for path in video_paths]
    frames = []
    for frame_pos in range(len(frame_indices)):
        frames.append(compose_views([batch[frame_pos] for batch in view_batches], target_size, view_layout))
    return frames


def build_model(args: argparse.Namespace):
    model = build_sam3_image_model(device=args.device)
    model.eval()
    processor = Sam3Processor(
        model,
        resolution=args.sam_resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    return processor


def convert_prompt_output(output: Dict, image: Image.Image) -> Dict:
    masks = output["masks"].detach().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]

    if masks.shape[0] == 0:
        combined = np.zeros((image.height, image.width), dtype=bool)
    else:
        combined = masks.astype(bool).any(axis=0)

    return {
        "combined": combined,
        "masks": masks.astype(bool),
        "boxes": boxes,
        "scores": scores,
    }


def empty_result(height: int, width: int, source: str = "") -> Dict:
    mask = np.zeros((height, width), dtype=bool)
    return {
        "combined": mask,
        "masks": mask[None],
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.array([], dtype=np.float32),
        "source": source,
    }


def xyxy_to_cxcywh_norm(box_xyxy: Sequence[float], image_w: int, image_h: int) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    x0 = min(max(x0, 0.0), float(image_w))
    x1 = min(max(x1, 0.0), float(image_w))
    y0 = min(max(y0, 0.0), float(image_h))
    y1 = min(max(y1, 0.0), float(image_h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [
        ((x0 + x1) * 0.5) / float(image_w),
        ((y0 + y1) * 0.5) / float(image_h),
        max(1.0, x1 - x0) / float(image_w),
        max(1.0, y1 - y0) / float(image_h),
    ]


def expand_xyxy_box(
    box_xyxy: Sequence[float],
    image_w: int,
    image_h: int,
    pad_ratio: float,
    pad_pixels: int,
) -> List[int]:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad_x = max(float(pad_pixels), (x1 - x0) * float(pad_ratio))
    pad_y = max(float(pad_pixels), (y1 - y0) * float(pad_ratio))
    return [
        int(max(0, np.floor(x0 - pad_x))),
        int(max(0, np.floor(y0 - pad_y))),
        int(min(image_w, np.ceil(x1 + pad_x))),
        int(min(image_h, np.ceil(y1 + pad_y))),
    ]


def box_mask_from_xyxy(box_xyxy: Sequence[float], image_w: int, image_h: int) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy]
    x0 = max(0, min(image_w, x0))
    x1 = max(0, min(image_w, x1))
    y0 = max(0, min(image_h, y0))
    y1 = max(0, min(image_h, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    mask = np.zeros((image_h, image_w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def constrain_result_to_manual_box(
    result: Dict,
    box_xyxy_abs: Sequence[float],
    image_w: int,
    image_h: int,
    pad_ratio: float,
    pad_pixels: int,
) -> Dict:
    roi_xyxy = expand_xyxy_box(box_xyxy_abs, image_w, image_h, pad_ratio, pad_pixels)
    roi_mask = box_mask_from_xyxy(roi_xyxy, image_w, image_h)
    masks = result["masks"]
    scores = result["scores"]

    kept_masks = []
    kept_scores = []
    roi_area = max(1, int(roi_mask.sum()))
    for idx, mask in enumerate(masks):
        clipped = np.logical_and(mask, roi_mask)
        clipped_area = int(clipped.sum())
        if clipped_area <= 0:
            continue
        if clipped_area / float(roi_area) < 0.02:
            continue
        kept_masks.append(clipped)
        if idx < scores.shape[0]:
            kept_scores.append(float(scores[idx]))

    if kept_masks:
        masks_arr = np.stack(kept_masks).astype(bool)
        scores_arr = np.asarray(kept_scores, dtype=np.float32) if kept_scores else np.array([], dtype=np.float32)
        combined = masks_arr.any(axis=0)
    else:
        combined = box_mask_from_xyxy(box_xyxy_abs, image_w, image_h)
        masks_arr = combined[None]
        scores_arr = np.asarray([1.0], dtype=np.float32)

    manual_box = np.asarray([box_xyxy_abs], dtype=np.float32)
    result = dict(result)
    result.update(
        {
            "combined": combined,
            "masks": masks_arr,
            "boxes": manual_box,
            "scores": scores_arr,
            "manual_roi_xyxy_abs": roi_xyxy,
            "manual_constrained": True,
        }
    )
    return result


def cxcywh_norm_to_xyxy_abs(box_cxcywh: Sequence[float], image_w: int, image_h: int) -> List[float]:
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    x0 = (cx - w * 0.5) * image_w
    x1 = (cx + w * 0.5) * image_w
    y0 = (cy - h * 0.5) * image_h
    y1 = (cy + h * 0.5) * image_h
    return [x0, y0, x1, y1]


def manual_box_entry_to_xyxy_abs(entry: Dict, image_w: int, image_h: int) -> List[float]:
    if "xyxy_abs" in entry:
        return [float(v) for v in entry["xyxy_abs"]]
    if "xyxy_norm" in entry:
        x0, y0, x1, y1 = [float(v) for v in entry["xyxy_norm"]]
        return [x0 * image_w, y0 * image_h, x1 * image_w, y1 * image_h]
    if "cxcywh_norm" in entry:
        return cxcywh_norm_to_xyxy_abs(entry["cxcywh_norm"], image_w, image_h)
    raise KeyError(f"Manual override entry must contain xyxy_abs, xyxy_norm, or cxcywh_norm: {entry}")


def interpolate_box(
    frame_idx: int,
    keyed_boxes: Dict[int, Sequence[float]],
) -> List[float] | None:
    if not keyed_boxes:
        return None
    if frame_idx in keyed_boxes:
        return [float(v) for v in keyed_boxes[frame_idx]]
    keys = sorted(keyed_boxes)
    if frame_idx < keys[0] or frame_idx > keys[-1]:
        return None
    prev_keys = [key for key in keys if key < frame_idx]
    next_keys = [key for key in keys if key > frame_idx]
    if not prev_keys or not next_keys:
        return None
    left = prev_keys[-1]
    right = next_keys[0]
    alpha = (frame_idx - left) / float(right - left)
    left_box = np.asarray(keyed_boxes[left], dtype=np.float32)
    right_box = np.asarray(keyed_boxes[right], dtype=np.float32)
    return ((1.0 - alpha) * left_box + alpha * right_box).tolist()


def load_manual_overrides(path: str | None) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("samples", data)


def manual_entries_for_sample_frame(
    manual_overrides: Dict,
    sample_id: int,
    frame_idx: int,
    image_w: int,
    image_h: int,
) -> List[Dict]:
    sample_cfg = manual_overrides.get(str(sample_id), manual_overrides.get(sample_id, {}))
    entries = sample_cfg.get("objects", sample_cfg.get("entries", [])) if isinstance(sample_cfg, dict) else []
    out = []
    for entry_idx, entry in enumerate(entries):
        base = {
            "name": entry.get("name", f"manual_{entry_idx}"),
            "role": entry.get("role", "object"),
            "source": entry.get("source", "manual"),
        }
        box_xyxy_abs = None
        if "frames" in entry:
            keyed_boxes = {
                int(key): manual_box_entry_to_xyxy_abs(value, image_w, image_h)
                for key, value in entry["frames"].items()
            }
            if entry.get("interpolate", True):
                box_xyxy_abs = interpolate_box(frame_idx, keyed_boxes)
            else:
                box_xyxy_abs = keyed_boxes.get(frame_idx)
        elif "frame" in entry:
            if int(entry["frame"]) == frame_idx:
                box_xyxy_abs = manual_box_entry_to_xyxy_abs(entry, image_w, image_h)
        else:
            box_xyxy_abs = manual_box_entry_to_xyxy_abs(entry, image_w, image_h)

        if box_xyxy_abs is None:
            continue
        item = dict(base)
        item["xyxy_abs"] = box_xyxy_abs
        item["cxcywh_norm"] = xyxy_to_cxcywh_norm(box_xyxy_abs, image_w, image_h)
        out.append(item)
    return out


def result_from_manual_entry(
    processor: Sam3Processor,
    state: Dict,
    image: Image.Image,
    entry: Dict,
    pad_ratio: float,
    pad_pixels: int,
) -> Dict:
    processor.reset_all_prompts(state)
    output = processor.add_geometric_prompt(box=entry["cxcywh_norm"], label=True, state=state)
    result = convert_prompt_output(output, image)
    result = constrain_result_to_manual_box(
        result,
        entry["xyxy_abs"],
        image.width,
        image.height,
        pad_ratio,
        pad_pixels,
    )
    result["source"] = entry.get("source", "manual")
    result["manual_name"] = entry.get("name", "manual")
    result["role"] = entry.get("role", "object")
    result["manual_box_xyxy_abs"] = entry["xyxy_abs"]
    result["manual_box_cxcywh_norm"] = entry["cxcywh_norm"]
    return result


def run_prompts_on_image(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: Sequence[str],
    dtype: str,
    device: str,
) -> Dict[str, Dict]:
    autocast_enabled = dtype == "bfloat16"
    device_type = "cuda" if device.startswith("cuda") else device
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        state = processor.set_image(image)
        outputs = {}
        for prompt in prompts:
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            outputs[prompt] = convert_prompt_output(output, image)
    return outputs


def run_manual_entries_on_image(
    processor: Sam3Processor,
    image: Image.Image,
    entries: Sequence[Dict],
    dtype: str,
    device: str,
    manual_box_pad_ratio: float,
    manual_box_pad_pixels: int,
) -> Dict[str, Dict]:
    if not entries:
        return {}
    autocast_enabled = dtype == "bfloat16"
    device_type = "cuda" if device.startswith("cuda") else device
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        state = processor.set_image(image)
        outputs = {}
        for entry_idx, entry in enumerate(entries):
            key = f"manual:{entry.get('role', 'object')}:{entry.get('name', entry_idx)}"
            outputs[key] = result_from_manual_entry(
                processor,
                state,
                image,
                entry,
                manual_box_pad_ratio,
                manual_box_pad_pixels,
            )
    return outputs


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.46) -> np.ndarray:
    out = image.copy()
    if mask.any():
        color_arr = np.array(color, dtype=np.float32)
        out_f = out.astype(np.float32)
        out_f[mask] = (1.0 - alpha) * out_f[mask] + alpha * color_arr
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or not mask.any():
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def motion_results_from_frames(
    frames: Sequence[np.ndarray],
    percentile: float,
    min_diff: float,
    dilate: int,
) -> List[Dict]:
    """Build a coarse per-frame motion mask against the first frame for QA only."""
    first = frames[0].astype(np.float32)
    results = []
    for frame in frames:
        diff = np.abs(frame.astype(np.float32) - first).mean(axis=2)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        threshold = max(float(min_diff), float(np.percentile(diff, percentile)))
        mask = diff > threshold
        mask = dilate_mask(mask, dilate)
        results.append(
            {
                "combined": mask,
                "masks": mask[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
                "threshold": threshold,
                "mean_diff": float(diff.mean()),
                "max_diff": float(diff.max()),
            }
        )
    return results


def union_results(results_by_prompt: Dict[str, List[Dict]]) -> List[Dict]:
    if not results_by_prompt:
        return []
    first_prompt = next(iter(results_by_prompt))
    frame_count = len(results_by_prompt[first_prompt])
    results = []
    for frame_idx in range(frame_count):
        masks = [results_by_prompt[prompt][frame_idx]["combined"] for prompt in results_by_prompt]
        combined = np.logical_or.reduce(masks) if masks else np.zeros_like(results_by_prompt[first_prompt][frame_idx]["combined"])
        results.append(
            {
                "combined": combined,
                "masks": combined[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
            }
        )
    return results


def merge_result_lists(*result_lists: Sequence[Dict]) -> List[Dict]:
    result_lists = [results for results in result_lists if results]
    if not result_lists:
        return []
    frame_count = len(result_lists[0])
    merged = []
    for frame_idx in range(frame_count):
        masks = [results[frame_idx]["combined"] for results in result_lists]
        combined = np.logical_or.reduce(masks)
        merged.append(
            {
                "combined": combined,
                "masks": combined[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
            }
        )
    return merged


def gated_motion_results(
    motion_results: Sequence[Dict],
    anchor_results: Sequence[Dict],
    dilate: int,
) -> List[Dict]:
    if not motion_results:
        return []
    if not anchor_results:
        return list(motion_results)
    results = []
    for motion_result, anchor_result in zip(motion_results, anchor_results):
        anchor = dilate_mask(anchor_result["combined"], dilate)
        combined = np.logical_and(motion_result["combined"], anchor)
        results.append(
            {
                "combined": combined,
                "masks": combined[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
                "anchor_area": int(anchor.sum()),
            }
        )
    return results


def prompt_role(prompt: str, task_text: str) -> str:
    """Assign a prompt to a QA role used for operation-focused mask design."""
    prompt_l = prompt.lower()
    task_l = task_text.lower()
    if prompt_l.startswith("manual:"):
        parts = prompt_l.split(":", 2)
        if len(parts) >= 2 and parts[1] in {"robot", "object", "target", "context"}:
            return parts[1]
    if "robot" in prompt_l or "gripper" in prompt_l:
        return "robot"
    if "handle" in prompt_l:
        return "target"
    if "plate" in prompt_l:
        return "object" if "push the plate" in task_l else "target"
    if "basket" in prompt_l:
        return "target"
    if "cookie box" in prompt_l:
        return "context"
    if "cabinet" in prompt_l or "stove" in prompt_l or "drawer" in prompt_l:
        return "context"
    return "object"


def role_area_limits(prompt: str, role: str, task_text: str) -> Tuple[float, float]:
    """Return max total and max per-view area ratios for selected instances."""
    prompt_l = prompt.lower()
    task_l = task_text.lower()
    if role == "robot":
        return 0.12, 0.20
    if role == "object":
        if "plate" in prompt_l and "push the plate" in task_l:
            return 0.24, 0.42
        if "bowl" in prompt_l:
            return 0.16, 0.30
        return 0.10, 0.22
    if role == "target":
        if "plate" in prompt_l:
            return 0.14, 0.26
        if "basket" in prompt_l:
            return 0.10, 0.22
        if "handle" in prompt_l:
            return 0.04, 0.08
        return 0.10, 0.20
    return 0.08, 0.14


def view_slices(mask_shape: Tuple[int, int], view_layout: str) -> List[Tuple[slice, slice]]:
    h, w = mask_shape
    if view_layout == "vertical":
        mid = h // 2
        return [(slice(0, mid), slice(0, w)), (slice(mid, h), slice(0, w))]
    if view_layout == "horizontal":
        mid = w // 2
        return [(slice(0, h), slice(0, mid)), (slice(0, h), slice(mid, w))]
    return [(slice(0, h), slice(0, w))]


def trim_view_edges(mask: np.ndarray, view_layout: str, margin: int) -> np.ndarray:
    if margin <= 0 or not mask.any():
        return mask.astype(bool)
    trimmed = mask.astype(bool).copy()
    for ys, xs in view_slices(mask.shape, view_layout):
        view = trimmed[ys, xs]
        if view.shape[0] <= margin * 2 or view.shape[1] <= margin * 2:
            continue
        view[:margin, :] = False
        view[-margin:, :] = False
        view[:, :margin] = False
        view[:, -margin:] = False
    return trimmed


def max_view_area_ratio(mask: np.ndarray, view_layout: str) -> float:
    max_ratio = 0.0
    for ys, xs in view_slices(mask.shape, view_layout):
        view = mask[ys, xs]
        view_area = float(view.size)
        if view_area > 0:
            max_ratio = max(max_ratio, float(view.sum()) / view_area)
    return max_ratio


def filter_role_frame_result(
    prompt: str,
    role: str,
    task_text: str,
    frame_result: Dict,
    image_area: int,
    view_layout: str,
    edge_margin: int,
) -> Dict:
    masks = frame_result["masks"]
    boxes = frame_result["boxes"]
    scores = frame_result["scores"]
    if masks.shape[0] == 0:
        empty = np.zeros_like(frame_result["combined"])
        return {
            "combined": empty,
            "masks": empty[None],
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "source_prompt": prompt,
            "role": role,
        }

    max_total_ratio, max_view_ratio = role_area_limits(prompt, role, task_text)
    kept_masks = []
    kept_boxes = []
    kept_scores = []
    for idx, mask in enumerate(masks):
        candidate = trim_view_edges(mask, view_layout, edge_margin)
        area = int(candidate.sum())
        if area <= 0:
            continue
        total_ratio = float(area) / float(image_area)
        view_ratio = max_view_area_ratio(candidate, view_layout)
        if total_ratio > max_total_ratio or view_ratio > max_view_ratio:
            continue
        kept_masks.append(candidate)
        if boxes.shape[0]:
            kept_boxes.append(boxes[idx] if idx < boxes.shape[0] else boxes[0])
        if scores.shape[0]:
            kept_scores.append(float(scores[idx] if idx < scores.shape[0] else scores[-1]))

    if not kept_masks:
        empty = np.zeros_like(frame_result["combined"])
        return {
            "combined": empty,
            "masks": empty[None],
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "source_prompt": prompt,
            "role": role,
        }

    masks_arr = np.stack(kept_masks).astype(bool)
    boxes_arr = np.asarray(kept_boxes, dtype=np.float32) if kept_boxes else np.zeros((0, 4), dtype=np.float32)
    scores_arr = np.asarray(kept_scores, dtype=np.float32) if kept_scores else np.array([], dtype=np.float32)
    return {
        "combined": masks_arr.any(axis=0),
        "masks": masks_arr,
        "boxes": boxes_arr,
        "scores": scores_arr,
        "source_prompt": prompt,
        "role": role,
    }


def role_prompt_results(
    selected_prompt_results: Dict[str, List[Dict]],
    task_text: str,
    image_area: int,
    view_layout: str,
    edge_margin: int,
) -> Dict[str, Dict[str, List[Dict]]]:
    roles: Dict[str, Dict[str, List[Dict]]] = {"robot": {}, "object": {}, "target": {}, "context": {}}
    for prompt, results in selected_prompt_results.items():
        role = prompt_role(prompt, task_text)
        roles.setdefault(role, {})[prompt] = [
            filter_role_frame_result(
                prompt,
                role,
                task_text,
                result,
                image_area,
                view_layout,
                edge_margin,
            )
            for result in results
        ]
    return roles


def union_role_results(role_results: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, List[Dict]]:
    return {
        role: union_results(prompt_results) if prompt_results else []
        for role, prompt_results in role_results.items()
    }


def intersect_result_lists(result_a: Sequence[Dict], result_b: Sequence[Dict]) -> List[Dict]:
    if not result_a or not result_b:
        return []
    results = []
    for frame_a, frame_b in zip(result_a, result_b):
        combined = np.logical_and(frame_a["combined"], frame_b["combined"])
        results.append(
            {
                "combined": combined,
                "masks": combined[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
            }
        )
    return results


def local_target_results(
    target_results: Sequence[Dict],
    actor_results: Sequence[Dict],
    motion_results: Sequence[Dict],
    dilate: int,
) -> List[Dict]:
    if not target_results:
        return []
    anchor = merge_result_lists(actor_results, motion_results)
    if not anchor:
        return list(target_results)
    localized = []
    for target, anchor_frame in zip(target_results, anchor):
        anchor_mask = dilate_mask(anchor_frame["combined"], dilate)
        combined = np.logical_and(target["combined"], anchor_mask)
        localized.append(
            {
                "combined": combined,
                "masks": combined[None],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
            }
        )
    return localized


def final_train_mask_results(
    role_unions: Dict[str, List[Dict]],
    motion_results: Sequence[Dict],
    target_local_dilate: int,
    final_motion_dilate: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    actor = merge_result_lists(role_unions.get("robot", []), role_unions.get("object", []))
    target = role_unions.get("target", [])
    target_local = local_target_results(
        target,
        actor,
        motion_results,
        target_local_dilate,
    )
    anchor = merge_result_lists(actor, target)
    if not anchor:
        return actor, target_local, [], []
    motion_near_final = gated_motion_results(motion_results, anchor, final_motion_dilate)
    final_mask = merge_result_lists(anchor, motion_near_final)
    return actor, target_local, motion_near_final, final_mask


def prompt_selection_config(prompt: str) -> Dict[str, float | int | bool]:
    """Heuristic instance filters for QA visualization, not final training labels."""
    prompt_l = prompt.lower()
    config: Dict[str, float | int | bool] = {
        "max_instances": 2,
        "min_area_ratio": 0.0003,
        "max_area_ratio": 0.10,
        "prefer_top": False,
        "motion_weight": 1.2,
        "top_weight": 0.0,
    }
    if "robot" in prompt_l or "gripper" in prompt_l:
        config.update(
            {
                "max_instances": 3,
                "min_area_ratio": 0.001,
                "max_area_ratio": 0.08,
                "prefer_top": True,
                "motion_weight": 0.7,
                "top_weight": 0.9,
            }
        )
    elif "drawer handle" in prompt_l:
        config.update({"max_instances": 2, "max_area_ratio": 0.025, "motion_weight": 1.6})
    elif "cabinet" in prompt_l or "stove" in prompt_l:
        config.update({"max_instances": 1, "min_area_ratio": 0.002, "max_area_ratio": 0.12, "motion_weight": 1.4})
    elif "plate" in prompt_l:
        config.update({"max_instances": 2, "min_area_ratio": 0.001, "max_area_ratio": 0.08, "motion_weight": 1.3})
    elif "basket" in prompt_l:
        config.update({"max_instances": 2, "min_area_ratio": 0.001, "max_area_ratio": 0.08, "motion_weight": 1.4})
    elif "drawer" in prompt_l:
        config.update({"max_instances": 1, "min_area_ratio": 0.001, "max_area_ratio": 0.08, "motion_weight": 1.5})
    return config


def select_prompt_frame_result(
    prompt: str,
    frame_result: Dict,
    motion_result: Dict,
    image_area: int,
    motion_dilate: int,
) -> Dict:
    masks = frame_result["masks"]
    boxes = frame_result["boxes"]
    scores = frame_result["scores"]
    if masks.shape[0] == 0:
        empty = np.zeros_like(frame_result["combined"])
        return {
            "combined": empty,
            "masks": empty[None],
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "selected_indices": [],
        }

    config = prompt_selection_config(prompt)
    motion_anchor = dilate_mask(motion_result["combined"], motion_dilate)
    candidates = []
    h = masks.shape[1]
    half_h = h // 2
    for idx, mask in enumerate(masks):
        area = int(mask.sum())
        if area <= 0:
            continue
        area_ratio = area / float(image_area)
        if area_ratio < float(config["min_area_ratio"]):
            continue
        if area_ratio > float(config["max_area_ratio"]):
            continue

        motion_overlap = float(np.logical_and(mask, motion_anchor).sum()) / float(area)
        top_fraction = float(mask[:half_h].sum()) / float(area)
        model_score = float(scores[idx]) if idx < scores.shape[0] else 0.0
        area_penalty = max(0.0, area_ratio / float(config["max_area_ratio"]) - 0.6) * 0.4
        score = (
            model_score
            + float(config["motion_weight"]) * motion_overlap
            + float(config["top_weight"]) * top_fraction
            - area_penalty
        )
        candidates.append((score, idx, motion_overlap, top_fraction, area_ratio))

    if not candidates:
        # Keep the least bad instance for diagnostics when all candidates were filtered out.
        fallback = []
        for idx, mask in enumerate(masks):
            area = int(mask.sum())
            if area <= 0:
                continue
            area_ratio = area / float(image_area)
            model_score = float(scores[idx]) if idx < scores.shape[0] else 0.0
            fallback.append((model_score - area_ratio, idx, 0.0, 0.0, area_ratio))
        candidates = fallback

    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = [idx for _, idx, _, _, _ in candidates[: int(config["max_instances"])]]
    if not selected:
        empty = np.zeros_like(frame_result["combined"])
        return {
            "combined": empty,
            "masks": empty[None],
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "selected_indices": [],
        }

    selected_masks = masks[selected].astype(bool)
    selected_boxes = boxes[selected] if boxes.shape[0] else np.zeros((0, 4), dtype=np.float32)
    selected_scores = scores[selected] if scores.shape[0] else np.array([], dtype=np.float32)
    combined = selected_masks.any(axis=0)
    return {
        "combined": combined,
        "masks": selected_masks,
        "boxes": selected_boxes,
        "scores": selected_scores,
        "selected_indices": selected,
    }


def select_prompt_results(
    prompt_results: Dict[str, List[Dict]],
    motion_results: Sequence[Dict],
    image_area: int,
    motion_dilate: int,
) -> Dict[str, List[Dict]]:
    selected: Dict[str, List[Dict]] = {}
    for prompt, results in prompt_results.items():
        selected[prompt] = [
            select_prompt_frame_result(prompt, result, motion_results[frame_idx], image_area, motion_dilate)
            for frame_idx, result in enumerate(results)
        ]
    return selected


def is_focus_prompt(prompt: str, task_text: str) -> bool:
    """Whether a prompt should anchor the focused interaction draft."""
    prompt_l = prompt.lower()
    task_l = task_text.lower()
    if "robot" in prompt_l or "gripper" in prompt_l:
        return True
    if "handle" in prompt_l:
        return True
    if any(token in prompt_l for token in ("cabinet", "stove", "top drawer", "middle drawer")):
        return False
    if "plate" in prompt_l and "push the plate" not in task_l:
        return False
    return True


def filter_focus_prompt_results(
    prompt_results: Dict[str, List[Dict]],
    task_text: str,
) -> Dict[str, List[Dict]]:
    return {
        prompt: results
        for prompt, results in prompt_results.items()
        if is_focus_prompt(prompt, task_text)
    }


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    for box in boxes:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_lines(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def make_sheet(
    frames: Sequence[np.ndarray],
    prompt_results: Dict[str, List[Dict]],
    selected_prompt_results: Dict[str, List[Dict]],
    role_unions: Dict[str, List[Dict]],
    motion_results: Sequence[Dict],
    gated_motion: Sequence[Dict],
    sam3_union_results: Sequence[Dict],
    interaction_results: Sequence[Dict],
    selected_union_results: Sequence[Dict],
    selected_focus_results: Sequence[Dict],
    selected_interaction_results: Sequence[Dict],
    actor_results: Sequence[Dict],
    target_local_results: Sequence[Dict],
    motion_near_final_results: Sequence[Dict],
    final_train_mask: Sequence[Dict],
    window: Dict,
    output_path: Path,
) -> None:
    cell_h, cell_w = frames[0].shape[:2]
    label_w = 380
    header_h = 128
    row_gap = 10
    row_specs: List[Tuple[str, str]] = [("RGB", "rgb")]
    if motion_results:
        row_specs.append(("motion cue", "motion"))
    if gated_motion:
        row_specs.append(("motion near SAM3", "gated_motion"))
    row_specs += [(prompt, f"prompt:{prompt}") for prompt in prompt_results.keys()]
    if sam3_union_results:
        row_specs.append(("SAM3 union", "sam3_union"))
    if interaction_results:
        row_specs.append(("interaction draft", "interaction"))
    if selected_union_results:
        row_specs.append(("selected SAM3", "selected_union"))
    if selected_focus_results:
        row_specs.append(("selected focus", "selected_focus"))
    if selected_interaction_results:
        row_specs.append(("selected interaction", "selected_interaction"))
    for role in ("robot", "object", "target", "context"):
        if role_unions.get(role):
            row_specs.append((f"role: {role}", f"role:{role}"))
    if actor_results:
        row_specs.append(("actor mask", "actor"))
    if target_local_results:
        row_specs.append(("local target", "target_local"))
    if motion_near_final_results:
        row_specs.append(("motion near final", "motion_near_final"))
    if final_train_mask:
        row_specs.append(("final train mask", "final_train"))

    sheet_w = label_w + cell_w * len(frames)
    sheet_h = header_h + len(row_specs) * cell_h + (len(row_specs) - 1) * row_gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    title_font = font(30)
    label_font = font(30)
    small_font = font(18)

    sample_label = window.get("sample_id", "unknown")
    sample_text = f"sample_{sample_label:03d}" if isinstance(sample_label, int) else f"sample_{sample_label}"
    title = f"{sample_text} | {window.get('task_text', '')}"
    draw.text((24, 18), title, fill=(15, 20, 25), font=title_font)
    meta = f"condition_idx={window.get('condition_idx')} | frames={window.get('frame_indices', [None])[0]}..{window.get('frame_indices', [None])[-1]} | 17F"
    draw.text((24, 68), meta, fill=(80, 85, 92), font=small_font)

    y = header_h
    for row_name, row_type in row_specs:
        draw.rectangle((0, y, label_w - 1, y + cell_h), fill=(232, 235, 239))
        for line_idx, line in enumerate(text_lines(row_name, 18)):
            draw.text((24, y + 28 + line_idx * 34), line, fill=(20, 24, 30), font=label_font)

        for frame_idx, frame in enumerate(frames):
            if row_type == "rgb":
                cell = frame
            elif row_type == "motion":
                cell = overlay_mask(frame, motion_results[frame_idx]["combined"], (255, 120, 40), alpha=0.50)
            elif row_type == "gated_motion":
                cell = overlay_mask(frame, gated_motion[frame_idx]["combined"], (255, 155, 55), alpha=0.56)
            elif row_type == "sam3_union":
                cell = frame.copy()
                for prompt_idx, prompt in enumerate(prompt_results):
                    color = PROMPT_COLORS[prompt_idx % len(PROMPT_COLORS)]
                    cell = overlay_mask(cell, prompt_results[prompt][frame_idx]["combined"], color, alpha=0.35)
                    cell = draw_boxes(cell, prompt_results[prompt][frame_idx]["boxes"], color)
            elif row_type == "interaction":
                cell = overlay_mask(frame, interaction_results[frame_idx]["combined"], (255, 80, 80), alpha=0.44)
            elif row_type == "selected_union":
                cell = frame.copy()
                for prompt_idx, prompt in enumerate(selected_prompt_results):
                    color = PROMPT_COLORS[prompt_idx % len(PROMPT_COLORS)]
                    cell = overlay_mask(cell, selected_prompt_results[prompt][frame_idx]["combined"], color, alpha=0.35)
                    cell = draw_boxes(cell, selected_prompt_results[prompt][frame_idx]["boxes"], color)
            elif row_type == "selected_focus":
                cell = overlay_mask(frame, selected_focus_results[frame_idx]["combined"], (255, 210, 60), alpha=0.48)
            elif row_type == "selected_interaction":
                cell = overlay_mask(frame, selected_interaction_results[frame_idx]["combined"], (210, 45, 45), alpha=0.48)
            elif row_type.startswith("role:"):
                role = row_type.split(":", 1)[1]
                color = ROLE_COLORS.get(role, (255, 210, 60))
                cell = overlay_mask(frame, role_unions[role][frame_idx]["combined"], color, alpha=0.48)
            elif row_type == "actor":
                cell = overlay_mask(frame, actor_results[frame_idx]["combined"], (255, 120, 35), alpha=0.48)
            elif row_type == "target_local":
                cell = overlay_mask(frame, target_local_results[frame_idx]["combined"], (35, 170, 255), alpha=0.50)
            elif row_type == "motion_near_final":
                cell = overlay_mask(frame, motion_near_final_results[frame_idx]["combined"], (255, 155, 55), alpha=0.54)
            elif row_type == "final_train":
                cell = overlay_mask(frame, final_train_mask[frame_idx]["combined"], ROLE_COLORS["final"], alpha=0.52)
            else:
                prompt = row_type.split(":", 1)[1]
                prompt_idx = list(prompt_results.keys()).index(prompt)
                color = PROMPT_COLORS[prompt_idx % len(PROMPT_COLORS)]
                result = prompt_results[prompt][frame_idx]
                cell = overlay_mask(frame, result["combined"], color)
                cell = draw_boxes(cell, result["boxes"], color)

            x = label_w + frame_idx * cell_w
            sheet.paste(Image.fromarray(cell), (x, y))
            if frame_idx == 0:
                draw.rectangle((x + 3, y + 3, x + cell_w - 4, y + cell_h - 4), outline=(40, 120, 255), width=5)
            if frame_idx == len(frames) - 1:
                draw.rectangle((x + 3, y + 3, x + cell_w - 4, y + cell_h - 4), outline=(255, 80, 80), width=5)
            draw.text((x + 8, y + 8), f"{frame_idx:02d}", fill=(255, 255, 255), font=small_font)
        y += cell_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def summarize_result_list(results: Sequence[Dict], image_area: int) -> Dict:
    if not results:
        return {}
    areas = [float(result["combined"].sum()) / float(image_area) for result in results]
    counts = [int(result["masks"].shape[0]) for result in results]
    mean_scores = [
        float(result["scores"].mean()) if result["scores"].size else 0.0
        for result in results
    ]
    top_areas = []
    bottom_areas = []
    for result in results:
        mask = result["combined"]
        half_h = mask.shape[0] // 2
        denom = float(mask.sum())
        if denom <= 0:
            top_areas.append(0.0)
            bottom_areas.append(0.0)
        else:
            top_areas.append(float(mask[:half_h].sum()) / denom)
            bottom_areas.append(float(mask[half_h:].sum()) / denom)

    return {
        "mask_area_ratio": areas,
        "num_instances": counts,
        "mean_score": mean_scores,
        "zero_frame_count": int(sum(area == 0.0 for area in areas)),
        "multi_instance_frame_count": int(sum(count > 1 for count in counts)),
        "mean_area_ratio": float(np.mean(areas)),
        "max_area_ratio": float(np.max(areas)),
        "mean_top_view_fraction": float(np.mean(top_areas)),
        "mean_bottom_view_fraction": float(np.mean(bottom_areas)),
    }


def summarize_prompt_results(prompt_results: Dict[str, List[Dict]], image_area: int) -> Dict:
    return {
        prompt: summarize_result_list(results, image_area)
        for prompt, results in prompt_results.items()
    }


def prompt_list_for_sample(sample_id: int, override: Iterable[str] | None, max_prompts: int) -> List[str]:
    if override:
        prompts = list(override)
    else:
        prompts = DEFAULT_PROMPTS.get(sample_id, ["object", "robot gripper"])
    return prompts[:max_prompts]


def write_summary_csv(all_stats: Dict[str, Dict], output_path: Path) -> None:
    fields = [
        "sample",
        "task_text",
        "kind",
        "name",
        "zero_frame_count",
        "multi_instance_frame_count",
        "mean_area_ratio",
        "max_area_ratio",
        "mean_top_view_fraction",
        "mean_bottom_view_fraction",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample_name, sample_stats in all_stats.items():
            task_text = sample_stats["window"].get("task_text", "")
            rows = {
                "motion": {"motion cue": sample_stats.get("motion", {})},
                "gated_motion": {"motion near SAM3": sample_stats.get("gated_motion", {})},
                "sam3_union": {"SAM3 union": sample_stats.get("sam3_union", {})},
                "interaction": {"interaction draft": sample_stats.get("interaction_draft", {})},
                "selected_union": {"selected SAM3": sample_stats.get("selected_union", {})},
                "selected_focus": {"selected focus": sample_stats.get("selected_focus", {})},
                "selected_interaction": {"selected interaction": sample_stats.get("selected_interaction", {})},
                "role_union": sample_stats.get("role_unions", {}),
                "v3_final": {
                    "actor mask": sample_stats.get("actor", {}),
                    "local target": sample_stats.get("target_local", {}),
                    "motion near final": sample_stats.get("motion_near_final", {}),
                    "final train mask": sample_stats.get("final_train_mask", {}),
                },
                "prompt": sample_stats.get("summary", {}),
                "selected_prompt": sample_stats.get("selected_summary", {}),
                "role_prompt": sample_stats.get("role_prompt_summary", {}),
            }
            for kind, named_stats in rows.items():
                for name, stats in named_stats.items():
                    writer.writerow(
                        {
                            "sample": sample_name,
                            "task_text": task_text,
                            "kind": kind,
                            "name": name,
                            "zero_frame_count": stats.get("zero_frame_count", ""),
                            "multi_instance_frame_count": stats.get("multi_instance_frame_count", ""),
                            "mean_area_ratio": stats.get("mean_area_ratio", ""),
                            "max_area_ratio": stats.get("max_area_ratio", ""),
                            "mean_top_view_fraction": stats.get("mean_top_view_fraction", ""),
                            "mean_bottom_view_fraction": stats.get("mean_bottom_view_fraction", ""),
                        }
                    )


def main() -> None:
    args = parse_args()
    target_size = (args.video_height, args.video_width)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_overrides = load_manual_overrides(args.manual_overrides)

    with open(args.windows_json, "r", encoding="utf-8") as file:
        windows = json.load(file)
    if not isinstance(windows, list):
        raise TypeError(f"Expected windows_json to contain a list, got {type(windows).__name__}")

    processor = build_model(args)
    all_stats = {}

    for sample_id in args.sample_ids:
        if sample_id < 0 or sample_id >= len(windows):
            raise IndexError(f"sample_id={sample_id} out of range for {len(windows)} windows")
        window = dict(windows[sample_id])
        window["sample_id"] = sample_id
        prompts = prompt_list_for_sample(sample_id, args.prompt, args.max_prompts)
        if not prompts:
            raise ValueError("At least one prompt is required")
        frames = load_window_frames(window, target_size, args.view_layout)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        motion_results = motion_results_from_frames(
            frames,
            percentile=args.motion_percentile,
            min_diff=args.motion_min_diff,
            dilate=args.motion_dilate,
        )

        prompt_results: Dict[str, List[Dict]] = {prompt: [] for prompt in prompts}
        manual_prompt_results: Dict[str, List[Dict]] = {}
        for frame_idx, pil_frame in enumerate(pil_frames):
            frame_results = run_prompts_on_image(processor, pil_frame, prompts, args.dtype, args.device)
            for prompt, result in frame_results.items():
                prompt_results[prompt].append(result)
                print(
                    f"sample={sample_id:03d} frame={frame_idx:02d} prompt={prompt!r} "
                    f"instances={result['masks'].shape[0]} area={int(result['combined'].sum())}",
                    flush=True,
                )
            manual_entries = manual_entries_for_sample_frame(
                manual_overrides,
                sample_id,
                frame_idx,
                pil_frame.width,
                pil_frame.height,
            )
            manual_results = run_manual_entries_on_image(
                processor,
                pil_frame,
                manual_entries,
                args.dtype,
                args.device,
                args.manual_box_pad_ratio,
                args.manual_box_pad_pixels,
            )
            for key in list(manual_prompt_results):
                if key not in manual_results:
                    manual_prompt_results[key].append(empty_result(pil_frame.height, pil_frame.width, source="manual"))
            for key, result in manual_results.items():
                if key not in manual_prompt_results:
                    manual_prompt_results[key] = [
                        empty_result(pil_frame.height, pil_frame.width, source="manual")
                        for _ in range(frame_idx)
                    ]
                manual_prompt_results[key].append(result)
                print(
                    f"sample={sample_id:03d} frame={frame_idx:02d} prompt={key!r} "
                    f"instances={result['masks'].shape[0]} area={int(result['combined'].sum())} "
                    f"box={result.get('manual_box_xyxy_abs')}",
                    flush=True,
                )

        for key, results in manual_prompt_results.items():
            while len(results) < len(frames):
                results.append(empty_result(frames[0].shape[0], frames[0].shape[1], source="manual"))

        image_area = frames[0].shape[0] * frames[0].shape[1]
        display_prompt_results = dict(prompt_results)
        display_prompt_results.update(manual_prompt_results)
        sam3_union = union_results(display_prompt_results)
        gated_motion = gated_motion_results(motion_results, sam3_union, args.interaction_dilate)
        interaction_draft = merge_result_lists(gated_motion, sam3_union)
        selected_prompt_results = select_prompt_results(
            prompt_results,
            motion_results,
            image_area,
            args.selection_motion_dilate,
        )
        selected_prompt_results.update(manual_prompt_results)
        selected_union = union_results(selected_prompt_results)
        focus_prompt_results = filter_focus_prompt_results(
            selected_prompt_results,
            str(window.get("task_text", "")),
        )
        selected_focus = union_results(focus_prompt_results)
        selected_gated_motion = gated_motion_results(motion_results, selected_focus, args.interaction_dilate)
        selected_interaction = merge_result_lists(selected_gated_motion, selected_focus)
        role_results = role_prompt_results(
            selected_prompt_results,
            str(window.get("task_text", "")),
            image_area,
            args.view_layout,
            args.view_edge_margin,
        )
        role_unions = union_role_results(role_results)
        actor, target_local, motion_near_final, final_train_mask = final_train_mask_results(
            role_unions,
            motion_results,
            args.target_local_dilate,
            args.final_motion_dilate,
        )
        sample_dir = output_dir / f"sample_{sample_id:03d}"
        make_sheet(
            frames,
            display_prompt_results,
            selected_prompt_results,
            role_unions,
            motion_results,
            gated_motion,
            sam3_union,
            interaction_draft,
            selected_union,
            selected_focus,
            selected_interaction,
            actor,
            target_local,
            motion_near_final,
            final_train_mask,
            window,
            sample_dir / f"sample_{sample_id:03d}_sam3_mask_debug_17f.png",
        )
        role_prompt_summary = {}
        for role, prompt_map in role_results.items():
            for prompt, results in prompt_map.items():
                role_prompt_summary[f"{role}/{prompt}"] = summarize_result_list(results, image_area)
        stats = {
            "window": window,
            "prompts": prompts,
            "manual_override_path": args.manual_overrides,
            "prompt_roles": {
                prompt: prompt_role(prompt, str(window.get("task_text", "")))
                for prompt in display_prompt_results
            },
            "motion": summarize_result_list(motion_results, image_area),
            "gated_motion": summarize_result_list(gated_motion, image_area),
            "sam3_union": summarize_result_list(sam3_union, image_area),
            "interaction_draft": summarize_result_list(interaction_draft, image_area),
            "selected_union": summarize_result_list(selected_union, image_area),
            "selected_focus": summarize_result_list(selected_focus, image_area),
            "selected_interaction": summarize_result_list(selected_interaction, image_area),
            "role_unions": {
                role: summarize_result_list(results, image_area)
                for role, results in role_unions.items()
            },
            "actor": summarize_result_list(actor, image_area),
            "target_local": summarize_result_list(target_local, image_area),
            "motion_near_final": summarize_result_list(motion_near_final, image_area),
            "final_train_mask": summarize_result_list(final_train_mask, image_area),
            "summary": summarize_prompt_results(display_prompt_results, image_area),
            "selected_summary": summarize_prompt_results(selected_prompt_results, image_area),
            "role_prompt_summary": role_prompt_summary,
        }
        with open(sample_dir / f"sample_{sample_id:03d}_sam3_mask_debug_stats.json", "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
        all_stats[f"sample_{sample_id:03d}"] = stats

    with open(output_dir / "sam3_mask_debug_summary.json", "w", encoding="utf-8") as file:
        json.dump(all_stats, file, indent=2)
    write_summary_csv(all_stats, output_dir / "sam3_mask_debug_summary.csv")


if __name__ == "__main__":
    main()
