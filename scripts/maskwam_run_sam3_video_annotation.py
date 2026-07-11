#!/usr/bin/env python3
"""Run SAM3 video propagation for MaskWAM-style LIBERO mask annotations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import av
import cv2
import numpy as np
from PIL import Image

from maskwam_view_schema import (
    BASE_TRAIN_ROLES,
    FINAL_ROLE,
    OPTIONAL_BASE_ROLES,
    VIEW_NAMES,
    VIEW_TRAIN_ROLES,
    base_role,
    role_color,
    split_masks_by_view,
    union_view_train_roles,
    view_role,
)

ROLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "object": (255, 60, 60),
    "target": (45, 175, 255),
    "robot": (185, 100, 255),
    "context": (120, 150, 160),
    "final_train_mask": (235, 40, 40),
    **{role: role_color(role) for role in VIEW_TRAIN_ROLES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--output_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass")
    parser.add_argument("--corrections", default=None, help="Optional point/box correction JSON.")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--gpus_to_use", nargs="+", type=int, default=None)
    parser.add_argument("--output_prob_thresh", type=float, default=0.5)
    parser.add_argument("--video_fps", type=float, default=8.0)
    parser.add_argument("--keep_composed_video", action="store_true")
    parser.add_argument("--skip_context", action="store_true", help="Do not run context prompts.")
    parser.add_argument(
        "--point_box_size",
        type=int,
        default=18,
        help="Convert clicked points to this square box size before calling SAM3 video propagation.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "prompt"


def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    src_h, src_w = frame.shape[:2]
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
        target_h = max(frame.shape[0] for frame in view_frames)
        target_w = max(frame.shape[1] for frame in view_frames)
        aligned = [
            frame if frame.shape[:2] == (target_h, target_w) else resize_with_padding(frame, (target_h, target_w))
            for frame in view_frames
        ]
        if view_layout == "vertical":
            frame = np.concatenate(aligned, axis=0)
        elif view_layout == "horizontal":
            frame = np.concatenate(aligned, axis=1)
        else:
            raise ValueError(f"Unsupported view_layout={view_layout!r}")
    if frame.shape[:2] != target_size:
        frame = resize_with_padding(frame, target_size)
    return frame


def load_sample_frames(sample: Dict[str, Any]) -> List[np.ndarray]:
    frame_indices = sample["frame_indices"]
    target_size = (int(sample["target_size"]["height"]), int(sample["target_size"]["width"]))
    view_layout = sample.get("view_layout", "vertical")
    batches = [read_video_frames(path, frame_indices) for path in sample["video_paths"]]
    frames = []
    for frame_pos in range(len(frame_indices)):
        frames.append(compose_views([batch[frame_pos] for batch in batches], target_size, view_layout))
    return frames


def save_composed_video(frames: Sequence[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_np in frames:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def save_rgb_frames(frames: Sequence[np.ndarray], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"frame_{idx:03d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    return paths


def mask_from_outputs(outputs: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return np.zeros(shape, dtype=bool)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.size == 0 or masks.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    return masks.astype(bool).any(axis=0)


def save_mask_sequence(masks: Sequence[np.ndarray], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, mask in enumerate(masks):
        path = output_dir / f"frame_{idx:03d}.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)
        paths.append(str(path))
    return paths


def read_mask_sequence(mask_dir: Path, frame_count: int) -> List[np.ndarray]:
    masks = []
    for idx in range(frame_count):
        path = mask_dir / f"frame_{idx:03d}.png"
        if not path.exists():
            masks.append(None)
            continue
        masks.append(np.asarray(Image.open(path).convert("L")) > 0)
    return masks


def union_masks(mask_lists: Iterable[Sequence[np.ndarray]], frame_count: int, shape: Tuple[int, int]) -> List[np.ndarray]:
    out = []
    for idx in range(frame_count):
        combined = np.zeros(shape, dtype=bool)
        for masks in mask_lists:
            if idx < len(masks) and masks[idx] is not None:
                combined |= masks[idx].astype(bool)
        out.append(combined)
    return out


def corrections_for_prompt(corrections: Dict[str, Any], sample_id: int, prompt_name: str, role: str) -> List[Dict[str, Any]]:
    sample_cfg = corrections.get("samples", {}).get(str(sample_id), {})
    entries = sample_cfg.get("prompts", [])
    matched = []
    for entry in entries:
        has_points = bool(entry.get("points_abs"))
        has_boxes = bool(entry.get("boxes_xywh_abs"))
        if entry.get("accepted_absent"):
            continue
        if not has_points and not has_boxes:
            continue
        entry_name = entry.get("name")
        name_matches = entry_name == prompt_name
        entry_role = entry.get("role")
        entry_base_role = base_role(entry_role)
        role_matches = entry_name in (None, "", "manual point", "manual box") and entry_base_role == role
        if entry_base_role != role:
            continue
        if name_matches or role_matches:
            matched.append(entry)
    return matched


def accepted_absent_for_view_prompt(
    corrections: Dict[str, Any],
    sample_id: int,
    prompt_name: str,
    view_prompt_role: str,
) -> List[Dict[str, Any]]:
    sample_cfg = corrections.get("samples", {}).get(str(sample_id), {})
    entries = sample_cfg.get("prompts", [])
    matched = []
    for entry in entries:
        if not entry.get("accepted_absent"):
            continue
        if entry.get("role") != view_prompt_role:
            continue
        if entry.get("name") != prompt_name:
            continue
        matched.append(entry)
    return matched


def mask_area_summary(masks: Sequence[np.ndarray], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    areas = [int(mask.sum()) for mask in masks]
    return {
        "zero_frame_count": int(sum(area == 0 for area in areas)),
        "mean_area_ratio": float(np.mean(areas) / float(frame_shape[0] * frame_shape[1])),
    }


def abs_points_to_rel(points: Sequence[Sequence[float]], frame_shape: Tuple[int, int]) -> List[List[float]]:
    height, width = frame_shape
    return [[float(x) / float(width), float(y) / float(height)] for x, y in points]


def rel_points_to_abs(points: Sequence[Sequence[float]], frame_shape: Tuple[int, int]) -> List[List[float]]:
    height, width = frame_shape
    return [[float(x) * float(width), float(y) * float(height)] for x, y in points]


def abs_boxes_xywh_to_rel(boxes: Sequence[Sequence[float]], frame_shape: Tuple[int, int]) -> List[List[float]]:
    height, width = frame_shape
    return [
        [
            float(x) / float(width),
            float(y) / float(height),
            float(w) / float(width),
            float(h) / float(height),
        ]
        for x, y, w, h in boxes
    ]


def rel_boxes_xywh_to_abs(boxes: Sequence[Sequence[float]], frame_shape: Tuple[int, int]) -> List[List[float]]:
    height, width = frame_shape
    return [
        [
            float(x) * float(width),
            float(y) * float(height),
            float(w) * float(width),
            float(h) * float(height),
        ]
        for x, y, w, h in boxes
    ]


def points_abs_to_boxes_xywh_abs(
    points: Sequence[Sequence[float]],
    frame_shape: Tuple[int, int],
    box_size: int,
) -> List[List[float]]:
    height, width = frame_shape
    half = max(1.0, float(box_size) * 0.5)
    boxes = []
    for x, y in points:
        x0 = max(0.0, float(x) - half)
        y0 = max(0.0, float(y) - half)
        x1 = min(float(width), float(x) + half)
        y1 = min(float(height), float(y) + half)
        boxes.append([x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)])
    return boxes


def add_prompt_request(
    session_id: str,
    prompt: Dict[str, Any],
    correction: Dict[str, Any] | None,
    frame_count: int,
    frame_shape: Tuple[int, int],
    point_box_size: int,
) -> Dict[str, Any]:
    frame_index = int((correction or {}).get("frame_index", 0))
    frame_index = max(0, min(frame_count - 1, frame_index))
    request: Dict[str, Any] = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_index,
        "output_prob_thresh": 0.5,
        "rel_coordinates": True,
    }
    if correction:
        if correction.get("points_abs"):
            points = abs_points_to_rel(correction["points_abs"], frame_shape)
            labels = correction.get("point_labels", [1] * len(points))
            request["points"] = points
            request["point_labels"] = labels
            request["obj_id"] = int(correction.get("obj_id", 1))
        elif correction.get("points_rel"):
            points = correction["points_rel"]
            labels = correction.get("point_labels", [1] * len(points))
            request["points"] = points
            request["point_labels"] = labels
            request["obj_id"] = int(correction.get("obj_id", 1))
        if correction.get("boxes_xywh_abs"):
            boxes = abs_boxes_xywh_to_rel(correction["boxes_xywh_abs"], frame_shape)
            labels = correction.get("box_labels", [1] * len(boxes))
            request["bounding_boxes"] = boxes
            request["bounding_box_labels"] = labels
            request["obj_id"] = int(correction.get("obj_id", 1))
        elif correction.get("boxes_xywh_rel"):
            boxes = correction["boxes_xywh_rel"]
            labels = correction.get("box_labels", [1] * len(boxes))
            request["bounding_boxes"] = boxes
            request["bounding_box_labels"] = labels
            request["obj_id"] = int(correction.get("obj_id", 1))
    if "points" not in request and "bounding_boxes" not in request:
        request["text"] = prompt["text"]
        request["rel_coordinates"] = True
    return request


def draw_box(mask: np.ndarray, box_xywh_abs: Sequence[float], value: bool) -> None:
    height, width = mask.shape
    x, y, w, h = [float(value) for value in box_xywh_abs]
    x0 = max(0, min(width, int(round(x))))
    y0 = max(0, min(height, int(round(y))))
    x1 = max(0, min(width, int(round(x + w))))
    y1 = max(0, min(height, int(round(y + h))))
    if x1 <= x0 or y1 <= y0:
        return
    mask[y0:y1, x0:x1] = value


def manual_correction_fallback_masks(
    correction_entries: Sequence[Dict[str, Any]],
    frame_count: int,
    frame_shape: Tuple[int, int],
    point_box_size: int,
) -> List[np.ndarray]:
    keyed_masks: Dict[int, np.ndarray] = {}
    for correction in correction_entries:
        frame_index = int(correction.get("frame_index", 0))
        frame_index = max(0, min(frame_count - 1, frame_index))
        mask = keyed_masks.setdefault(frame_index, np.zeros(frame_shape, dtype=bool))

        boxes_abs = list(correction.get("boxes_xywh_abs") or [])
        boxes_abs.extend(rel_boxes_xywh_to_abs(correction.get("boxes_xywh_rel") or [], frame_shape))
        box_labels = list(correction.get("box_labels") or [1] * len(boxes_abs))
        if len(box_labels) < len(boxes_abs):
            box_labels.extend([1] * (len(boxes_abs) - len(box_labels)))
        for box, label in zip(boxes_abs, box_labels):
            draw_box(mask, box, bool(int(label) > 0))

        points_abs = list(correction.get("points_abs") or [])
        points_abs.extend(rel_points_to_abs(correction.get("points_rel") or [], frame_shape))
        point_labels = list(correction.get("point_labels") or [1] * len(points_abs))
        if len(point_labels) < len(points_abs):
            point_labels.extend([1] * (len(points_abs) - len(point_labels)))
        point_boxes = points_abs_to_boxes_xywh_abs(points_abs, frame_shape, point_box_size)
        for box, label in zip(point_boxes, point_labels):
            draw_box(mask, box, bool(int(label) > 0))

    positive_frames = [idx for idx, mask in keyed_masks.items() if mask.any()]
    if not positive_frames:
        return [np.zeros(frame_shape, dtype=bool) for _ in range(frame_count)]

    out = []
    for frame_index in range(frame_count):
        nearest = min(positive_frames, key=lambda value: (abs(value - frame_index), value))
        out.append(keyed_masks[nearest].copy())
    return out


def correction_start_frame(correction_entries: Sequence[Dict[str, Any]], frame_count: int) -> int | None:
    if not correction_entries:
        return None
    frame_indices = []
    for correction in correction_entries:
        frame_index = int(correction.get("frame_index", 0))
        frame_indices.append(max(0, min(frame_count - 1, frame_index)))
    return min(frame_indices) if frame_indices else None


def run_prompt_propagation(
    predictor: Any,
    resource_path: Path,
    prompt: Dict[str, Any],
    correction_entries: Sequence[Dict[str, Any]],
    frame_count: int,
    frame_shape: Tuple[int, int],
    output_prob_thresh: float,
    point_box_size: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    status: Dict[str, Any] = {"fallback_used": False, "fallback_reason": ""}
    if correction_entries and any(entry.get("force_manual_mask") for entry in correction_entries):
        fallback_masks = manual_correction_fallback_masks(correction_entries, frame_count, frame_shape, point_box_size)
        if any(mask.any() for mask in fallback_masks):
            return fallback_masks, {"fallback_used": True, "fallback_reason": "manual_forced"}

    session_id = None
    try:
        response = predictor.handle_request({"type": "start_session", "resource_path": str(resource_path)})
        session_id = response["session_id"]
        entries = list(correction_entries) or [None]
        for correction in entries:
            request = add_prompt_request(session_id, prompt, correction, frame_count, frame_shape, point_box_size)
            request["output_prob_thresh"] = output_prob_thresh
            predictor.handle_request(request)
        masks_by_frame: Dict[int, np.ndarray] = {}
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "start_frame_index": correction_start_frame(correction_entries, frame_count),
                "output_prob_thresh": output_prob_thresh,
            }
        ):
            frame_index = int(response["frame_index"])
            outputs = response["outputs"]
            masks_by_frame[frame_index] = mask_from_outputs(outputs, frame_shape)
        if not masks_by_frame:
            raise RuntimeError(f"No propagation output for prompt {prompt['name']!r}")
        masks = [masks_by_frame.get(idx, np.zeros(frame_shape, dtype=bool)) for idx in range(frame_count)]
        if correction_entries and not any(mask.any() for mask in masks):
            fallback_masks = manual_correction_fallback_masks(correction_entries, frame_count, frame_shape, point_box_size)
            if any(mask.any() for mask in fallback_masks):
                status["fallback_used"] = True
                status["fallback_reason"] = "sam3_correction_all_zero"
                masks = fallback_masks
        elif correction_entries and any(not mask.any() for mask in masks):
            fallback_masks = manual_correction_fallback_masks(correction_entries, frame_count, frame_shape, point_box_size)
            if any(mask.any() for mask in fallback_masks):
                filled = 0
                for frame_index, mask in enumerate(masks):
                    if not mask.any() and fallback_masks[frame_index].any():
                        masks[frame_index] = fallback_masks[frame_index]
                        filled += 1
                if filled:
                    status["fallback_used"] = True
                    status["fallback_reason"] = f"sam3_partial_zero_filled:{filled}"
        return masks, status
    except Exception as error:
        if correction_entries:
            fallback_masks = manual_correction_fallback_masks(correction_entries, frame_count, frame_shape, point_box_size)
            if any(mask.any() for mask in fallback_masks):
                status["fallback_used"] = True
                status["fallback_reason"] = f"sam3_exception:{type(error).__name__}:{error}"
                return fallback_masks, status
        raise
    finally:
        if session_id is not None:
            predictor.handle_request({"type": "close_session", "session_id": session_id, "run_gc_collect": True})


def build_predictor(version: str, gpus_to_use: Sequence[int] | None) -> Any:
    from sam3.model_builder import build_sam3_predictor

    kwargs: Dict[str, Any] = {"version": version}
    if version == "sam3":
        kwargs["gpus_to_use"] = gpus_to_use
    return build_sam3_predictor(**kwargs)


def select_samples(manifest: Dict[str, Any], sample_ids: Sequence[int] | None, max_samples: int | None) -> List[Dict[str, Any]]:
    samples = manifest["samples"]
    if sample_ids is not None:
        wanted = {int(value) for value in sample_ids}
        samples = [sample for sample in samples if int(sample["sample_id"]) in wanted]
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    corrections = load_json(args.corrections) if args.corrections else {"samples": {}}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(args.sam3_version, args.gpus_to_use)

    run_summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "corrections": str(Path(args.corrections).resolve()) if args.corrections else None,
        "sam3_version": args.sam3_version,
        "samples": [],
    }
    try:
        for sample in select_samples(manifest, args.sample_ids, args.max_samples):
            sample_id = int(sample["sample_id"])
            sample_name = sample["sample_name"]
            sample_dir = output_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            frames = load_sample_frames(sample)
            frame_shape = frames[0].shape[:2]
            rgb_dir = sample_dir / "rgb"
            save_rgb_frames(frames, rgb_dir)
            video_path = sample_dir / "composed_17f.mp4"
            if args.keep_composed_video:
                save_composed_video(frames, video_path, fps=args.video_fps)

            prompt_summaries = []
            masks_by_role: Dict[str, List[Sequence[np.ndarray]]] = {
                role: [] for role in (*BASE_TRAIN_ROLES, *OPTIONAL_BASE_ROLES, *VIEW_TRAIN_ROLES)
            }
            for prompt in sample["prompts"]:
                if args.skip_context and prompt["role"] == "context":
                    continue
                corrections_for_this_prompt = corrections_for_prompt(
                    corrections,
                    sample_id,
                    prompt["name"],
                    prompt["role"],
                )
                masks, propagation_status = run_prompt_propagation(
                    predictor,
                    rgb_dir,
                    prompt,
                    corrections_for_this_prompt,
                    len(frames),
                    frame_shape,
                    args.output_prob_thresh,
                    args.point_box_size,
                )
                prompt_slug = f"{prompt['role']}__{slugify(prompt['name'])}"
                mask_dir = sample_dir / "masks" / "by_prompt" / prompt_slug
                save_mask_sequence(masks, mask_dir)
                masks_by_role.setdefault(prompt["role"], []).append(masks)
                prompt_summaries.append(
                    {
                        "name": prompt["name"],
                        "role": prompt["role"],
                        "text": prompt["text"],
                        "mask_dir": str(mask_dir),
                        **mask_area_summary(masks, frame_shape),
                        "correction_count": len(corrections_for_this_prompt),
                        "fallback_used": bool(propagation_status["fallback_used"]),
                        "fallback_reason": propagation_status["fallback_reason"],
                        "view_derived": False,
                    }
                )
                if prompt["role"] in BASE_TRAIN_ROLES:
                    for view, view_masks in split_masks_by_view(masks, sample.get("view_layout", "vertical")).items():
                        view_prompt_role = view_role(view, prompt["role"])
                        accepted_absent_entries = accepted_absent_for_view_prompt(
                            corrections,
                            sample_id,
                            prompt["name"],
                            view_prompt_role,
                        )
                        if accepted_absent_entries:
                            view_masks = [np.zeros(frame_shape, dtype=bool) for _ in view_masks]
                        view_prompt_slug = f"{view_prompt_role}__{slugify(prompt['name'])}"
                        view_mask_dir = sample_dir / "masks" / "by_prompt" / view_prompt_slug
                        save_mask_sequence(view_masks, view_mask_dir)
                        masks_by_role.setdefault(view_prompt_role, []).append(view_masks)
                        prompt_summaries.append(
                            {
                                "name": prompt["name"],
                                "role": view_prompt_role,
                                "base_role": prompt["role"],
                                "view": view,
                                "text": prompt["text"],
                                "mask_dir": str(view_mask_dir),
                                **mask_area_summary(view_masks, frame_shape),
                                "correction_count": len(corrections_for_this_prompt),
                                "accepted_absent": bool(accepted_absent_entries),
                                "accepted_absent_count": len(accepted_absent_entries),
                                "accepted_absent_reasons": [
                                    entry.get("absent_reason", "") for entry in accepted_absent_entries
                                ],
                                "fallback_used": bool(propagation_status["fallback_used"]),
                                "fallback_reason": propagation_status["fallback_reason"],
                                "view_derived": True,
                            }
                        )
                print(f"{sample_name} prompt={prompt['name']!r} role={prompt['role']} done", flush=True)

            role_summaries = {}
            for role in (*BASE_TRAIN_ROLES, *OPTIONAL_BASE_ROLES, *VIEW_TRAIN_ROLES):
                role_masks = union_masks(masks_by_role.get(role, []), len(frames), frame_shape)
                save_mask_sequence(role_masks, sample_dir / "masks" / role)
                role_summaries[role] = mask_area_summary(role_masks, frame_shape)

            role_to_masks = {
                role: read_mask_sequence(sample_dir / "masks" / role, len(frames))
                for role in VIEW_TRAIN_ROLES
            }
            final_masks = union_view_train_roles(
                role_to_masks,
                len(frames),
                frame_shape,
            )
            save_mask_sequence(final_masks, sample_dir / "masks" / FINAL_ROLE)
            sample_meta = {
                "sample": sample,
                "view_schema": {
                    "enabled": True,
                    "view_layout": sample.get("view_layout", "vertical"),
                    "views": list(VIEW_NAMES),
                    "train_roles": list(VIEW_TRAIN_ROLES),
                    "final_train_mask": "union of view-aware object/target/robot roles",
                },
                "prompt_summaries": prompt_summaries,
                "role_summaries": role_summaries,
                "final_train_mask": mask_area_summary(final_masks, frame_shape),
            }
            write_json(sample_dir / "annotation_metadata.json", sample_meta)
            run_summary["samples"].append(
                {
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "metadata": str(sample_dir / "annotation_metadata.json"),
                    "final_train_mask_dir": str(sample_dir / "masks" / "final_train_mask"),
                }
            )
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()

    write_json(output_dir / "annotation_run_summary.json", run_summary)
    print(output_dir / "annotation_run_summary.json")


if __name__ == "__main__":
    main()
