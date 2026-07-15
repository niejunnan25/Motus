#!/usr/bin/env python3
"""Render RGB-synchronized Progress and trajectory-alignment diagnostic videos."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_COLORS = {
    "A0": (230, 96, 45),
    "A1": (45, 45, 230),
    "A2": (48, 166, 47),
    "A3": (210, 48, 148),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress_eval_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--benchmark_json", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--vae_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["A2"])
    parser.add_argument("--episode_indices", nargs="*", type=int, default=[])
    parser.add_argument("--case_selection_csv", type=Path, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_benchmark(path: Path) -> tuple[List[str], List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_paths = payload.get("sample_path_list")
    task_texts = payload.get("sample_path_task")
    if not isinstance(episode_paths, list) or not isinstance(task_texts, list):
        raise ValueError(f"Invalid Robo-Dopamine benchmark JSON: {path}")
    if len(episode_paths) != len(task_texts):
        raise ValueError("Benchmark episode and task lists have different lengths")
    return [str(value) for value in episode_paths], [str(value) for value in task_texts]


def load_progress_outputs(root: Path, models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    for model in models:
        path = root / model / "episode_outputs.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Progress outputs for {model}: {path}")
        payload = torch_load(path)
        episodes: Dict[str, Any] = {}
        for episode in payload["episodes"]:
            name = str(episode["episode_name"])
            episodes[name] = episode
        outputs[model] = episodes
    reference = set(next(iter(outputs.values())))
    for model, episodes in outputs.items():
        if set(episodes) != reference:
            raise ValueError(f"Episode set differs for model {model}")
    return outputs


def load_cache_manifest(cache_dir: Path) -> Dict[str, Path]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    result: Dict[str, Path] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            result[str(entry["episode_name"])] = cache_dir / str(entry["cache_file"])
    return result


def selected_episode_indices(args: argparse.Namespace, count: int) -> List[int]:
    indices: List[int] = []
    for index in args.episode_indices:
        if index < 0 or index >= count:
            raise IndexError(f"episode index {index} is outside [0, {count})")
        if index not in indices:
            indices.append(index)
    if args.case_selection_csv is not None:
        with args.case_selection_csv.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                name = str(row["episode_name"])
                suffix = name.rsplit("episode_", 1)[-1]
                index = int(suffix)
                if index not in indices:
                    indices.append(index)
    if not indices:
        raise ValueError("Provide --episode_indices or --case_selection_csv")
    if args.max_cases is not None:
        indices = indices[: max(0, int(args.max_cases))]
    return indices


def list_frames(directory: Path) -> List[Path]:
    frames = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        raise FileNotFoundError(f"No image frames under {directory}")
    return frames


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


class TrajectoryDecoder:
    def __init__(self, repo_root: Path, vae_path: Path, device: str) -> None:
        bak_root = str((repo_root / "bak").resolve())
        if bak_root not in sys.path:
            sys.path.insert(0, bak_root)
        from wan.modules.vae2_2 import Wan2_2_VAE

        self.device = device
        self.vae = Wan2_2_VAE(vae_pth=str(vae_path), device=device)

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        value = latent.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            decoded = self.vae.decode([value])[0]
        frames = (
            (decoded.float().clamp(-1.0, 1.0) + 1.0)
            .mul(127.5)
            .byte()
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
        return frames[..., ::-1].copy()


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (32, 32, 32),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def paste_letterboxed(
    canvas: np.ndarray,
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    background: tuple[int, int, int] = (238, 240, 243),
) -> None:
    x, y, width, height = rect
    canvas[y : y + height, x : x + width] = background
    scale = min(width / image.shape[1], height / image.shape[0])
    new_width = max(1, int(round(image.shape[1] * scale)))
    new_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left = x + (width - new_width) // 2
    top = y + (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (185, 190, 198), 1)


def chart_points(
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> np.ndarray:
    x, y, width, height = rect
    if values.size == 1:
        xs = np.asarray([x + width // 2], dtype=np.int32)
    else:
        xs = np.linspace(x, x + width - 1, values.size).round().astype(np.int32)
    normalized = np.clip((values - y_min) / max(1e-8, y_max - y_min), 0.0, 1.0)
    ys = (y + height - 1 - normalized * (height - 1)).round().astype(np.int32)
    return np.stack([xs, ys], axis=1)


def draw_progress_chart(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    target: np.ndarray,
    prediction: np.ndarray,
    current: int,
    model_color: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (250, 250, 250), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (170, 175, 182), 1)
    for fraction in np.linspace(0.0, 1.0, 6):
        line_y = int(round(y + height - fraction * height))
        cv2.line(canvas, (x, line_y), (x + width, line_y), (224, 226, 230), 1)
        put_text(canvas, f"{fraction:.1f}", (x + 4, max(y + 14, line_y - 3)), scale=0.38)
    target_points = chart_points(target, rect)
    prediction_points = chart_points(prediction, rect)
    cv2.polylines(canvas, [target_points], False, (38, 38, 38), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [prediction_points], False, model_color, 2, cv2.LINE_AA)
    current_x = int(prediction_points[current, 0])
    cv2.line(canvas, (current_x, y), (current_x, y + height), (80, 80, 80), 1)
    cv2.circle(canvas, tuple(target_points[current]), 5, (38, 38, 38), -1)
    cv2.circle(canvas, tuple(prediction_points[current]), 5, model_color, -1)
    put_text(canvas, "GT", (x + width - 105, y + 18), scale=0.42, color=(38, 38, 38), thickness=2)
    put_text(canvas, "Pred", (x + width - 65, y + 18), scale=0.42, color=model_color, thickness=2)
    put_text(canvas, "Progress over source episode", (x + 8, y - 8), scale=0.52, thickness=2)


def alignment_heatmap(alignment: np.ndarray) -> np.ndarray:
    positive = alignment[alignment > 0]
    vmax = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    normalized = np.clip(alignment / max(vmax, 1e-8), 0.0, 1.0)
    gray = np.round(normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)


def draw_alignment_chart(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    heatmap: np.ndarray,
    target: np.ndarray,
    argmax_slots: np.ndarray,
    current: int,
) -> None:
    x, y, width, height = rect
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas[y : y + height, x : x + width] = resized
    num_queries = target.size
    num_slots = heatmap.shape[1]

    def point(query_index: int, slot: float) -> tuple[int, int]:
        px = x + int(round(slot / max(1, num_slots - 1) * (width - 1)))
        py = y + int(round(query_index / max(1, num_queries - 1) * (height - 1)))
        return px, py

    gt_points = np.asarray(
        [point(index, float(target[index] * (num_slots - 1))) for index in range(num_queries)],
        dtype=np.int32,
    )
    match_points = np.asarray(
        [point(index, float(argmax_slots[index])) for index in range(num_queries)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [gt_points], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [match_points], False, (255, 220, 30), 1, cv2.LINE_AA)
    current_y = point(current, 0.0)[1]
    cv2.line(canvas, (x, current_y), (x + width, current_y), (20, 20, 230), 2)
    cv2.circle(canvas, point(current, float(argmax_slots[current])), 6, (20, 20, 230), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (170, 175, 182), 1)
    put_text(
        canvas,
        "Alignment: white=time label, cyan=argmax, red=current",
        (x + 8, y - 8),
        scale=0.48,
        thickness=2,
    )
    put_text(canvas, "slot 0", (x + 4, y + height - 6), scale=0.38, color=(255, 255, 255))
    put_text(
        canvas,
        f"slot {num_slots - 1}",
        (x + width - 58, y + height - 6),
        scale=0.38,
        color=(255, 255, 255),
    )


def draw_distribution(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    probabilities: np.ndarray,
    gt_slot: float,
    matched_slot: int,
    model_color: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (247, 248, 250), -1)
    maximum = max(float(probabilities.max()), 1e-8)
    bar_width = width / probabilities.size
    for slot, probability in enumerate(probabilities):
        left = int(round(x + slot * bar_width))
        right = int(round(x + (slot + 1) * bar_width))
        bar_height = int(round(probability / maximum * (height - 16)))
        cv2.rectangle(
            canvas,
            (left, y + height - bar_height),
            (max(left + 1, right - 1), y + height),
            model_color,
            -1,
        )
    gt_x = x + int(round(gt_slot / max(1, probabilities.size - 1) * width))
    match_x = x + int(round(matched_slot / max(1, probabilities.size - 1) * width))
    cv2.line(canvas, (gt_x, y), (gt_x, y + height), (25, 25, 25), 2)
    cv2.line(canvas, (match_x, y), (match_x, y + height), (20, 20, 230), 2)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (170, 175, 182), 1)
    put_text(
        canvas,
        "Current 53-slot probability (black=time label, red=argmax)",
        (x, y - 7),
        scale=0.44,
        thickness=2,
    )


def wrap_header(text: str, width: int = 125) -> List[str]:
    return textwrap.wrap(" ".join(text.split()), width=width, break_long_words=False)[:2]


def render_frame(
    *,
    width: int,
    height: int,
    model: str,
    task_text: str,
    episode_name: str,
    query_index: int,
    frame_index: int,
    high_frame: np.ndarray,
    wrist_frame: np.ndarray,
    generated_frames: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    alignment: np.ndarray,
    heatmap: np.ndarray,
    argmax_slots: np.ndarray,
) -> np.ndarray:
    if width < 1280 or height < 720:
        raise ValueError("Video canvas must be at least 1280x720")
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    header_height = 74
    canvas[:header_height] = (36, 42, 51)
    header_lines = wrap_header(f"{episode_name} | {task_text}")
    for line_index, line in enumerate(header_lines):
        put_text(
            canvas,
            line,
            (18, 27 + line_index * 25),
            scale=0.58,
            color=(245, 245, 245),
            thickness=1,
        )

    margin = 18
    top_y = header_height + 24
    top_height = int(height * 0.43)
    square_width = int(width * 0.155)
    memory_width = int(width * 0.12)
    gap = 14
    image_rects = [
        (margin, top_y, square_width, top_height),
        (margin + square_width + gap, top_y, square_width, top_height),
        (margin + 2 * (square_width + gap), top_y, memory_width, top_height),
        (margin + 2 * (square_width + gap) + memory_width + gap, top_y, memory_width, top_height),
    ]

    gt_slot_float = float(target[query_index] * (alignment.shape[1] - 1))
    gt_slot = int(np.clip(round(gt_slot_float), 0, generated_frames.shape[0] - 1))
    matched_slot = int(argmax_slots[query_index])
    paste_letterboxed(canvas, high_frame, image_rects[0])
    paste_letterboxed(canvas, wrist_frame, image_rects[1])
    paste_letterboxed(canvas, generated_frames[gt_slot], image_rects[2])
    paste_letterboxed(canvas, generated_frames[matched_slot], image_rects[3])
    labels = [
        "Current high view",
        "Current wrist view",
        f"Time-label slot {gt_slot}",
        f"Generated argmax slot {matched_slot}",
    ]
    for rect, label in zip(image_rects, labels):
        put_text(canvas, label, (rect[0], rect[1] - 7), scale=0.43, thickness=2)

    info_x = image_rects[-1][0] + image_rects[-1][2] + 26
    info_width = width - info_x - margin
    model_color = MODEL_COLORS.get(model, (70, 70, 210))
    error = abs(float(prediction[query_index] - target[query_index]))
    delta = (
        float(prediction[query_index] - prediction[query_index - 1])
        if query_index > 0
        else 0.0
    )
    probabilities = alignment[query_index]
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))))
    top_slots = np.argsort(probabilities)[-5:][::-1]
    top_text = "  ".join(f"{slot}:{probabilities[slot]:.3f}" for slot in top_slots)
    lines = [
        f"Model: {model}",
        f"Query: {query_index + 1}/{target.size}   source frame: {frame_index}",
        f"GT progress: {target[query_index]:.4f}",
        f"Pred progress: {prediction[query_index]:.4f}",
        f"Absolute error: {error:.4f}",
        f"Frame-to-frame delta: {delta:+.4f}",
        f"Time-label slot: {gt_slot_float:.2f}",
        f"Argmax slot: {matched_slot}   p={probabilities[matched_slot]:.3f}",
        f"Alignment entropy: {entropy:.3f}",
        f"Top slots: {top_text}",
    ]
    line_spacing = 23
    for index, line in enumerate(lines):
        put_text(
            canvas,
            line,
            (info_x, top_y + 20 + index * line_spacing),
            scale=0.48 if index else 0.65,
            color=model_color if index == 0 else (35, 35, 35),
            thickness=2 if index == 0 else 1,
        )
    alerts = []
    if error >= 0.10:
        alerts.append("LARGE ABSOLUTE ERROR")
    if abs(delta) >= 0.10:
        alerts.append("LARGE PROGRESS JUMP")
    if delta <= -0.05:
        alerts.append("BACKWARD PROGRESS")
    if alerts:
        alert_y = top_y + 20 + len(lines) * line_spacing + 4
        cv2.rectangle(
            canvas,
            (info_x, alert_y - 21),
            (width - margin, alert_y + 8),
            (226, 232, 255),
            -1,
        )
        put_text(
            canvas,
            " | ".join(alerts),
            (info_x + 6, alert_y),
            scale=0.49,
            color=(25, 25, 210),
            thickness=2,
        )

    distribution_rect = (info_x, top_y + top_height - 105, info_width, 100)
    draw_distribution(
        canvas,
        distribution_rect,
        probabilities,
        gt_slot_float,
        matched_slot,
        model_color,
    )

    chart_y = top_y + top_height + 38
    chart_height = height - chart_y - 28
    chart_gap = 42
    chart_width = (width - 2 * margin - chart_gap) // 2
    draw_progress_chart(
        canvas,
        (margin, chart_y, chart_width, chart_height),
        target,
        prediction,
        query_index,
        model_color,
    )
    draw_alignment_chart(
        canvas,
        (margin + chart_width + chart_gap, chart_y, chart_width, chart_height),
        heatmap,
        target,
        argmax_slots,
        query_index,
    )
    return canvas


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def event_summary(
    episode_name: str,
    task_text: str,
    model: str,
    frame_indices: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    alignment: np.ndarray,
) -> Dict[str, Any]:
    errors = np.abs(prediction - target)
    deltas = np.diff(prediction, prepend=prediction[0])
    argmax_slots = alignment.argmax(axis=1)
    top_error_indices = np.argsort(errors)[-10:][::-1]
    top_jump_indices = np.argsort(np.abs(deltas))[-10:][::-1]

    def row(index: int) -> Dict[str, Any]:
        return {
            "query_index": int(index),
            "frame_index": int(frame_indices[index]),
            "target": float(target[index]),
            "prediction": float(prediction[index]),
            "absolute_error": float(errors[index]),
            "delta": float(deltas[index]),
            "time_label_slot": float(target[index] * (alignment.shape[1] - 1)),
            "argmax_slot": int(argmax_slots[index]),
            "argmax_probability": float(alignment[index, argmax_slots[index]]),
        }

    return {
        "episode_name": episode_name,
        "task_text": task_text,
        "model": model,
        "queries": int(target.size),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "max_absolute_error": float(errors.max()),
        "max_absolute_jump": float(np.abs(deltas).max()),
        "top_error_frames": [row(int(index)) for index in top_error_indices],
        "top_jump_frames": [row(int(index)) for index in top_jump_indices],
    }


def render_model_video(
    *,
    args: argparse.Namespace,
    model: str,
    episode_name: str,
    task_text: str,
    output: Mapping[str, Any],
    high_paths: Sequence[Path],
    wrist_paths: Sequence[Path],
    generated_frames: np.ndarray,
) -> Dict[str, Any]:
    safe_episode = episode_name.rsplit("/", 1)[-1]
    video_path = args.output_dir / "videos" / model / f"{safe_episode}.mp4"
    poster_path = args.output_dir / "posters" / model / f"{safe_episode}_worst.png"
    events_path = args.output_dir / "events" / model / f"{safe_episode}.json"

    frame_indices = torch.as_tensor(output["frame_indices"]).long().numpy()
    target = torch.as_tensor(output["target"]).float().numpy()
    prediction = torch.as_tensor(output["prediction"]).float().numpy()
    alignment = torch.as_tensor(output["alignment_probabilities"]).float().numpy()
    if frame_indices.ndim != 1 or target.ndim != 1 or prediction.ndim != 1:
        raise ValueError(f"Progress outputs must be one-dimensional for {model}/{episode_name}")
    if alignment.ndim != 2:
        raise ValueError(
            f"alignment_probabilities must be [queries,slots] for {model}/{episode_name}"
        )
    if not (
        frame_indices.size == target.size == prediction.size == alignment.shape[0]
    ):
        raise ValueError(f"Progress tensor lengths differ for {model}/{episode_name}")
    if target.size == 0:
        raise ValueError(f"Progress outputs are empty for {model}/{episode_name}")
    if alignment.shape[1] != generated_frames.shape[0]:
        raise ValueError(
            f"Alignment has {alignment.shape[1]} slots but generated trajectory has "
            f"{generated_frames.shape[0]} frames for {model}/{episode_name}"
        )
    if not all(np.isfinite(value).all() for value in (target, prediction, alignment)):
        raise ValueError(f"Progress outputs contain NaN or inf for {model}/{episode_name}")
    if (target < 0.0).any() or (target > 1.0).any():
        raise ValueError(f"Progress targets must be in [0,1] for {model}/{episode_name}")
    if (alignment < -1e-6).any() or not np.allclose(
        alignment.sum(axis=1), 1.0, atol=5e-3
    ):
        raise ValueError(
            f"Alignment rows must be non-negative probability distributions for "
            f"{model}/{episode_name}"
        )
    if (
        frame_indices.min() < 0
        or frame_indices.max() >= len(high_paths)
        or len(high_paths) != len(wrist_paths)
    ):
        raise ValueError(f"RGB frame count differs for {episode_name}")

    summary = event_summary(
        episode_name,
        task_text,
        model,
        frame_indices,
        target,
        prediction,
        alignment,
    )
    summary["video_path"] = str(video_path)
    summary["poster_path"] = str(poster_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.skip_existing and video_path.is_file() and poster_path.is_file():
        print(f"skip existing {model}/{safe_episode}", flush=True)
        return summary

    heatmap = alignment_heatmap(alignment)
    argmax_slots = alignment.argmax(axis=1)
    worst_index = int(np.abs(prediction - target).argmax())
    writer = FfmpegWriter(video_path, args.width, args.height, args.fps, args.crf)
    worst_frame: np.ndarray | None = None
    try:
        for query_index, frame_index in enumerate(frame_indices.tolist()):
            frame = render_frame(
                width=args.width,
                height=args.height,
                model=model,
                task_text=task_text,
                episode_name=episode_name,
                query_index=query_index,
                frame_index=frame_index,
                high_frame=read_rgb(high_paths[frame_index]),
                wrist_frame=read_rgb(wrist_paths[frame_index]),
                generated_frames=generated_frames,
                target=target,
                prediction=prediction,
                alignment=alignment,
                heatmap=heatmap,
                argmax_slots=argmax_slots,
            )
            writer.write(frame)
            if query_index == worst_index:
                worst_frame = frame.copy()
    finally:
        writer.close()
    if worst_frame is None:
        raise RuntimeError(f"No poster frame produced for {model}/{episode_name}")
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(poster_path), worst_frame):
        raise RuntimeError(f"Failed to write poster: {poster_path}")
    print(f"rendered {model}/{safe_episode}: {video_path}", flush=True)
    return summary


def write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    new_rows = list(rows)
    if not new_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_name",
        "task_text",
        "model",
        "queries",
        "mae",
        "rmse",
        "max_absolute_error",
        "max_absolute_jump",
        "video_path",
        "poster_path",
    ]
    merged: Dict[tuple[str, str], Mapping[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                merged[(str(row["episode_name"]), str(row["model"]))] = row
    for row in new_rows:
        merged[(str(row["episode_name"]), str(row["model"]))] = row
    ordered_rows = [merged[key] for key in sorted(merged)]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_rows)


def main() -> None:
    args = parse_args()
    for path in (
        args.progress_eval_dir,
        args.cache_dir,
        args.images_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (args.benchmark_json, args.vae_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    episode_paths, task_texts = load_benchmark(args.benchmark_json)
    indices = selected_episode_indices(args, len(episode_paths))
    outputs = load_progress_outputs(args.progress_eval_dir, args.models)
    cache_manifest = load_cache_manifest(args.cache_dir)
    repo_root = Path(__file__).resolve().parent.parent
    decoder = TrajectoryDecoder(repo_root, args.vae_path, args.device)
    rows: List[Mapping[str, Any]] = []

    for episode_index in indices:
        relative_path = episode_paths[episode_index]
        task_text = task_texts[episode_index]
        episode_name = f"robo_dopamine_bench/{relative_path}"
        if episode_name not in cache_manifest:
            raise KeyError(f"No cache entry for {episode_name}")
        cache = torch_load(cache_manifest[episode_name])
        generated_frames = decoder.decode(torch.as_tensor(cache["trajectory_latent"]))
        if generated_frames.shape[0] != 53:
            raise ValueError(
                f"Expected 53 generated frames for {episode_name}, got {generated_frames.shape[0]}"
            )
        episode_root = args.images_root / relative_path
        high_paths = list_frames(episode_root / "cam_high")
        wrist_paths = list_frames(episode_root / "cam_left_wrist")
        for model in args.models:
            output = outputs[model].get(episode_name)
            if output is None:
                raise KeyError(f"No Progress output for {model}/{episode_name}")
            rows.append(
                render_model_video(
                    args=args,
                    model=model,
                    episode_name=episode_name,
                    task_text=task_text,
                    output=output,
                    high_paths=high_paths,
                    wrist_paths=wrist_paths,
                    generated_frames=generated_frames,
                )
            )
    write_manifest(args.output_dir / "manifest.csv", rows)
    print(f"complete: {len(rows)} videos under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
