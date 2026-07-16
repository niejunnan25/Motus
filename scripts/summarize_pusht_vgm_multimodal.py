#!/usr/bin/env python3
"""Aggregate PushT VGM seed/solver ablations into path-level evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pusht_trajectory_metrics import (
    generated_approach_mode,
    label_entropy,
    pairwise_path_rms,
)

# The primary metric requires both rendered entities in at least 16/17 frames.
# This post-hoc secondary diagnostic asks a narrower task question: did the T
# block reach its goal while the pusher remained observable for most of the path?
TASK_PUSHER_MIN_DETECTION_RATE = 12.0 / 17.0
TASK_BLOCK_MIN_DETECTION_RATE = 16.0 / 17.0


def read_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def condition_groups(step_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    rows = read_json(step_dir / "sample_metrics.json")
    result: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(int(row["condition_number"]), []).append(row)
    return result


def seed_dir(step_dir: Path, row: Dict[str, Any]) -> Path:
    pattern = f"condition_{int(row['condition_number']):02d}_*"
    condition_dir = next(step_dir.glob(pattern))
    return condition_dir / f"seed_{int(row['seed']):04d}"


def task_validity(
    row: Dict[str, Any],
    tracks: Any,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    pred_goal = tracks["pred_block_xy"][-1]
    gt_goal = tracks["gt_block_xy"][-1]
    block_goal_error = (
        float(np.linalg.norm(pred_goal - gt_goal))
        if np.isfinite(pred_goal).all() and np.isfinite(gt_goal).all()
        else float("inf")
    )
    checks = {
        "pusher_observable": (
            float(row["pusher_detection_rate"]) >= TASK_PUSHER_MIN_DETECTION_RATE
            and thresholds["pusher_area_min"]
            <= float(row["pusher_area_median"])
            <= thresholds["pusher_area_max"]
        ),
        "block_observable": (
            float(row["block_detection_rate"]) >= TASK_BLOCK_MIN_DETECTION_RATE
            and thresholds["block_area_min"]
            <= float(row["block_area_median"])
            <= thresholds["block_area_max"]
        ),
        "temporally_smooth": bool(row["check_temporally_smooth"]),
        "nontrivial_block_motion": bool(row["check_nontrivial_block_motion"]),
        "motion_contact_coupled": bool(row["check_motion_contact_coupled"]),
        "appearance_plausible": bool(row["check_appearance_plausible"]),
        "start_consistent": (
            float(row["start_entity_error"]) <= thresholds["endpoint_entity_error_max"]
        ),
        "block_goal_consistent": (
            block_goal_error <= thresholds["endpoint_entity_error_max"]
        ),
    }
    return {
        "valid": bool(all(checks.values())),
        "block_goal_error": block_goal_error,
        "checks": checks,
    }


def summarize_condition(step_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    thresholds = read_json(step_dir / "calibration.json")["thresholds"]
    samples = []
    for row in rows:
        tracks = np.load(seed_dir(step_dir, row) / "trajectory_tracks.npz")
        samples.append((row, tracks, task_validity(row, tracks, thresholds)))

    strict_samples = [sample for sample in samples if bool(sample[0]["valid"])]
    task_samples = [sample for sample in samples if bool(sample[2]["valid"])]
    def mode(tracks: Any) -> str:
        result = generated_approach_mode(
            tracks["pred_pusher_xy"],
            tracks["pred_block_xy"],
            tracks["pred_surface_distance"],
            contact_distance_max=float(thresholds["moving_contact_distance_max"]),
        )
        return str(result["sector"])

    strict_sectors = [mode(tracks) for _, tracks, _ in strict_samples]
    task_sectors = [mode(tracks) for _, tracks, _ in task_samples]
    return {
        "condition_number": int(rows[0]["condition_number"]),
        "episode_name": rows[0]["episode_name"],
        "num_inference_steps": int(rows[0]["num_inference_steps"]),
        "samples": len(rows),
        # Keep the original names as aliases for the primary strict metric.
        "valid_samples": len(strict_samples),
        "valid_rate": float(len(strict_samples) / len(rows)),
        "valid_mode_count": len(set(strict_sectors)),
        "valid_mode_entropy": label_entropy(strict_sectors),
        "pusher_pairwise_rms": pairwise_path_rms(
            [tracks["pred_pusher_xy"] for _, tracks, _ in strict_samples]
        ),
        "block_pairwise_rms": pairwise_path_rms(
            [tracks["pred_block_xy"] for _, tracks, _ in strict_samples]
        ),
        "strict_valid_samples": len(strict_samples),
        "strict_valid_rate": float(len(strict_samples) / len(rows)),
        "task_valid_samples": len(task_samples),
        "task_valid_rate": float(len(task_samples) / len(rows)),
        "task_valid_mode_count": len(set(task_sectors)),
        "task_valid_mode_entropy": label_entropy(task_sectors),
        "task_pusher_pairwise_rms": pairwise_path_rms(
            [tracks["pred_pusher_xy"] for _, tracks, _ in task_samples]
        ),
        "task_block_pairwise_rms": pairwise_path_rms(
            [tracks["pred_block_xy"] for _, tracks, _ in task_samples]
        ),
        "mean_block_goal_error": float(
            np.mean([diagnostic["block_goal_error"] for _, _, diagnostic in samples])
        ),
        "mean_pusher_detection_rate": float(
            np.mean([float(row["pusher_detection_rate"]) for row in rows])
        ),
        "mean_block_detection_rate": float(
            np.mean([float(row["block_detection_rate"]) for row in rows])
        ),
        "mean_start_entity_error": float(np.mean([float(row["start_entity_error"]) for row in rows])),
        "mean_goal_entity_error": float(np.mean([float(row["goal_entity_error"]) for row in rows])),
        "mean_pixel_mse_to_single_gt": float(np.mean([float(row["pixel_mse_to_single_gt"]) for row in rows])),
    }


def aggregate_steps(condition_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def finite_mean(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.mean()) if len(array) else float("nan")

    result = []
    for steps in sorted({int(row["num_inference_steps"]) for row in condition_rows}):
        current = [row for row in condition_rows if int(row["num_inference_steps"]) == steps]
        result.append(
            {
                "num_inference_steps": steps,
                "conditions": len(current),
                "valid_rate": float(np.mean([row["valid_rate"] for row in current])),
                "strict_valid_rate": float(
                    np.mean([row["strict_valid_rate"] for row in current])
                ),
                "valid_mode_count": float(np.mean([row["valid_mode_count"] for row in current])),
                "valid_mode_entropy": float(np.mean([row["valid_mode_entropy"] for row in current])),
                "pusher_pairwise_rms": finite_mean([row["pusher_pairwise_rms"] for row in current]),
                "block_pairwise_rms": finite_mean([row["block_pairwise_rms"] for row in current]),
                "task_valid_rate": float(
                    np.mean([row["task_valid_rate"] for row in current])
                ),
                "task_valid_mode_count": float(
                    np.mean([row["task_valid_mode_count"] for row in current])
                ),
                "task_valid_mode_entropy": float(
                    np.mean([row["task_valid_mode_entropy"] for row in current])
                ),
                "task_pusher_pairwise_rms": finite_mean(
                    [row["task_pusher_pairwise_rms"] for row in current]
                ),
                "task_block_pairwise_rms": finite_mean(
                    [row["task_block_pairwise_rms"] for row in current]
                ),
                "mean_block_goal_error": float(
                    np.mean([row["mean_block_goal_error"] for row in current])
                ),
                "mean_pusher_detection_rate": float(
                    np.mean([row["mean_pusher_detection_rate"] for row in current])
                ),
                "mean_block_detection_rate": float(
                    np.mean([row["mean_block_detection_rate"] for row in current])
                ),
                "mean_start_entity_error": float(np.mean([row["mean_start_entity_error"] for row in current])),
                "mean_goal_entity_error": float(np.mean([row["mean_goal_entity_error"] for row in current])),
                "mean_pixel_mse_to_single_gt": float(
                    np.mean([row["mean_pixel_mse_to_single_gt"] for row in current])
                ),
            }
        )
    return result


def plot_step_metrics(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["num_inference_steps"])
    steps = [row["num_inference_steps"] for row in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(
        steps,
        [row["strict_valid_rate"] for row in ordered],
        marker="o",
        linewidth=2.2,
        label="Strict complete-path validity",
    )
    axes[0, 0].plot(
        steps,
        [row["task_valid_rate"] for row in ordered],
        marker="o",
        linewidth=2.2,
        label="Post-hoc task validity",
    )
    axes[0, 0].set_title("Validity tiers", fontsize=13)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend()

    panels = (
        (axes[0, 1], "task_valid_mode_count", "Task-valid approach modes / condition", (0.0, 4.1)),
        (axes[1, 0], "task_pusher_pairwise_rms", "Task-valid pusher path diversity", None),
        (axes[1, 1], "mean_pusher_detection_rate", "Mean pusher detection across 17 frames", (0.0, 1.05)),
    )
    for axis, key, title, ylim in panels:
        axis.plot(steps, [row[key] for row in ordered], marker="o", linewidth=2.2)
        axis.set_title(title, fontsize=13)
        if ylim is not None:
            axis.set_ylim(*ylim)
        for x, row in zip(steps, ordered):
            if np.isfinite(row[key]):
                axis.annotate(
                    f"{row[key]:.3f}",
                    (x, row[key]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                )
    for axis in axes.flat:
        axis.set_xlabel("Denoising steps")
        axis.grid(alpha=0.25)
    fig.suptitle("PushT VGM solver-step ablation: strict and post-hoc task evidence", fontsize=17)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pareto(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for row in rows:
        if not np.isfinite(row["task_pusher_pairwise_rms"]):
            continue
        axis.scatter(row["task_pusher_pairwise_rms"], row["task_valid_rate"], s=90)
        x_offset = 8
        y_offset = 6
        if row["num_inference_steps"] == 50:
            y_offset = -18
        axis.annotate(
            f"{row['num_inference_steps']} steps",
            (row["task_pusher_pairwise_rms"], row["task_valid_rate"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
        )
    axis.set_xlabel("Task-valid pusher path diversity (normalized pairwise RMS)")
    axis.set_ylabel("Post-hoc task-valid rate")
    axis.set_ylim(-0.03, 1.05)
    axis.set_title("Post-hoc task-valid multimodality diagnostic")
    axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_condition_paths(step_dirs: Sequence[Path], condition_number: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(step_dirs), figsize=(5 * len(step_dirs), 5), constrained_layout=True)
    if len(step_dirs) == 1:
        axes = [axes]
    for axis, step_dir in zip(axes, step_dirs):
        rows = condition_groups(step_dir)[condition_number]
        thresholds = read_json(step_dir / "calibration.json")["thresholds"]
        first_tracks = np.load(seed_dir(step_dir, rows[0]) / "trajectory_tracks.npz")
        first_frame = first_tracks["gt_frames"][0]
        axis.imshow(first_frame, extent=(0, 1, 1, 0))
        gt = first_tracks["gt_pusher_xy"]
        axis.plot(gt[:, 0], gt[:, 1], color="black", linewidth=3, label="GT pusher")
        for row in rows:
            tracks = np.load(seed_dir(step_dir, row) / "trajectory_tracks.npz")
            path = tracks["pred_pusher_xy"]
            task_valid = task_validity(row, tracks, thresholds)["valid"]
            if row["valid"]:
                style, alpha = "-", 0.95
            elif task_valid:
                style, alpha = "--", 0.85
            else:
                style, alpha = ":", 0.35
            axis.plot(path[:, 0], path[:, 1], style, linewidth=1.7, alpha=alpha, label=f"seed {row['seed']}")
            axis.scatter(path[0, 0], path[0, 1], s=18, color="green")
            axis.scatter(path[-1, 0], path[-1, 1], s=22, color="red")
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)
        axis.set_title(f"{rows[0]['num_inference_steps']} steps")
        axis.set_aspect("equal")
    axes[0].set_ylabel("Pusher paths; solid=strict, dashed=post-hoc task, dotted=invalid")
    fig.suptitle(f"Condition {condition_number:02d}: same endpoints, same seeds", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_full_17f_overview(step_dir: Path, condition_number: int, output_path: Path) -> None:
    rows = sorted(condition_groups(step_dir)[condition_number], key=lambda row: int(row["seed"]))
    first_tracks = np.load(seed_dir(step_dir, rows[0]) / "trajectory_tracks.npz")
    frame_count = int(first_tracks["gt_frames"].shape[0])
    tile = 112
    row_label = 390
    header = 58
    frame_label = 22
    rendered_rows = 1 + len(rows)
    canvas = Image.new(
        "RGB",
        (row_label + frame_count * tile, header + rendered_rows * (tile + frame_label)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    label_font = load_font(20)
    frame_font = load_font(15)
    steps = int(rows[0]["num_inference_steps"])
    draw.text(
        (12, 11),
        f"Condition {condition_number:02d} | {rows[0]['episode_name']} | {steps} denoising steps",
        fill=(16, 24, 32),
        font=title_font,
    )

    thresholds = read_json(step_dir / "calibration.json")["thresholds"]
    videos = [("Ground truth", first_tracks["gt_frames"])]
    for row in rows:
        tracks = np.load(seed_dir(step_dir, row) / "trajectory_tracks.npz")
        task_valid = task_validity(row, tracks, thresholds)["valid"]
        videos.append(
            (
                f"Seed {int(row['seed'])} | strict={bool(row['valid'])} | task={task_valid}",
                tracks["pred_frames"],
            )
        )

    for row_idx, (label, frames) in enumerate(videos):
        y = header + row_idx * (tile + frame_label)
        draw.text((12, y + tile // 2), label, fill=(16, 24, 32), font=label_font)
        for frame_idx in range(frame_count):
            x = row_label + frame_idx * tile
            frame = Image.fromarray(frames[frame_idx]).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x, y + frame_label))
            draw.text((x + 4, y + 2), f"{frame_idx:02d}", fill=(24, 32, 40), font=frame_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="directory containing steps_* evaluator outputs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--steps",
        default="1,4,10,20,50",
        help="comma-separated solver steps to include; formal protocol is capped at 50",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_steps = {int(value.strip()) for value in args.steps.split(",") if value.strip()}
    if not selected_steps or max(selected_steps) > 50:
        raise ValueError(f"formal solver steps must be non-empty and <= 50, got {sorted(selected_steps)}")
    all_step_dirs = sorted(
        path for path in input_root.glob("steps_*") if (path / "sample_metrics.json").exists()
    )
    step_dirs = [path for path in all_step_dirs if int(path.name.split("_")[-1]) in selected_steps]
    if not step_dirs:
        raise FileNotFoundError(f"No complete steps_* runs under {input_root}")
    excluded_steps = sorted(
        int(path.name.split("_")[-1]) for path in all_step_dirs if path not in step_dirs
    )

    condition_rows = []
    for step_dir in step_dirs:
        for rows in condition_groups(step_dir).values():
            condition_rows.append(summarize_condition(step_dir, rows))
    step_rows = aggregate_steps(condition_rows)

    with (output_dir / "condition_summary.json").open("w") as file:
        json.dump(condition_rows, file, indent=2)
    with (output_dir / "step_summary.json").open("w") as file:
        json.dump(step_rows, file, indent=2)
    write_csv(output_dir / "condition_summary.csv", condition_rows)
    write_csv(output_dir / "step_summary.csv", step_rows)
    with (output_dir / "summary_manifest.json").open("w") as file:
        json.dump(
            {
                "included_steps": sorted(selected_steps),
                "excluded_available_steps": excluded_steps,
                "maximum_formal_step": max(selected_steps),
            },
            file,
            indent=2,
        )
    plot_step_metrics(step_rows, output_dir / "solver_step_metrics.png")
    plot_pareto(step_rows, output_dir / "validity_diversity_pareto.png")

    condition_ids = sorted(condition_groups(step_dirs[0]))
    for condition_number in condition_ids:
        plot_condition_paths(
            step_dirs,
            condition_number,
            output_dir / f"condition_{condition_number:02d}_all_steps_paths.png",
        )
        for step_dir in step_dirs:
            step = int(condition_groups(step_dir)[condition_number][0]["num_inference_steps"])
            save_full_17f_overview(
                step_dir,
                condition_number,
                output_dir / "full_17f_overviews" / f"steps_{step:03d}_condition_{condition_number:02d}.png",
            )


if __name__ == "__main__":
    main()
