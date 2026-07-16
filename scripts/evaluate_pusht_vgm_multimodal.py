#!/usr/bin/env python3
"""Generate repeated PushT bridges and measure path-level validity.

This evaluator intentionally separates diversity from validity. Different pixels
or different random seeds are not counted as useful modes unless the rendered
pusher and T block remain trackable, move smoothly, and reach the conditioned
goal neighborhood.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.eval_vgm_bridge_stage1 import build_dataset, build_model, tensor_to_uint8
from scripts.pusht_trajectory_metrics import generated_approach_mode


LOGGER = logging.getLogger(__name__)


def parse_int_list(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return result


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def color_mask(frame: np.ndarray, prototypes: Sequence[Sequence[float]], threshold: float) -> np.ndarray:
    rgb = frame.astype(np.float32) / 255.0
    palette = np.asarray(prototypes, dtype=np.float32) / 255.0
    distance = ((rgb[..., None, :] - palette[None, None, ...]) ** 2).sum(axis=-1)
    return distance.min(axis=-1) <= threshold**2


def mask_centroid(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    count = float(mask.sum())
    if count < 1.0:
        return np.asarray([np.nan, np.nan], dtype=np.float32), 0.0
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    xy = np.asarray([(xx * mask).sum() / count, (yy * mask).sum() / count], dtype=np.float32)
    xy /= np.asarray([mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)
    return xy, count / float(mask.size)


def extract_tracks(frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract color-stable PushT entities from uint8 RGB frames."""
    pusher_xy: List[np.ndarray] = []
    block_xy: List[np.ndarray] = []
    pusher_area: List[float] = []
    block_area: List[float] = []
    surface_distance: List[float] = []
    for frame in frames:
        pusher_mask = color_mask(
            frame,
            prototypes=((65, 105, 225), (78, 126, 255), (63, 107, 225)),
            threshold=0.27,
        )
        block_mask = color_mask(
            frame,
            prototypes=((141, 161, 182), (143, 163, 184), (119, 136, 153)),
            threshold=0.19,
        )
        xy, area = mask_centroid(pusher_mask)
        pusher_xy.append(xy)
        pusher_area.append(area)
        xy, area = mask_centroid(block_mask)
        block_xy.append(xy)
        block_area.append(area)
        if pusher_mask.any() and block_mask.any():
            distance_pixels = distance_transform_edt(~block_mask)[pusher_mask]
            image_diagonal = math.hypot(frame.shape[0] - 1, frame.shape[1] - 1)
            surface_distance.append(float(distance_pixels.min() / image_diagonal))
        else:
            surface_distance.append(float("nan"))
    return {
        "pusher_xy": np.stack(pusher_xy),
        "block_xy": np.stack(block_xy),
        "pusher_area": np.asarray(pusher_area, dtype=np.float32),
        "block_area": np.asarray(block_area, dtype=np.float32),
        "surface_distance": np.asarray(surface_distance, dtype=np.float32),
    }


def finite_steps(xy: np.ndarray) -> np.ndarray:
    delta = np.diff(xy, axis=0)
    value = np.linalg.norm(delta, axis=-1)
    return value[np.isfinite(value)]


def trajectory_features(tracks: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    pusher_xy = tracks["pusher_xy"]
    block_xy = tracks["block_xy"]
    pusher_step = finite_steps(pusher_xy)
    block_step = finite_steps(block_xy)
    block_step_all = np.linalg.norm(np.diff(block_xy, axis=0), axis=-1)
    finite_block_step = block_step_all[np.isfinite(block_step_all)]
    if len(finite_block_step):
        motion_threshold = max(0.0025, float(np.quantile(finite_block_step, 0.50)))
        moving = np.concatenate([[False], block_step_all >= motion_threshold])
        moving_surface = tracks["surface_distance"][moving & np.isfinite(tracks["surface_distance"])]
    else:
        moving_surface = np.asarray([], dtype=np.float32)
    distance = np.linalg.norm(pusher_xy - block_xy, axis=-1)
    finite_distance = np.flatnonzero(np.isfinite(distance))
    closest_distance = (
        float(distance[finite_distance].min()) if len(finite_distance) else float("nan")
    )
    mode = generated_approach_mode(
        pusher_xy,
        block_xy,
        tracks["surface_distance"],
    )

    return {
        "pusher_detection_rate": float(np.isfinite(pusher_xy).all(axis=-1).mean()),
        "block_detection_rate": float(np.isfinite(block_xy).all(axis=-1).mean()),
        "pusher_area_median": float(np.nanmedian(tracks["pusher_area"])),
        "block_area_median": float(np.nanmedian(tracks["block_area"])),
        "pusher_max_step": float(pusher_step.max()) if len(pusher_step) else float("nan"),
        "block_max_step": float(block_step.max()) if len(block_step) else float("nan"),
        "pusher_path_length": float(pusher_step.sum()),
        "block_path_length": float(block_step.sum()),
        "moving_contact_distance_q90": (
            float(np.quantile(moving_surface, 0.90)) if len(moving_surface) else float("nan")
        ),
        "closest_pusher_block_distance": closest_distance,
        "closest_frame": int(mode["index"]),
        "approach_sector": str(mode["sector"]),
        "approach_sector_index": int(mode["sector_index"]),
        "approach_from_contact": bool(mode["from_contact"]),
    }


def percentile_thresholds(real_features: Sequence[Dict[str, float | str]]) -> Dict[str, float]:
    def values(key: str) -> np.ndarray:
        result = np.asarray([float(row[key]) for row in real_features], dtype=np.float64)
        return result[np.isfinite(result)]

    return {
        "pusher_area_min": float(np.quantile(values("pusher_area_median"), 0.01) * 0.35),
        "pusher_area_max": float(np.quantile(values("pusher_area_median"), 0.99) * 3.0),
        "block_area_min": float(np.quantile(values("block_area_median"), 0.01) * 0.35),
        "block_area_max": float(np.quantile(values("block_area_median"), 0.99) * 3.0),
        "pusher_max_step_max": float(np.quantile(values("pusher_max_step"), 0.99) * 1.75),
        "block_max_step_max": float(np.quantile(values("block_max_step"), 0.99) * 1.75),
        "block_path_length_min": float(np.quantile(values("block_path_length"), 0.05) * 0.20),
        # Real rendered masks normally touch exactly. Keep a small image-space
        # tolerance for VAE blur instead of turning the calibrated zero into an
        # unrealistically exact generated-pixel requirement.
        "moving_contact_distance_max": max(
            0.025,
            float(np.quantile(values("moving_contact_distance_q90"), 0.99) * 1.75),
        ),
        "endpoint_entity_error_max": 0.12,
        "pixel_mse_max": 0.08,
    }


def validity_checks(
    features: Dict[str, float | str],
    pred_tracks: Dict[str, np.ndarray],
    gt_tracks: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    pixel_mse_to_single_gt: float,
) -> Dict[str, Any]:
    start_endpoint = np.concatenate(
        [
            pred_tracks["pusher_xy"][0] - gt_tracks["pusher_xy"][0],
            pred_tracks["block_xy"][0] - gt_tracks["block_xy"][0],
        ]
    )
    goal_endpoint = np.concatenate(
        [
            pred_tracks["pusher_xy"][-1] - gt_tracks["pusher_xy"][-1],
            pred_tracks["block_xy"][-1] - gt_tracks["block_xy"][-1],
        ]
    )
    start_entity_error = (
        float(np.linalg.norm(start_endpoint)) if np.isfinite(start_endpoint).all() else float("inf")
    )
    goal_entity_error = (
        float(np.linalg.norm(goal_endpoint)) if np.isfinite(goal_endpoint).all() else float("inf")
    )
    checks = {
        "entities_trackable": (
            float(features["pusher_detection_rate"]) >= 0.94
            and float(features["block_detection_rate"]) >= 0.94
            and float(features["pusher_area_median"]) >= thresholds["pusher_area_min"]
            and float(features["pusher_area_median"]) <= thresholds["pusher_area_max"]
            and float(features["block_area_median"]) >= thresholds["block_area_min"]
            and float(features["block_area_median"]) <= thresholds["block_area_max"]
        ),
        "temporally_smooth": (
            float(features["pusher_max_step"]) <= thresholds["pusher_max_step_max"]
            and float(features["block_max_step"]) <= thresholds["block_max_step_max"]
        ),
        "nontrivial_block_motion": (
            float(features["block_path_length"]) >= thresholds["block_path_length_min"]
        ),
        "motion_contact_coupled": (
            float(features["moving_contact_distance_q90"]) <= thresholds["moving_contact_distance_max"]
        ),
        # PushT has a static background. This broad threshold rejects globally
        # broken/noisy renders while allowing moving objects to follow another
        # valid path than the single recorded demonstration.
        "appearance_plausible": pixel_mse_to_single_gt <= thresholds["pixel_mse_max"],
        "endpoints_consistent": (
            start_entity_error <= thresholds["endpoint_entity_error_max"]
            and goal_entity_error <= thresholds["endpoint_entity_error_max"]
        ),
    }
    return {
        "start_entity_error": start_entity_error,
        "goal_entity_error": goal_entity_error,
        "checks": checks,
        "valid": bool(all(checks.values())),
    }


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    return np.stack([tensor_to_uint8(frame) for frame in video])


def save_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def save_full_17f_sheet(path: Path, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    frame_count = min(len(gt), len(pred))
    tile = 160
    row_label = 190
    header = 54
    frame_label = 24
    canvas = Image.new("RGB", (row_label + frame_count * tile, header + 2 * (tile + frame_label)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    label_font = load_font(24)
    frame_font = load_font(17)
    draw.text((12, 10), title, fill=(16, 24, 32), font=title_font)
    for row, (name, frames) in enumerate((("Ground truth", gt), ("Generated", pred))):
        y = header + row * (tile + frame_label)
        draw.text((12, y + tile // 2 - 14), name, fill=(16, 24, 32), font=label_font)
        for frame_idx in range(frame_count):
            x = row_label + frame_idx * tile
            image = Image.fromarray(frames[frame_idx]).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + frame_label))
            draw.text((x + 5, y + 2), f"{frame_idx:02d}", fill=(24, 32, 40), font=frame_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def calibration_windows(dataset: Any) -> Iterable[Dict[str, Any]]:
    for episode_idx in range(len(dataset.episodes)):
        try:
            yield dataset.get_bridge_window(episode_idx)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("Skipping calibration episode %s: %s", episode_idx, exc)


def sample_to_gt_uint8(sample: Dict[str, Any]) -> np.ndarray:
    video = torch.cat([sample["first_frame"].unsqueeze(0), sample["video_frames"]], dim=0)
    return tensor_video_to_uint8(video)


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


@torch.no_grad()
def generate_one(model: Any, sample: Dict[str, Any], seed: int, steps: int) -> torch.Tensor:
    first = sample["first_frame"].unsqueeze(0).to(model.device, dtype=model.dtype)
    video = sample["video_frames"].unsqueeze(0).to(model.device, dtype=model.dtype)
    tail_n = int(model.config.tail_condition_frames)
    last = video[:, -1] if tail_n else None
    tail = video[:, -tail_n:] if tail_n else None
    generator = torch.Generator(device=model.device).manual_seed(seed)
    return model.sample_bridge(
        first_frame=first,
        last_frame=last,
        tail_frames=tail,
        num_inference_steps=steps,
        generator=generator,
    )[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--calibration_config",
        default="configs/vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k.yaml",
        help="train-split config used only to calibrate fixed visual-validity thresholds",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episode_indices", type=parse_int_list, default=parse_int_list("0,7,14,21"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4,5,6,7"))
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.config)
    dataset = build_dataset(config)
    calibration_dataset = build_dataset(OmegaConf.load(args.calibration_config))
    selected = sorted(set(args.episode_indices))
    if selected[-1] >= len(dataset.episodes):
        raise IndexError(f"episode index {selected[-1]} exceeds dataset size {len(dataset.episodes)}")

    LOGGER.info(
        "Calibrating validity thresholds from %s real training trajectories",
        len(calibration_dataset.episodes),
    )
    real_features = []
    for sample in calibration_windows(calibration_dataset):
        real_features.append(trajectory_features(extract_tracks(sample_to_gt_uint8(sample))))
    thresholds = percentile_thresholds(real_features)

    LOGGER.info("Loading model from %s", args.checkpoint)
    model = build_model(config, Path(args.checkpoint))
    model.eval()

    rows: List[Dict[str, Any]] = []
    window_manifest: List[Dict[str, Any]] = []
    for condition_number, episode_idx in enumerate(selected):
        sample = dataset.get_bridge_window(episode_idx)
        gt = sample_to_gt_uint8(sample)
        gt_tracks = extract_tracks(gt)
        episode_name = str(sample["episode_name"])
        window_manifest.append(
            {
                "condition_number": condition_number,
                "dataset_episode_index": episode_idx,
                "episode_name": episode_name,
                "frame_indices": sample["frame_indices"],
                "total_frames": sample["total_frames"],
                "video_path": sample["video_path"],
            }
        )
        condition_dir = output_dir / f"condition_{condition_number:02d}_{episode_name}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        save_video(condition_dir / "ground_truth_17f.mp4", gt, args.fps)

        for seed in args.seeds:
            LOGGER.info(
                "Generating condition=%s/%s episode=%s seed=%s steps=%s",
                condition_number + 1,
                len(selected),
                episode_name,
                seed,
                args.num_inference_steps,
            )
            pred = tensor_video_to_uint8(generate_one(model, sample, seed, args.num_inference_steps))
            pred_tracks = extract_tracks(pred)
            features = trajectory_features(pred_tracks)
            pixel_mse = float(np.mean((pred.astype(np.float32) / 255.0 - gt.astype(np.float32) / 255.0) ** 2))
            validity = validity_checks(features, pred_tracks, gt_tracks, thresholds, pixel_mse)
            row = {
                "condition_number": condition_number,
                "dataset_episode_index": episode_idx,
                "episode_name": episode_name,
                "seed": seed,
                "num_inference_steps": args.num_inference_steps,
                "pixel_mse_to_single_gt": pixel_mse,
                **features,
                "start_entity_error": validity["start_entity_error"],
                "goal_entity_error": validity["goal_entity_error"],
                "valid": validity["valid"],
                **{f"check_{key}": value for key, value in validity["checks"].items()},
            }
            rows.append(row)

            seed_dir = condition_dir / f"seed_{seed:04d}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            save_video(seed_dir / "generated_17f.mp4", pred, args.fps)
            save_full_17f_sheet(
                seed_dir / "gt_vs_generated_17f.png",
                gt,
                pred,
                title=f"{episode_name} | seed={seed} | steps={args.num_inference_steps} | valid={validity['valid']}",
            )
            np.savez_compressed(
                seed_dir / "trajectory_tracks.npz",
                pred_frames=pred,
                gt_frames=gt,
                pred_pusher_xy=pred_tracks["pusher_xy"],
                pred_block_xy=pred_tracks["block_xy"],
                pred_pusher_area=pred_tracks["pusher_area"],
                pred_block_area=pred_tracks["block_area"],
                pred_surface_distance=pred_tracks["surface_distance"],
                gt_pusher_xy=gt_tracks["pusher_xy"],
                gt_block_xy=gt_tracks["block_xy"],
                gt_pusher_area=gt_tracks["pusher_area"],
                gt_block_area=gt_tracks["block_area"],
                gt_surface_distance=gt_tracks["surface_distance"],
            )
            with (seed_dir / "metrics.json").open("w") as file:
                json.dump(json_value(row), file, indent=2)

    with (output_dir / "windows.json").open("w") as file:
        json.dump(json_value(window_manifest), file, indent=2)
    with (output_dir / "calibration.json").open("w") as file:
        json.dump(
            json_value({"thresholds": thresholds, "real_features": real_features}),
            file,
            indent=2,
        )
    with (output_dir / "sample_metrics.json").open("w") as file:
        json.dump(json_value(rows), file, indent=2)
    write_rows(output_dir / "sample_metrics.csv", rows)
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "calibration_config": args.calibration_config,
        "num_inference_steps": args.num_inference_steps,
        "conditions": len(selected),
        "seeds_per_condition": len(args.seeds),
        "valid_rate": float(np.mean([bool(row["valid"]) for row in rows])),
        "mean_start_entity_error": float(np.mean([float(row["start_entity_error"]) for row in rows])),
        "mean_goal_entity_error": float(np.mean([float(row["goal_entity_error"]) for row in rows])),
        "thresholds": thresholds,
    }
    with (output_dir / "run_summary.json").open("w") as file:
        json.dump(json_value(summary), file, indent=2)
    LOGGER.info("Finished: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
