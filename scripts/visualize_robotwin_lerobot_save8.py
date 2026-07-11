#!/usr/bin/env python3
"""Visualize Motus RoboTwin future-frame predictions on LeRobot parquet datasets.

This script reads the 50 `single_task_clean_*` RoboTwin LeRobot datasets directly
from parquet, reconstructs the three-camera Motus input frame, runs the RoboTwin
Motus checkpoint, and saves GT-vs-pred future-frame visualizations.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.utils.image_utils import resize_with_padding  # noqa: E402


LOGGER = logging.getLogger("robotwin_lerobot_visualize")

SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

IMAGE_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        default="/mnt/workspace/users/niejunnan/lerobot_datasets",
        help="Root containing single_task_clean_* LeRobot datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "eval_outputs" / "robotwin_lerobot_save8_visualization"),
        help="Directory for visualization outputs.",
    )
    parser.add_argument(
        "--policy_dir",
        default=str(REPO_ROOT / "inference" / "robotwin" / "Motus"),
        help="Directory containing deploy_policy_save8.py and utils/robotwin.yml.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/workspace/users/niejunnan/codebase/Motus/pretrained_models/Motus_robotwin2",
        help="Motus checkpoint directory containing mp_rank_00_model_states.pt.",
    )
    parser.add_argument(
        "--wan_path",
        default="/mnt/workspace/users/niejunnan/codebase/Motus/pretrained_models/Wan2.2-TI2V-5B",
        help="Wan2.2-TI2V-5B directory.",
    )
    parser.add_argument(
        "--vlm_path",
        default="/mnt/workspace/users/niejunnan/codebase/Motus/pretrained_models/Qwen3-VL-2B-Instruct",
        help="Qwen3-VL-2B-Instruct directory.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated task names or dataset directory names. "
            "Examples: adjust_bottle,single_task_clean_click_alarmclock. "
            "Default: all single_task_clean_* except single_task_clean_beat_block_hammer."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Total samples to render.")
    parser.add_argument("--samples_per_task", type=int, default=1, help="Candidate episodes per task.")
    parser.add_argument(
        "--window_strategy",
        choices=("middle", "first", "random"),
        default="middle",
        help="How to choose the condition frame within an episode.",
    )
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--fps", type=int, default=4, help="FPS for saved GT-vs-pred mp4 files.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override model inference steps. Default uses policy config, normally 10.",
    )
    parser.add_argument("--video_height", type=int, default=384)
    parser.add_argument("--video_width", type=int, default=320)
    parser.add_argument("--global_downsample_rate", type=int, default=3)
    parser.add_argument("--video_action_freq_ratio", type=int, default=2)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only test dataset reading/saving; uses GT frames as fake predictions.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def normalize_task_name(name: str) -> str:
    return name if name.startswith("single_task_clean_") else f"single_task_clean_{name}"


def iter_task_dirs(dataset_root: Path, task_filter: Optional[str]) -> List[Path]:
    all_dirs = sorted(
        p
        for p in dataset_root.glob("single_task_clean_*")
        if p.is_dir() and p.name != "single_task_clean_beat_block_hammer"
    )
    if not task_filter:
        return all_dirs

    wanted = {normalize_task_name(part.strip()) for part in task_filter.split(",") if part.strip()}
    selected = [p for p in all_dirs if p.name in wanted]
    missing = sorted(wanted - {p.name for p in selected})
    if missing:
        raise FileNotFoundError(f"Requested tasks not found under {dataset_root}: {missing}")
    return selected


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parquet_path_for_episode(task_dir: Path, episode_index: int) -> Path:
    info_path = task_dir / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    episode_chunk = episode_index // chunks_size
    return task_dir / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)


def choose_condition_index(
    total_frames: int,
    action_chunk_size: int,
    global_downsample_rate: int,
    strategy: str,
    rng: random.Random,
) -> int:
    physical_chunk_size = action_chunk_size * global_downsample_rate
    max_condition_idx = total_frames - physical_chunk_size - 1
    if max_condition_idx < 0:
        raise ValueError(
            f"Episode is too short: total_frames={total_frames}, required>{physical_chunk_size}"
        )
    if strategy == "first":
        return 0
    if strategy == "middle":
        return max_condition_idx // 2
    return rng.randint(0, max_condition_idx)


def calculate_indices(
    condition_idx: int,
    total_frames: int,
    num_video_frames: int,
    video_action_freq_ratio: int,
    global_downsample_rate: int,
) -> Tuple[List[int], List[int]]:
    action_chunk_size = num_video_frames * video_action_freq_ratio
    action_indices = [
        min(condition_idx + (idx + 1) * global_downsample_rate, total_frames - 1)
        for idx in range(action_chunk_size)
    ]
    video_indices: List[int] = []
    for idx in range(num_video_frames):
        action_step = (idx + 1) * video_action_freq_ratio - 1
        video_indices.append(action_indices[action_step])
    return video_indices, action_indices


def decode_lerobot_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None and value.get("path"):
            raise ValueError("Image dict has path but no bytes; external image files are not supported here")
    elif isinstance(value, (bytes, bytearray)):
        data = value
    else:
        raise TypeError(f"Unsupported LeRobot image value type: {type(value)!r}")

    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image)


def reconstruct_motus_rgb_frame(row: pd.Series, target_size: Tuple[int, int]) -> np.ndarray:
    """Reconstruct Motus' three-view RGB input and resize/pad to target_size=(H,W)."""
    high = decode_lerobot_image(row["observation.images.cam_high"])
    left = decode_lerobot_image(row["observation.images.cam_left_wrist"])
    right = decode_lerobot_image(row["observation.images.cam_right_wrist"])

    orig_h, orig_w = high.shape[:2]
    half_h, half_w = orig_h // 2, orig_w // 2
    left_resized = cv2.resize(left, (half_w, half_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, (half_w, half_h), interpolation=cv2.INTER_AREA)
    combined = np.vstack([high, np.hstack([left_resized, right_resized])])
    return resize_with_padding(combined, target_size)


def rgb_to_tensor_chw(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def tensor_chw_to_rgb(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().float().clamp(0, 1)
    if frame.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(frame.shape)}")
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    arr = (frame.numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def normalize_predicted_frames(predicted_frames: torch.Tensor) -> torch.Tensor:
    """Convert model output to [T,C,H,W]."""
    if predicted_frames.dim() == 5:
        predicted_frames = predicted_frames.squeeze(0)
    if predicted_frames.dim() != 4:
        raise ValueError(f"Expected predicted frames [C,T,H,W] or [T,C,H,W], got {tuple(predicted_frames.shape)}")
    if predicted_frames.shape[0] in (1, 3):
        return predicted_frames.permute(1, 0, 2, 3)
    return predicted_frames


def add_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    out = Image.new("RGB", (image.width, image.height + label_height), (0, 0, 0))
    out.paste(image, (0, label_height))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), label, fill=(255, 255, 255))
    return out


def make_sheet(condition: torch.Tensor, gt_frames: torch.Tensor, pred_frames: torch.Tensor) -> Image.Image:
    condition_img = Image.fromarray(tensor_chw_to_rgb(condition)).convert("RGB")
    gt_imgs = [Image.fromarray(tensor_chw_to_rgb(frame)).convert("RGB") for frame in gt_frames]
    pred_imgs = [Image.fromarray(tensor_chw_to_rgb(frame)).convert("RGB") for frame in pred_frames]

    top_panels = [add_label(condition_img, "condition")]
    bottom_panels = [add_label(condition_img, "condition")]
    for idx, image in enumerate(gt_imgs):
        top_panels.append(add_label(image, f"gt_{idx + 1:02d}"))
    for idx, image in enumerate(pred_imgs):
        bottom_panels.append(add_label(image, f"pred_{idx + 1:02d}"))

    cell_w = max(panel.width for panel in top_panels + bottom_panels)
    cell_h = max(panel.height for panel in top_panels + bottom_panels)
    cols = max(len(top_panels), len(bottom_panels))
    sheet = Image.new("RGB", (cols * cell_w, 2 * cell_h), (0, 0, 0))

    for col, panel in enumerate(top_panels):
        sheet.paste(panel, (col * cell_w, 0))
    for col, panel in enumerate(bottom_panels):
        sheet.paste(panel, (col * cell_w, cell_h))
    return sheet


def make_video_panel(gt_frame: torch.Tensor, pred_frame: torch.Tensor) -> np.ndarray:
    gt = add_label(Image.fromarray(tensor_chw_to_rgb(gt_frame)).convert("RGB"), "GT")
    pred = add_label(Image.fromarray(tensor_chw_to_rgb(pred_frame)).convert("RGB"), "Pred")
    width = max(gt.width, pred.width)
    height = gt.height + pred.height
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(gt, (0, 0))
    canvas.paste(pred, (0, gt.height))
    return np.asarray(canvas)


def write_gt_pred_mp4(path: Path, gt_frames: torch.Tensor, pred_frames: torch.Tensor, fps: int) -> None:
    panels = [make_video_panel(gt, pred) for gt, pred in zip(gt_frames, pred_frames)]
    if not panels:
        return
    height, width = panels[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    for panel in panels:
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    writer.release()


def save_sample_outputs(
    sample_dir: Path,
    condition: torch.Tensor,
    gt_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    metadata: Dict[str, Any],
    fps: int,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    Image.fromarray(tensor_chw_to_rgb(condition)).save(frames_dir / "condition.png")
    for idx, frame in enumerate(gt_frames):
        Image.fromarray(tensor_chw_to_rgb(frame)).save(frames_dir / f"gt_{idx + 1:02d}.png")
    for idx, frame in enumerate(pred_frames):
        Image.fromarray(tensor_chw_to_rgb(frame)).save(frames_dir / f"pred_{idx + 1:02d}.png")

    make_sheet(condition, gt_frames, pred_frames).save(sample_dir / "gt_pred_sheet.png")
    write_gt_pred_mp4(sample_dir / "gt_pred.mp4", gt_frames, pred_frames, fps=fps)
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_windows(args: argparse.Namespace, rng: random.Random) -> List[Dict[str, Any]]:
    dataset_root = Path(args.dataset_root)
    task_dirs = iter_task_dirs(dataset_root, args.tasks)
    windows: List[Dict[str, Any]] = []
    action_chunk_size = args.num_video_frames * args.video_action_freq_ratio

    for task_dir in task_dirs:
        episodes = read_jsonl(task_dir / "meta" / "episodes.jsonl")
        episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
        selected_episodes = episodes[: max(1, args.samples_per_task)]

        for ep in selected_episodes:
            episode_index = int(ep["episode_index"])
            total_frames = int(ep["length"])
            try:
                condition_idx = choose_condition_index(
                    total_frames=total_frames,
                    action_chunk_size=action_chunk_size,
                    global_downsample_rate=args.global_downsample_rate,
                    strategy=args.window_strategy,
                    rng=rng,
                )
            except ValueError as exc:
                LOGGER.warning("Skipping short episode %s/%s: %s", task_dir.name, episode_index, exc)
                continue

            video_indices, action_indices = calculate_indices(
                condition_idx=condition_idx,
                total_frames=total_frames,
                num_video_frames=args.num_video_frames,
                video_action_freq_ratio=args.video_action_freq_ratio,
                global_downsample_rate=args.global_downsample_rate,
            )
            instruction = ""
            if ep.get("tasks"):
                instruction = str(ep["tasks"][0])
            windows.append(
                {
                    "task_dir": str(task_dir),
                    "task_name": task_dir.name.replace("single_task_clean_", ""),
                    "episode_index": episode_index,
                    "episode_length": total_frames,
                    "instruction": instruction,
                    "condition_idx": condition_idx,
                    "video_indices": video_indices,
                    "action_indices": action_indices,
                }
            )
            if args.num_samples and len(windows) >= args.num_samples:
                return windows
    return windows


def load_window_tensors(window: Dict[str, Any], target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_dir = Path(window["task_dir"])
    parquet_path = parquet_path_for_episode(task_dir, int(window["episode_index"]))
    columns = [STATE_COLUMN, ACTION_COLUMN, *IMAGE_COLUMNS]
    df = pd.read_parquet(parquet_path, columns=columns)

    condition_idx = int(window["condition_idx"])
    video_indices = [int(idx) for idx in window["video_indices"]]

    condition_rgb = reconstruct_motus_rgb_frame(df.iloc[condition_idx], target_size)
    gt_rgb = [reconstruct_motus_rgb_frame(df.iloc[idx], target_size) for idx in video_indices]

    condition = rgb_to_tensor_chw(condition_rgb)
    gt_frames = torch.stack([rgb_to_tensor_chw(frame) for frame in gt_rgb], dim=0)

    # Current proprioceptive state for model conditioning. Use observation.state
    # because this LeRobot dataset stores current 14-DoF joint state there.
    state = torch.as_tensor(np.asarray(df.iloc[condition_idx][STATE_COLUMN]), dtype=torch.float32)
    if state.numel() != 14:
        raise ValueError(f"Expected 14-D observation.state, got shape {tuple(state.shape)}")
    return condition, gt_frames, state


def load_policy(args: argparse.Namespace):
    policy_dir = Path(args.policy_dir)
    if str(policy_dir) not in sys.path:
        sys.path.insert(0, str(policy_dir))
    if str(policy_dir / "models") not in sys.path:
        sys.path.insert(0, str(policy_dir / "models"))

    from deploy_policy_save8 import MotusSave8Policy  # noqa: WPS433

    config_path = policy_dir / "utils" / "robotwin.yml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MotusSave8Policy(
        checkpoint_path=args.checkpoint,
        config_path=str(config_path),
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        device=device,
        log_dir=args.output_dir,
        task_name="offline_lerobot",
    )


def encode_t5(policy: Any, instruction: str) -> List[torch.Tensor]:
    t5_out = policy.t5_encoder([instruction], policy.device)
    if isinstance(t5_out, torch.Tensor):
        return [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
    if isinstance(t5_out, list):
        return t5_out
    raise ValueError(f"Unexpected T5 encoder output type: {type(t5_out)!r}")


def run_model_for_window(
    policy: Any,
    condition: torch.Tensor,
    state: torch.Tensor,
    instruction: str,
    num_inference_steps: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    first_frame = condition.unsqueeze(0).to(policy.device)
    state_batch = state.unsqueeze(0).to(policy.device)
    language_embeddings = encode_t5(policy, instruction)
    first_frame_pil = policy._tensor_to_pil_image(condition.cpu())
    vlm_inputs = policy._preprocess_vlm_messages(instruction, first_frame_pil)

    with torch.no_grad():
        predicted_frames, _ = policy.model.inference_step(
            first_frame=first_frame,
            state=state_batch,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=[vlm_inputs],
        )
    return normalize_predicted_frames(predicted_frames).cpu()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    windows = build_windows(args, rng)
    if not windows:
        raise RuntimeError("No valid visualization windows were selected")

    with open(output_dir / "windows.json", "w", encoding="utf-8") as f:
        json.dump(windows, f, indent=2, ensure_ascii=False)
    LOGGER.info("Selected %d windows. Output: %s", len(windows), output_dir)

    policy = None
    if not args.dry_run:
        policy = load_policy(args)
        default_steps = int(policy.config_dict["model"]["inference"]["num_inference_timesteps"])
        num_inference_steps = args.num_inference_steps or default_steps
    else:
        num_inference_steps = args.num_inference_steps or 0

    target_size = (args.video_height, args.video_width)
    metrics_rows: List[Dict[str, Any]] = []

    for sample_idx, window in enumerate(windows):
        LOGGER.info(
            "Sample %03d: %s episode=%s condition=%s",
            sample_idx,
            window["task_name"],
            window["episode_index"],
            window["condition_idx"],
        )
        condition, gt_frames, state = load_window_tensors(window, target_size)
        prefixed_instruction = SCENE_PREFIX + window["instruction"]

        if args.dry_run:
            pred_frames = gt_frames.clone()
        else:
            pred_frames = run_model_for_window(
                policy=policy,
                condition=condition,
                state=state,
                instruction=prefixed_instruction,
                num_inference_steps=num_inference_steps,
                seed=args.seed + sample_idx,
            )
            pred_frames = pred_frames[: args.num_video_frames]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mse = torch.mean((pred_frames.float() - gt_frames[: pred_frames.shape[0]].float()) ** 2).item()
        metadata = {
            **window,
            "sample_index": sample_idx,
            "state_source": STATE_COLUMN,
            "state_shape": list(state.shape),
            "prefixed_instruction": prefixed_instruction,
            "num_inference_steps": num_inference_steps,
            "dry_run": bool(args.dry_run),
            "mse": mse,
        }
        sample_dir = samples_dir / f"sample_{sample_idx:03d}_{window['task_name']}"
        save_sample_outputs(sample_dir, condition, gt_frames, pred_frames, metadata, fps=args.fps)
        metrics_rows.append(
            {
                "sample_index": sample_idx,
                "task_name": window["task_name"],
                "episode_index": window["episode_index"],
                "condition_idx": window["condition_idx"],
                "mse": mse,
                "sample_dir": str(sample_dir),
            }
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2, ensure_ascii=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    LOGGER.info("Done. Wrote %d samples to %s", len(metrics_rows), samples_dir)


if __name__ == "__main__":
    main()
