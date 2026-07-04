#!/usr/bin/env python3
"""Evaluate VGM bridge stage1 checkpoints on fixed video windows."""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover
    safe_load_file = None

sys.path.append(str(Path(__file__).parent.parent))

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset, video_bridge_collate_fn
from models.vgm_bridge_stage1 import VGMBridgeStage1, VGMBridgeStage1Config


logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_step(path: Path) -> Optional[int]:
    match = re.search(r"step_(\d+)", str(path))
    return int(match.group(1)) if match else None


def build_dataset(config: Any, max_episodes: Optional[int] = None) -> VideoBridgeDataset:
    return VideoBridgeDataset(
        dataset_dir=[str(path) for path in config.dataset.dataset_dir],
        global_downsample_rate=config.common.global_downsample_rate,
        num_video_frames=config.common.num_video_frames,
        video_size=(config.common.video_height, config.common.video_width),
        max_episodes=max_episodes if max_episodes is not None else config.dataset.get("max_episodes", None),
        require_language_embedding=config.dataset.get("require_language_embedding", False),
        video_extensions=list(config.dataset.get("video_extensions", [".mp4"])),
        data_format=config.dataset.get("data_format", "auto"),
        image_column=config.dataset.get("image_column", "image"),
        image_columns=config.dataset.get("image_columns", None),
        view_layout=config.dataset.get("view_layout", "single"),
        task_language_embedding_dir=config.dataset.get("task_language_embedding_dir", None),
        task_language_embedding_pattern=config.dataset.get("task_language_embedding_pattern", "task_{task_index:06d}.pt"),
        cache_scan=config.dataset.get("cache_scan", True),
        val=True,
    )


def build_model(config: Any, checkpoint_path: Path) -> VGMBridgeStage1:
    model_config = VGMBridgeStage1Config(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        video_precision=config.model.wan.precision,
        num_video_frames=config.common.num_video_frames,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        tail_condition_frames=config.common.get("tail_condition_frames", 1),
        load_pretrained_backbones=False,
    )
    model = VGMBridgeStage1(model_config)
    load_model_checkpoint(model, checkpoint_path)
    return model


def load_model_checkpoint(model: VGMBridgeStage1, checkpoint_path: Path) -> None:
    if checkpoint_path.is_dir():
        candidates = [
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin",
            checkpoint_path / "training_state.pt",
        ]
        candidates += sorted(checkpoint_path.glob("pytorch_model_*.bin"))
        checkpoint_file = next((path for path in candidates if path.exists()), None)
        if checkpoint_file is None:
            raise FileNotFoundError(f"No model checkpoint found under {checkpoint_path}")
    else:
        checkpoint_file = checkpoint_path

    logger.info("Loading model checkpoint from %s", checkpoint_file)
    if checkpoint_file.suffix == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("safetensors is required to load model.safetensors")
        state_dict = safe_load_file(str(checkpoint_file), device="cpu")
    else:
        state = torch.load(checkpoint_file, map_location="cpu")
        state_dict = state.get("model", state) if isinstance(state, dict) else state

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    model_keys = set(model.state_dict().keys())
    matched_wan_keys = [
        key for key in cleaned.keys()
        if key in model_keys and key.startswith("video_model.wan_model.")
    ]
    if len(matched_wan_keys) < 10:
        raise RuntimeError(
            f"Checkpoint {checkpoint_file} does not look like a VGMBridgeStage1 model "
            f"checkpoint: only {len(matched_wan_keys)} WAN tensors match the model."
        )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing checkpoint keys: %s", missing[:20])
    if unexpected:
        logger.warning("Unexpected checkpoint keys: %s", unexpected[:20])
    logger.info("Checkpoint loaded: %s tensors; matched WAN tensors=%s", len(cleaned), len(matched_wan_keys))


def fixed_windows(dataset: VideoBridgeDataset, num_samples: int) -> List[Dict[str, Any]]:
    if len(dataset.episodes) == 0:
        raise RuntimeError("No episodes found for evaluation")

    windows = []
    stride = max(1, len(dataset.episodes) // max(1, num_samples))
    episode_indices = [(idx * stride) % len(dataset.episodes) for idx in range(num_samples)]

    for sample_idx, episode_index in enumerate(episode_indices):
        episode = dataset.episodes[episode_index]
        total_frames = dataset._episode_frame_count(episode)
        max_cond = total_frames - 1 - dataset.num_video_frames * dataset.global_downsample_rate
        if max_cond < 0:
            logger.warning("Skipping short episode %s: %s frames", episode.get("episode_name"), total_frames)
            continue
        # Spread windows across each episode instead of always using the exact center.
        fraction = ((sample_idx % 5) + 1) / 6.0
        condition_idx = int(round(max_cond * fraction))
        windows.append(dataset.get_bridge_window(episode_index, condition_idx=condition_idx))

    if not windows:
        raise RuntimeError("No valid fixed windows could be loaded")
    return windows


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[Dict[str, Any]]:
    for start in range(0, len(items), batch_size):
        batch = video_bridge_collate_fn(items[start : start + batch_size])
        if batch is not None:
            yield batch


def tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle((x - 4, y - 2, x + 8 + len(text) * 7, y + 14), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 24, 31))


def make_comparison_frame(gt_frame: torch.Tensor, pred_frame: torch.Tensor, frame_idx: int) -> np.ndarray:
    gt_image = Image.fromarray(tensor_to_uint8(gt_frame))
    pred_image = Image.fromarray(tensor_to_uint8(pred_frame))
    width, height = gt_image.size
    label_h = 24
    canvas = Image.new("RGB", (width, height * 2 + label_h), (255, 255, 255))
    canvas.paste(gt_image, (0, label_h))
    canvas.paste(pred_image, (0, label_h + height))
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (8, 5), f"t={frame_idx:02d}  GT")
    draw_label(draw, (8, label_h + height + 5), "Pred")
    return np.asarray(canvas)


def save_comparison_video(
    gt_full: torch.Tensor,
    pred_full: torch.Tensor,
    output_path: Path,
    fps: int,
) -> None:
    if imageio is None:
        raise RuntimeError("imageio is required to write MP4 comparison videos")
    frames = [
        make_comparison_frame(gt_full[t], pred_full[t], t)
        for t in range(min(gt_full.shape[0], pred_full.shape[0]))
    ]
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)


def save_contact_sheet(gt_full: torch.Tensor, pred_full: torch.Tensor, output_path: Path) -> None:
    frame_count = min(gt_full.shape[0], pred_full.shape[0])
    gt_images = [Image.fromarray(tensor_to_uint8(gt_full[t])) for t in range(frame_count)]
    pred_images = [Image.fromarray(tensor_to_uint8(pred_full[t])) for t in range(frame_count)]
    width, height = gt_images[0].size
    label_h = 28
    canvas = Image.new("RGB", (width * frame_count, height * 2 + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "GT row; Pred row", fill=(20, 24, 31))
    for idx, image in enumerate(gt_images):
        canvas.paste(image, (idx * width, label_h))
    for idx, image in enumerate(pred_images):
        canvas.paste(image, (idx * width, label_h + height))
    canvas.save(output_path)


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def pixel_metrics(gt_full: torch.Tensor, pred_full: torch.Tensor, tail_condition_frames: int = 1) -> Dict[str, float]:
    frame_count = min(gt_full.shape[1], pred_full.shape[1])
    gt_full = gt_full[:, :frame_count]
    pred_full = pred_full[:, :frame_count]
    tail_condition_frames = max(1, min(int(tail_condition_frames), frame_count - 1))

    metrics: Dict[str, float] = {}
    full_mse = F.mse_loss(pred_full, gt_full).item()
    metrics["full_mse"] = full_mse
    metrics["full_psnr"] = psnr_from_mse(full_mse)
    metrics["full_mae"] = F.l1_loss(pred_full, gt_full).item()

    if frame_count > 2:
        middle_gt = gt_full[:, 1:-1]
        middle_pred = pred_full[:, 1:-1]
        middle_mse = F.mse_loss(middle_pred, middle_gt).item()
        metrics["middle_mse"] = middle_mse
        metrics["middle_psnr"] = psnr_from_mse(middle_mse)
        metrics["middle_mae"] = F.l1_loss(middle_pred, middle_gt).item()

    generated_end = frame_count - tail_condition_frames
    if generated_end > 1:
        generated_gt = gt_full[:, 1:generated_end]
        generated_pred = pred_full[:, 1:generated_end]
        generated_mse = F.mse_loss(generated_pred, generated_gt).item()
        metrics["generated_mse"] = generated_mse
        metrics["generated_psnr"] = psnr_from_mse(generated_mse)
        metrics["generated_mae"] = F.l1_loss(generated_pred, generated_gt).item()

    condition_gt = torch.cat([gt_full[:, 0:1], gt_full[:, -tail_condition_frames:]], dim=1)
    condition_pred = torch.cat([pred_full[:, 0:1], pred_full[:, -tail_condition_frames:]], dim=1)
    condition_mse = F.mse_loss(condition_pred, condition_gt).item()
    metrics["condition_mse"] = condition_mse
    metrics["condition_psnr"] = psnr_from_mse(condition_mse)
    return metrics


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    result = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result


@torch.no_grad()
def evaluate_loss(
    model: VGMBridgeStage1,
    windows: List[Dict[str, Any]],
    batch_size: int,
    seed: int,
    loss_repeats: int,
) -> Dict[str, float]:
    rows = []
    repeats = max(1, int(loss_repeats))
    for repeat_idx in range(repeats):
        for batch_idx, batch in enumerate(batched(windows, batch_size)):
            step_seed = seed + repeat_idx * 100_000 + batch_idx
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)
            first_frame = batch["first_frame"].to(model.device, dtype=model.dtype)
            video_frames = batch["video_frames"].to(model.device, dtype=model.dtype)
            language_embeddings = batch["language_embedding"]
            if language_embeddings is not None:
                language_embeddings = language_embeddings.to(model.device, dtype=model.dtype)
            loss_dict = model.training_step(
                first_frame=first_frame,
                video_frames=video_frames,
                language_embeddings=language_embeddings,
                return_dict=True,
            )
            rows.append({key: float(value.detach().cpu()) for key, value in loss_dict.items()})
    return mean_metrics(rows)


@torch.no_grad()
def evaluate_samples(
    model: VGMBridgeStage1,
    windows: List[Dict[str, Any]],
    batch_size: int,
    output_dir: Path,
    num_inference_steps: int,
    seed: int,
    fps: int,
) -> Dict[str, float]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    sample_index = 0

    for batch_idx, batch in enumerate(batched(windows, batch_size)):
        first_frame = batch["first_frame"].to(model.device, dtype=model.dtype)
        video_frames = batch["video_frames"].to(model.device, dtype=model.dtype)
        last_frame = video_frames[:, -1]
        tail_n = int(model.config.tail_condition_frames)
        tail_frames = video_frames[:, -tail_n:]
        language_embeddings = batch["language_embedding"]
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(model.device, dtype=model.dtype)

        generator = torch.Generator(device=model.device).manual_seed(seed + batch_idx)
        pred_full = model.sample_bridge(
            first_frame=first_frame,
            last_frame=last_frame,
            tail_frames=tail_frames,
            language_embeddings=language_embeddings,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        gt_full = torch.cat([first_frame.unsqueeze(1).float(), video_frames.float()], dim=1).clamp(0, 1)
        batch_metrics = pixel_metrics(
            gt_full.float(),
            pred_full.float(),
            tail_condition_frames=model.config.tail_condition_frames,
        )
        metric_rows.append(batch_metrics)

        for local_idx in range(pred_full.shape[0]):
            name = f"sample_{sample_index:03d}"
            save_contact_sheet(
                gt_full[local_idx],
                pred_full[local_idx],
                sample_dir / f"{name}_sheet.png",
            )
            save_comparison_video(
                gt_full[local_idx],
                pred_full[local_idx],
                sample_dir / f"{name}_gt_pred.mp4",
                fps=fps,
            )
            sample_index += 1

    return mean_metrics(metric_rows)


def write_csv(path: Path, data: Dict[str, float]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key in sorted(data):
            writer.writerow([key, data[key]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VGM bridge stage1 checkpoints")
    parser.add_argument("--config", default="configs/vgm_bridge_v0.yaml")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_step_* directory or model file")
    parser.add_argument("--output_dir", default="eval_outputs/vgm_bridge_stage1")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loss_repeats", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    if not torch.cuda.is_available():
        raise RuntimeError("eval_vgm_bridge_stage1.py requires CUDA")

    torch.cuda.set_device(0)
    config = OmegaConf.load(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_step = parse_step(checkpoint_path)
    run_name = checkpoint_path.parent.name if checkpoint_path.parent.name else "checkpoint"
    output_dir = Path(args.output_dir) / run_name / (f"step_{checkpoint_step}" if checkpoint_step else checkpoint_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataset...")
    dataset = build_dataset(config, max_episodes=args.max_episodes)
    windows = fixed_windows(dataset, args.num_samples)
    with (output_dir / "windows.json").open("w") as file:
        json.dump(
            [
                {
                    "episode_name": item["episode_name"],
                    "condition_idx": item["condition_idx"],
                    "frame_indices": item["frame_indices"],
                    "total_frames": item["total_frames"],
                    "video_path": item["video_path"],
                    "task_index": item.get("task_index"),
                    "task_text": item.get("task_text"),
                }
                for item in windows
            ],
            file,
            indent=2,
        )

    logger.info("Building model...")
    model = build_model(config, checkpoint_path)
    model.eval()

    logger.info("Evaluating fixed-window training loss...")
    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "num_samples": len(windows),
        "loss_repeats": args.loss_repeats,
        "num_inference_steps": args.num_inference_steps,
        "loss": evaluate_loss(model, windows, args.batch_size, args.seed, args.loss_repeats),
    }

    if not args.loss_only:
        logger.info("Generating bridge samples...")
        metrics["samples"] = evaluate_samples(
            model=model,
            windows=windows,
            batch_size=args.batch_size,
            output_dir=output_dir,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            fps=args.fps,
        )

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    flat_metrics = {}
    for group, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    flat_metrics[f"{group}.{key}"] = float(value)
    write_csv(output_dir / "metrics.csv", flat_metrics)

    logger.info("=== VGM Bridge Stage1 Eval ===")
    for key, value in flat_metrics.items():
        logger.info("%s: %.6f", key, value)
    logger.info("Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
