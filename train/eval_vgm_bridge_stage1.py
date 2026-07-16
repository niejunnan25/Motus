#!/usr/bin/env python3
"""Evaluate VGM bridge stage1 checkpoints on fixed video windows."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

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
from models.vgm_multiview import SEPARATE_VIEW_MODES
from utils.config_utils import load_config_with_base


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
        video_size=(
            config.dataset.get("video_height", config.common.video_height),
            config.dataset.get("video_width", config.common.video_width),
        ),
        max_episodes=max_episodes if max_episodes is not None else config.dataset.get("max_episodes", None),
        require_language_embedding=config.dataset.get("require_language_embedding", False),
        video_extensions=list(config.dataset.get("video_extensions", [".mp4"])),
        data_format=config.dataset.get("data_format", "auto"),
        image_column=config.dataset.get("image_column", "image"),
        image_columns=config.dataset.get("image_columns", None),
        view_layout=config.dataset.get("view_layout", "single"),
        return_separate_views=config.dataset.get(
            "return_separate_views",
            config.common.get("multiview_mode", "legacy") in SEPARATE_VIEW_MODES,
        ),
        view_video_size=tuple(config.dataset.get("view_video_size", [224, 224])),
        task_language_embedding_dir=config.dataset.get("task_language_embedding_dir", None),
        task_language_embedding_pattern=config.dataset.get("task_language_embedding_pattern", "task_{task_index:06d}.pt"),
        task_language_caption_version=config.dataset.get("task_language_caption_version", None),
        load_state=config.dataset.get(
            "load_state",
            config.common.get("state_condition_mode", "none") != "none",
        ),
        state_column=config.dataset.get("state_column", "observation.state"),
        bridge_sampling_mode=config.dataset.get("bridge_sampling_mode", "sliding_window"),
        bridge_sampling_jitter=config.dataset.get("bridge_sampling_jitter", False),
        index_seeded_sampling=config.dataset.get("index_seeded_sampling", False),
        sampling_seed=int(config.training.get("seed", 0)),
        load_role_mask=config.dataset.get("load_role_mask", False),
        role_mask_columns=config.dataset.get("role_mask_columns", None),
        role_mask_render_mode=config.dataset.get("role_mask_render_mode", "binary"),
        role_mask_foreground_ids=config.dataset.get("role_mask_foreground_ids", [1, 2, 4]),
        role_mask_palette=config.dataset.get("role_mask_palette", None),
        role_mask_cache_dir=config.dataset.get("role_mask_cache_dir", None),
        role_mask_memory_cache_size=config.dataset.get("role_mask_memory_cache_size", 1),
        strict_role_mask=config.dataset.get("strict_role_mask", True),
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
        conditioning_mode=config.common.get("conditioning_mode", "v0"),
        mask_channels=config.common.get("mask_channels", 4),
        interaction_loss_enabled=config.common.get("interaction_loss_enabled", False),
        interaction_weight_mode=config.common.get("interaction_weight_mode", "motion_edge"),
        interaction_motion_weight=config.common.get("interaction_motion_weight", 0.0),
        interaction_edge_weight=config.common.get("interaction_edge_weight", 0.0),
        interaction_max_weight=config.common.get("interaction_max_weight", 4.0),
        interaction_warmup_steps=config.common.get("interaction_warmup_steps", 0),
        interaction_mask_percentile=config.common.get("interaction_mask_percentile", 0.92),
        interaction_mask_min_threshold=config.common.get("interaction_mask_min_threshold", 0.012),
        interaction_mask_dilation=config.common.get("interaction_mask_dilation", 7),
        interaction_proximity_dilation=config.common.get("interaction_proximity_dilation", 57),
        state_condition_mode=config.common.get("state_condition_mode", "none"),
        state_num_tokens=config.common.get("state_num_tokens", 4),
        state_hidden_dim=config.common.get("state_hidden_dim", 1024),
        state_dropout=config.common.get("state_dropout", 0.0),
        state_clip=config.common.get("state_clip", 10.0),
        role_mask_fusion_mode=config.common.get("role_mask_fusion_mode", "none"),
        role_mask_loss_weight=config.common.get("role_mask_loss_weight", 1.0),
        role_mask_training_mode=config.common.get("role_mask_training_mode", "legacy"),
        role_mask_condition_mode=config.common.get("role_mask_condition_mode", "legacy"),
        role_mask_prompt_dropout=config.common.get("role_mask_prompt_dropout", 0.5),
        role_mask_loss_warmup_steps=config.common.get("role_mask_loss_warmup_steps", 0),
        role_rgb_max_weight=config.common.get("role_rgb_max_weight", 8.0),
        role_rgb_weight_warmup_steps=config.common.get("role_rgb_weight_warmup_steps", 1000),
        multiview_mode=config.common.get("multiview_mode", "legacy"),
        num_views=config.common.get("num_views", 2),
        multiview_layout=config.common.get("multiview_layout", config.dataset.get("view_layout", "vertical")),
        view_capacity_mode=config.common.get("view_capacity_mode", "shared"),
        multiview_layer_indices=config.common.get("multiview_layer_indices", None),
        multiview_adapter_dim=config.common.get("multiview_adapter_dim", 256),
        multiview_adapter_heads=config.common.get("multiview_adapter_heads", 8),
        multiview_scene_tokens=config.common.get("multiview_scene_tokens", 4),
        multiview_consistency_weight=config.common.get("multiview_consistency_weight", 0.1),
        multiview_consistency_temperature=config.common.get("multiview_consistency_temperature", 0.07),
        multiview_consistency_projection_dim=config.common.get("multiview_consistency_projection_dim", 128),
        multiview_output_layout=config.common.get("multiview_output_layout", "vertical"),
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

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_file} is incompatible with the requested evaluation config. "
            "Refusing to evaluate with missing, unexpected, or shape-mismatched tensors."
        ) from exc
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
        if dataset.bridge_sampling_mode == "full_episode_uniform":
            required_frames = dataset.num_video_frames + 1
            if total_frames < required_frames:
                logger.warning(
                    "Skipping short episode %s: %s frames; needs at least %s",
                    episode.get("episode_name"),
                    total_frames,
                    required_frames,
                )
                continue
            windows.append(dataset.get_bridge_window(episode_index))
            continue

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


def fixed_windows_from_manifest(
    dataset: VideoBridgeDataset,
    manifest_path: Path,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reload the exact episode/window definitions used by another experiment."""
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Window manifest must contain a non-empty JSON list: {manifest_path}")
    if num_samples is not None:
        entries = entries[: int(num_samples)]
    episode_indices: Dict[str, List[int]] = {}
    for index, episode in enumerate(dataset.episodes):
        episode_indices.setdefault(str(episode.get("episode_name")), []).append(index)

    windows = []
    for entry in entries:
        episode_name = str(entry.get("episode_name"))
        matches = episode_indices.get(episode_name, [])
        if len(matches) != 1:
            raise ValueError(
                f"Window manifest episode {episode_name!r} matched {len(matches)} dataset episodes"
            )
        window = dataset.get_bridge_window(
            matches[0],
            condition_idx=entry.get("condition_idx"),
        )
        expected_indices = [int(value) for value in entry.get("frame_indices", [])]
        if expected_indices and window["frame_indices"] != expected_indices:
            raise ValueError(
                f"Window indices changed for {episode_name}: manifest={expected_indices}, "
                f"dataset={window['frame_indices']}"
            )
        windows.append(window)
    return windows


def window_records(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "episode_name": item["episode_name"],
            "condition_idx": item["condition_idx"],
            "frame_indices": item["frame_indices"],
            "total_frames": item["total_frames"],
            "video_path": item["video_path"],
            "task_index": item.get("task_index"),
            "task_text": item.get("task_text"),
            "language_caption": item.get("language_caption"),
            "language_caption_version": item.get("language_caption_version"),
            "language_caption_field": item.get("language_caption_field"),
            "language_caption_manifest": item.get("language_caption_manifest"),
            "language_embedding_path": item.get("language_embedding_path"),
        }
        for item in windows
    ]


def window_fingerprint(records: List[Dict[str, Any]]) -> str:
    # Absolute dataset/cache paths differ across machines and are not part of
    # the sampled-window identity. Hash only semantic episode/frame fields.
    stable_records = [
        {
            "episode_name": record.get("episode_name"),
            "task_index": record.get("task_index"),
            "frame_indices": record.get("frame_indices"),
            "total_frames": record.get("total_frames"),
        }
        for record in records
    ]
    canonical = json.dumps(stable_records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[Dict[str, Any]]:
    for start in range(0, len(items), batch_size):
        batch = video_bridge_collate_fn(items[start : start + batch_size])
        if batch is not None:
            yield batch


def tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


class DinoFrameEncoder:
    """Optional local DINO/DINOv2 encoder for perceptual and slot diagnostics."""

    def __init__(self, model_path: Path, device: torch.device, batch_size: int = 32) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for --dino_model_path") from exc
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        ).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.batch_size = max(1, int(batch_size))

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        features = []
        for start in range(0, video.shape[0], self.batch_size):
            images = [tensor_to_uint8(frame) for frame in video[start : start + self.batch_size]]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(self.device, non_blocking=True)
                for key, value in inputs.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                output = self.model(**inputs)
            feature = getattr(output, "pooler_output", None)
            if feature is None:
                feature = output.last_hidden_state[:, 0]
            features.append(F.normalize(feature.float(), dim=-1))
        return torch.cat(features, dim=0)


class LPIPSMetric:
    """Optional true LPIPS metric; imported only when explicitly requested."""

    def __init__(self, device: torch.device, network: str = "alex", batch_size: int = 32) -> None:
        try:
            import lpips
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The lpips package is required for --enable_lpips; install it in the eval environment"
            ) from exc
        self.model = lpips.LPIPS(net=network).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.batch_size = max(1, int(batch_size))

    @torch.no_grad()
    def __call__(self, gt: torch.Tensor, pred: torch.Tensor) -> float:
        values = []
        for start in range(0, gt.shape[0], self.batch_size):
            gt_batch = gt[start : start + self.batch_size].to(self.device).float() * 2.0 - 1.0
            pred_batch = pred[start : start + self.batch_size].to(self.device).float() * 2.0 - 1.0
            values.append(self.model(gt_batch, pred_batch).flatten().float().cpu())
        return float(torch.cat(values).mean())


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle((x - 4, y - 2, x + 8 + len(text) * 7, y + 14), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 24, 31))


def visualization_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


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
    """Save every frame in two full rows with readable row and slot labels."""
    frame_count = min(gt_full.shape[0], pred_full.shape[0])
    gt_images = [Image.fromarray(tensor_to_uint8(gt_full[t])) for t in range(frame_count)]
    pred_images = [Image.fromarray(tensor_to_uint8(pred_full[t])) for t in range(frame_count)]
    width, height = gt_images[0].size
    header_h = 34
    row_label_w = max(84, width // 3)
    canvas = Image.new(
        "RGB",
        (row_label_w + width * frame_count, height * 2 + header_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    row_font = visualization_font(max(18, min(30, height // 10)))
    slot_font = visualization_font(max(12, min(20, width // 10)))
    draw.text((10, header_h + height // 2 - 10), "GT", fill=(20, 24, 31), font=row_font)
    draw.text(
        (10, header_h + height + height // 2 - 10),
        "Pred",
        fill=(20, 24, 31),
        font=row_font,
    )
    for idx, image in enumerate(gt_images):
        x = row_label_w + idx * width
        canvas.paste(image, (x, header_h))
        label = f"{idx:02d}"
        label_box = draw.textbbox((0, 0), label, font=slot_font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (x + max(2, (width - label_width) // 2), 6),
            label,
            fill=(20, 24, 31),
            font=slot_font,
        )
    for idx, image in enumerate(pred_images):
        canvas.paste(image, (row_label_w + idx * width, header_h + height))
    canvas.save(output_path)


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def normalize_positive_map(value: torch.Tensor) -> torch.Tensor:
    mean = value.mean(dim=tuple(range(1, value.dim())), keepdim=True).clamp_min(1e-6)
    return value / mean


def edge_strength_video(video: torch.Tensor) -> torch.Tensor:
    """Return Sobel edge strength for [B, T, C, H, W] videos as [B, T, 1, H, W]."""
    batch_size, frame_count, _, height, width = video.shape
    gray = video.mean(dim=2, keepdim=True)
    flat = gray.reshape(batch_size * frame_count, 1, height, width)
    kernel_x = flat.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = flat.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    grad_x = F.conv2d(flat, kernel_x, padding=1)
    grad_y = F.conv2d(flat, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
    return edge.view(batch_size, frame_count, 1, height, width)


def weighted_video_mse(gt: torch.Tensor, pred: torch.Tensor, weight: torch.Tensor) -> float:
    error = (pred - gt).pow(2).mean(dim=2, keepdim=True)
    weighted = error * weight
    return (weighted.sum() / weight.sum().clamp_min(1.0)).item()


def pixel_metrics(
    gt_full: torch.Tensor,
    pred_full: torch.Tensor,
    tail_condition_frames: int = 1,
    hard_clamped_tail_frames: Optional[int] = None,
) -> Dict[str, float]:
    frame_count = min(gt_full.shape[1], pred_full.shape[1])
    gt_full = gt_full[:, :frame_count]
    pred_full = pred_full[:, :frame_count]
    tail_condition_frames = max(0, min(int(tail_condition_frames), frame_count - 1))
    if hard_clamped_tail_frames is None:
        hard_clamped_tail_frames = tail_condition_frames
    hard_clamped_tail_frames = max(
        0,
        min(int(hard_clamped_tail_frames), frame_count - 1),
    )

    metrics: Dict[str, float] = {}
    full_mse = F.mse_loss(pred_full, gt_full).item()
    metrics["full_mse"] = full_mse
    metrics["full_psnr"] = psnr_from_mse(full_mse)
    metrics["full_mae"] = F.l1_loss(pred_full, gt_full).item()
    endpoint_mse = F.mse_loss(pred_full[:, -1:], gt_full[:, -1:]).item()
    metrics["endpoint_mse"] = endpoint_mse
    metrics["endpoint_psnr"] = psnr_from_mse(endpoint_mse)
    metrics["endpoint_mae"] = F.l1_loss(pred_full[:, -1:], gt_full[:, -1:]).item()

    if frame_count > 2:
        middle_gt = gt_full[:, 1:-1]
        middle_pred = pred_full[:, 1:-1]
        middle_mse = F.mse_loss(middle_pred, middle_gt).item()
        metrics["middle_mse"] = middle_mse
        metrics["middle_psnr"] = psnr_from_mse(middle_mse)
        metrics["middle_mae"] = F.l1_loss(middle_pred, middle_gt).item()

    generated_end = frame_count - hard_clamped_tail_frames
    if generated_end > 1:
        generated_gt = gt_full[:, 1:generated_end]
        generated_pred = pred_full[:, 1:generated_end]
        generated_mse = F.mse_loss(generated_pred, generated_gt).item()
        metrics["generated_mse"] = generated_mse
        metrics["generated_psnr"] = psnr_from_mse(generated_mse)
        metrics["generated_mae"] = F.l1_loss(generated_pred, generated_gt).item()

        motion = gt_full.new_zeros(gt_full.shape[0], frame_count, 1, gt_full.shape[-2], gt_full.shape[-1])
        motion[:, 1:] = (gt_full[:, 1:] - gt_full[:, :-1]).abs().mean(dim=2, keepdim=True)
        generated_motion = normalize_positive_map(motion[:, 1:generated_end])
        motion_mse = weighted_video_mse(generated_gt, generated_pred, generated_motion)
        metrics["generated_motion_mse"] = motion_mse
        metrics["generated_motion_psnr"] = psnr_from_mse(motion_mse)

        edge = edge_strength_video(gt_full)
        generated_edge = normalize_positive_map(edge[:, 1:generated_end])
        edge_mse = weighted_video_mse(generated_gt, generated_pred, generated_edge)
        metrics["generated_edge_mse"] = edge_mse
        metrics["generated_edge_psnr"] = psnr_from_mse(edge_mse)

        interaction_weight = (1.0 + 2.0 * generated_motion + generated_edge).clamp(max=4.0)
        interaction_weight = normalize_positive_map(interaction_weight)
        interaction_mse = weighted_video_mse(generated_gt, generated_pred, interaction_weight)
        metrics["generated_interaction_mse"] = interaction_mse
        metrics["generated_interaction_psnr"] = psnr_from_mse(interaction_mse)

    condition_parts_gt = [gt_full[:, 0:1]]
    condition_parts_pred = [pred_full[:, 0:1]]
    if tail_condition_frames > 0:
        condition_parts_gt.append(gt_full[:, -tail_condition_frames:])
        condition_parts_pred.append(pred_full[:, -tail_condition_frames:])
    condition_gt = torch.cat(condition_parts_gt, dim=1)
    condition_pred = torch.cat(condition_parts_pred, dim=1)
    condition_mse = F.mse_loss(condition_pred, condition_gt).item()
    metrics["condition_mse"] = condition_mse
    metrics["condition_psnr"] = psnr_from_mse(condition_mse)
    return metrics


def split_rgb_views(
    video: torch.Tensor,
    view_layout: str,
    view_names: Optional[List[str]],
) -> List[tuple[str, torch.Tensor]]:
    """Split [B,F,C,H,W] or [F,C,H,W] RGB mosaics into named camera tensors."""
    names = list(view_names or ["view_0"])
    if view_layout == "single":
        return [(names[0] if names else "view_0", video)]
    if view_layout not in {"vertical", "horizontal"}:
        raise ValueError(f"Unsupported view_layout={view_layout!r}")
    split_dim = -2 if view_layout == "vertical" else -1
    if video.shape[split_dim] % len(names) != 0:
        raise ValueError(
            f"RGB {view_layout} dimension {video.shape[split_dim]} is not divisible by {len(names)} views"
        )
    chunks = torch.chunk(video, len(names), dim=split_dim)
    result = []
    for index, (name, chunk) in enumerate(zip(names, chunks)):
        clean_name = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()
        result.append((clean_name or f"view_{index}", chunk))
    return result


def multiview_pixel_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    *,
    tail_condition_frames: int,
    hard_clamped_tail_frames: Optional[int] = None,
    view_layout: str,
    view_names: Optional[List[str]],
) -> Dict[str, float]:
    if view_layout == "single":
        return {}
    gt_views = split_rgb_views(gt, view_layout, view_names)
    pred_views = split_rgb_views(pred, view_layout, view_names)
    metrics: Dict[str, float] = {}
    macro: Dict[str, List[float]] = {}
    for (name, gt_view), (pred_name, pred_view) in zip(gt_views, pred_views):
        if name != pred_name:
            raise RuntimeError(f"GT/pred view names differ: {name!r} vs {pred_name!r}")
        view_metrics = pixel_metrics(
            gt_view,
            pred_view,
            tail_condition_frames,
            hard_clamped_tail_frames=hard_clamped_tail_frames,
        )
        for key, value in view_metrics.items():
            metrics[f"view_{name}_{key}"] = value
            macro.setdefault(key, []).append(value)
    for key, values in macro.items():
        metrics[f"view_macro_{key}"] = float(np.mean(values))
    return metrics


def cross_view_motion_sync_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    *,
    view_layout: str,
    view_names: Optional[List[str]],
) -> Dict[str, float]:
    """Compare high/wrist temporal motion profiles without requiring a feature backbone."""
    if view_layout == "single":
        return {}
    gt_views = split_rgb_views(gt, view_layout, view_names)
    pred_views = split_rgb_views(pred, view_layout, view_names)
    if len(gt_views) != 2 or len(pred_views) != 2:
        return {}

    def profile(video: torch.Tensor) -> torch.Tensor:
        motion = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(2, 3, 4))
        return motion / motion.mean(dim=1, keepdim=True).clamp_min(1e-6)

    gt_profiles = [profile(view) for _, view in gt_views]
    pred_profiles = [profile(view) for _, view in pred_views]
    gt_sync = (gt_profiles[0] - gt_profiles[1]).abs().mean()
    pred_sync = (pred_profiles[0] - pred_profiles[1]).abs().mean()
    steps = max(1, pred_profiles[0].shape[1] - 1)
    gt_peak_offset = (
        gt_profiles[0].argmax(dim=1) - gt_profiles[1].argmax(dim=1)
    ).abs().float().mean() / steps
    pred_peak_offset = (
        pred_profiles[0].argmax(dim=1) - pred_profiles[1].argmax(dim=1)
    ).abs().float().mean() / steps
    return {
        "xview_gt_motion_profile_l1": float(gt_sync.cpu()),
        "xview_pred_motion_profile_l1": float(pred_sync.cpu()),
        "xview_motion_profile_excess_l1": float((pred_sync - gt_sync).abs().cpu()),
        "xview_gt_motion_peak_offset": float(gt_peak_offset.cpu()),
        "xview_pred_motion_peak_offset": float(pred_peak_offset.cpu()),
        "xview_motion_peak_offset_error": float((pred_peak_offset - gt_peak_offset).abs().cpu()),
    }


def trajectory_frame_features(video: torch.Tensor, spatial_size: int = 16) -> torch.Tensor:
    """Build deterministic RGB/edge/delta features for slot-level diagnostics."""
    if video.ndim != 4:
        raise ValueError(f"video must be [F,C,H,W], got {tuple(video.shape)}")
    rgb = F.adaptive_avg_pool2d(video.float(), (spatial_size, spatial_size))
    delta = rgb - rgb[0:1]
    edge = edge_strength_video(video.unsqueeze(0).float())[0]
    edge = F.adaptive_avg_pool2d(edge, (spatial_size, spatial_size))
    features = torch.cat(
        [
            0.5 * rgb.flatten(1),
            2.0 * delta.flatten(1),
            edge.flatten(1),
        ],
        dim=1,
    )
    return F.normalize(features, dim=1)


def trajectory_slot_diagnostics(
    gt_video: torch.Tensor,
    pred_video: torch.Tensor,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Align every generated frame to all GT slots and measure jumps/collisions."""
    frame_count = min(gt_video.shape[0], pred_video.shape[0])
    if frame_count < 2:
        raise ValueError("Slot diagnostics require at least two frames")
    gt_features = trajectory_frame_features(gt_video[:frame_count])
    pred_features = trajectory_frame_features(pred_video[:frame_count])
    return trajectory_slot_diagnostics_from_features(gt_features, pred_features)


def trajectory_slot_diagnostics_from_features(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Compute slot diagnostics from normalized frame feature matrices [F,D]."""
    frame_count = min(gt_features.shape[0], pred_features.shape[0])
    if frame_count < 2 or gt_features.ndim != 2 or pred_features.ndim != 2:
        raise ValueError(
            "Slot feature diagnostics require [F,D] tensors with at least two frames"
        )
    gt_features = F.normalize(gt_features[:frame_count].float(), dim=1)
    pred_features = F.normalize(pred_features[:frame_count].float(), dim=1)
    similarity = pred_features @ gt_features.transpose(0, 1)
    alignment = similarity.argmax(dim=1)
    target = torch.arange(frame_count, device=alignment.device)
    normalizer = float(max(1, frame_count - 1))
    steps = alignment[1:] - alignment[:-1]
    unique_ratio = alignment.unique().numel() / float(frame_count)

    far_gap = max(4, frame_count // 10)
    positions = torch.arange(frame_count, device=alignment.device)
    far_pairs = (positions[:, None] - positions[None, :]).abs() >= far_gap
    upper = torch.triu(torch.ones_like(far_pairs), diagonal=1).bool()
    candidate_pairs = far_pairs & upper
    same_slot = alignment[:, None].eq(alignment[None, :])
    far_collision = (
        (same_slot & candidate_pairs).sum().float()
        / candidate_pairs.sum().clamp_min(1).float()
    )
    diagonal = similarity.diagonal().mean()
    off_diagonal_mask = ~torch.eye(frame_count, device=similarity.device, dtype=torch.bool)
    off_diagonal = similarity[off_diagonal_mask].mean()
    metrics = {
        "slot_alignment_mae": float((alignment - target).abs().float().mean().cpu() / normalizer),
        "slot_backward_rate": float((steps < 0).float().mean().cpu()),
        "slot_large_jump_rate": float((steps.abs() > max(2, frame_count // 10)).float().mean().cpu()),
        "slot_unique_match_ratio": float(unique_ratio),
        "slot_collision_rate": float(1.0 - unique_ratio),
        "slot_far_collision_rate": float(far_collision.cpu()),
        "slot_start_error": float(alignment[0].float().cpu() / normalizer),
        "slot_end_error": float((alignment[-1] - (frame_count - 1)).abs().float().cpu() / normalizer),
        "slot_diagonal_similarity": float(diagonal.cpu()),
        "slot_diagonal_margin": float((diagonal - off_diagonal).cpu()),
    }
    return metrics, similarity, alignment


def save_similarity_matrix(
    similarity: torch.Tensor,
    alignment: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """Save a readable generated-row versus GT-column similarity heatmap."""
    values = similarity.detach().float().cpu().clamp(-1.0, 1.0)
    normalized = ((values + 1.0) * 0.5).numpy()
    red = (255.0 * normalized).astype(np.uint8)
    blue = (255.0 * (1.0 - normalized)).astype(np.uint8)
    green = (80.0 + 120.0 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
    heatmap = Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")
    cell_size = max(6, 420 // max(1, values.shape[0]))
    heatmap = heatmap.resize(
        (values.shape[1] * cell_size, values.shape[0] * cell_size),
        resample=Image.Resampling.NEAREST,
    )
    margin_left, margin_top, margin_right, margin_bottom = 58, 34, 18, 38
    canvas = Image.new(
        "RGB",
        (
            margin_left + heatmap.width + margin_right,
            margin_top + heatmap.height + margin_bottom,
        ),
        (255, 255, 255),
    )
    canvas.paste(heatmap, (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(20, 24, 31))
    draw.text((margin_left + heatmap.width // 2 - 22, margin_top + heatmap.height + 18), "GT slot", fill=(20, 24, 31))
    draw.text((5, margin_top + heatmap.height // 2), "Pred", fill=(20, 24, 31))
    frame_count = values.shape[0]
    for tick in sorted({0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1}):
        x = margin_left + tick * cell_size + cell_size // 2
        y = margin_top + tick * cell_size + cell_size // 2
        draw.text((x - 5, margin_top + heatmap.height + 3), str(tick), fill=(20, 24, 31))
        draw.text((margin_left - 24, y - 6), str(tick), fill=(20, 24, 31))
    points = [
        (
            margin_left + int(slot) * cell_size + cell_size // 2,
            margin_top + row * cell_size + cell_size // 2,
        )
        for row, slot in enumerate(alignment.detach().cpu().tolist())
    ]
    if len(points) > 1:
        draw.line(points, fill=(20, 245, 80), width=max(1, cell_size // 3))
    canvas.save(output_path)


def rendered_role_ids(
    value: torch.Tensor,
    render_mode: str,
    role_mask_palette: Optional[Dict[Any, Any]] = None,
) -> torch.Tensor:
    """Convert rendered RoleMask RGB frames into stable integer role IDs."""
    if render_mode == "binary":
        return (value.mean(dim=2) >= 0.5).long()

    configured_palette = role_mask_palette or {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        4: [0, 0, 255],
    }
    palette_by_id = {int(role_id): color for role_id, color in configured_palette.items()}
    role_ids = sorted(role_id for role_id in palette_by_id if role_id in {0, 1, 2, 4})
    if role_ids != [0, 1, 2, 4]:
        raise ValueError(f"Color role-mask evaluation requires palette entries 0,1,2,4; got {role_ids}")
    palette = value.new_tensor([palette_by_id[role_id] for role_id in role_ids]) / 255.0
    role_id_tensor = value.new_tensor(role_ids, dtype=torch.long)
    channels_last = value.permute(0, 1, 3, 4, 2)
    distance = (channels_last.unsqueeze(-2) - palette).pow(2).sum(dim=-1)
    return role_id_tensor[distance.argmin(dim=-1)]


def role_region_pixel_metrics(
    gt_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    gt_role: torch.Tensor,
    render_mode: str,
    role_mask_palette: Optional[Dict[Any, Any]] = None,
    tail_condition_frames: int = 0,
    hard_clamped_tail_frames: Optional[int] = None,
) -> Dict[str, float]:
    """Measure RGB fidelity only where task-role labels say manipulation occurs."""
    frame_count = min(gt_rgb.shape[1], pred_rgb.shape[1], gt_role.shape[1])
    if hard_clamped_tail_frames is None:
        hard_clamped_tail_frames = tail_condition_frames
    generated_end = frame_count - max(
        0,
        min(int(hard_clamped_tail_frames), frame_count - 1),
    )
    if generated_end <= 1:
        return {}
    error = (pred_rgb[:, 1:generated_end] - gt_rgb[:, 1:generated_end]).pow(2).mean(dim=2)
    role_ids = rendered_role_ids(
        gt_role[:, :frame_count],
        render_mode,
        role_mask_palette=role_mask_palette,
    )[:, 1:generated_end]

    metrics: Dict[str, float] = {}
    regions = {"foreground": role_ids != 0}
    if render_mode != "binary":
        regions.update(
            {
                "active": role_ids == 1,
                "target": role_ids == 2,
                "robot": role_ids == 4,
            }
        )
    for name, region in regions.items():
        count = region.sum()
        if int(count) == 0:
            continue
        mse = (error * region.float()).sum() / count.float()
        value = float(mse.cpu())
        metrics[f"rgb_role_{name}_mse"] = value
        metrics[f"rgb_role_{name}_psnr"] = psnr_from_mse(value)
    return metrics


def role_trajectory_metrics(
    gt_role: torch.Tensor,
    pred_role: torch.Tensor,
    role_names: Dict[int, str],
    view_layout: str = "single",
    view_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compare per-view centroids and motion without mixing independent cameras."""
    if gt_role.shape != pred_role.shape:
        raise ValueError(
            f"GT/pred RoleMask ID tensors must match, got {gt_role.shape} and {pred_role.shape}"
        )
    names = list(view_names or ["view_0"])
    if view_layout == "single":
        if len(names) != 1:
            raise ValueError("view_layout='single' requires exactly one view name")
        views = [("", gt_role, pred_role)]
    else:
        if view_layout not in {"vertical", "horizontal"}:
            raise ValueError(f"Unsupported view_layout={view_layout!r}")
        if len(names) < 2:
            raise ValueError(f"view_layout={view_layout!r} requires at least two views")
        split_dim = -2 if view_layout == "vertical" else -1
        split_size = gt_role.shape[split_dim]
        if split_size % len(names) != 0:
            raise ValueError(
                f"RoleMask {view_layout} dimension {split_size} is not divisible by {len(names)} views"
            )
        gt_views = torch.chunk(gt_role, len(names), dim=split_dim)
        pred_views = torch.chunk(pred_role, len(names), dim=split_dim)
        views = []
        for index, (name, gt_view, pred_view) in enumerate(zip(names, gt_views, pred_views)):
            clean_name = re.sub(r"[^A-Za-z0-9]+", "_", str(name).split(".")[-1]).strip("_")
            clean_name = clean_name or f"view_{index}"
            views.append((clean_name, gt_view, pred_view))

    def centroid(mask: torch.Tensor, height: int, width: int) -> Optional[torch.Tensor]:
        points = mask.nonzero(as_tuple=False)
        if points.numel() == 0:
            return None
        y = points[:, 0].float().mean() / max(height - 1, 1)
        x = points[:, 1].float().mean() / max(width - 1, 1)
        return torch.stack([x, y])

    metrics: Dict[str, float] = {}
    macro_values: Dict[tuple[str, str], List[float]] = {}
    for view_name, gt_view, pred_view in views:
        height, width = gt_view.shape[-2:]
        for role_id, role_name in role_names.items():
            position_errors = []
            motion_errors = []
            for batch_idx in range(gt_view.shape[0]):
                gt_centroids = []
                pred_centroids = []
                for frame_idx in range(gt_view.shape[1]):
                    gt_centroid = centroid(
                        gt_view[batch_idx, frame_idx] == role_id, height, width
                    )
                    pred_centroid = centroid(
                        pred_view[batch_idx, frame_idx] == role_id, height, width
                    )
                    gt_centroids.append(gt_centroid)
                    pred_centroids.append(pred_centroid)
                    if gt_centroid is not None:
                        position_errors.append(
                            torch.linalg.vector_norm(pred_centroid - gt_centroid)
                            if pred_centroid is not None
                            else gt_view.new_tensor(math.sqrt(2.0), dtype=torch.float32)
                        )
                for frame_idx in range(1, len(gt_centroids)):
                    gt_prev, gt_now = gt_centroids[frame_idx - 1], gt_centroids[frame_idx]
                    pred_prev, pred_now = pred_centroids[frame_idx - 1], pred_centroids[frame_idx]
                    if gt_prev is None or gt_now is None:
                        continue
                    if pred_prev is None or pred_now is None:
                        motion_errors.append(
                            gt_view.new_tensor(math.sqrt(2.0), dtype=torch.float32)
                        )
                    else:
                        motion_errors.append(
                            torch.linalg.vector_norm(
                                (pred_now - pred_prev) - (gt_now - gt_prev)
                            )
                        )
            suffix = f"{view_name}_" if view_name else ""
            if position_errors:
                value = float(torch.stack(position_errors).mean().cpu())
                metrics[f"role_mask_{role_name}_{suffix}centroid_l2"] = value
                macro_values.setdefault((role_name, "centroid_l2"), []).append(value)
            if motion_errors:
                value = float(torch.stack(motion_errors).mean().cpu())
                metrics[f"role_mask_{role_name}_{suffix}motion_l2"] = value
                macro_values.setdefault((role_name, "motion_l2"), []).append(value)

    if len(views) > 1:
        for (role_name, metric_name), values in macro_values.items():
            metrics[f"role_mask_{role_name}_macro_{metric_name}"] = float(np.mean(values))
    return metrics


def role_mask_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    render_mode: str,
    role_mask_palette: Optional[Dict[Any, Any]] = None,
    tail_condition_frames: int = 0,
    view_layout: str = "single",
    view_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Measure generated role masks without letting conditioned endpoints inflate the result."""
    frame_count = min(gt.shape[1], pred.shape[1])
    gt = gt[:, :frame_count]
    pred = pred[:, :frame_count]
    tail_condition_frames = max(0, min(int(tail_condition_frames), frame_count - 1))
    generated_end = frame_count - tail_condition_frames
    if generated_end <= 1:
        raise ValueError(
            f"Role-mask metrics need at least one generated frame; got frame_count={frame_count}, "
            f"tail_condition_frames={tail_condition_frames}"
        )

    def foreground_metrics(gt_foreground: torch.Tensor, pred_foreground: torch.Tensor) -> Dict[str, float]:
        intersection = (gt_foreground & pred_foreground).sum().float()
        gt_count = gt_foreground.sum().float()
        pred_count = pred_foreground.sum().float()
        union = (gt_foreground | pred_foreground).sum().float()
        return {
            "role_mask_foreground_iou": float((intersection / union.clamp_min(1.0)).cpu()),
            "role_mask_foreground_f1": float((2 * intersection / (gt_count + pred_count).clamp_min(1.0)).cpu()),
            "role_mask_foreground_precision": float((intersection / pred_count.clamp_min(1.0)).cpu()),
            "role_mask_foreground_recall": float((intersection / gt_count.clamp_min(1.0)).cpu()),
        }

    def add_full_metrics(metrics: Dict[str, float], full_metrics: Dict[str, float]) -> Dict[str, float]:
        metrics.update(
            {
                f"role_mask_full_{key.removeprefix('role_mask_')}": value
                for key, value in full_metrics.items()
            }
        )
        return metrics

    if render_mode == "binary":
        gt_foreground_full = rendered_role_ids(gt, render_mode) != 0
        pred_foreground_full = rendered_role_ids(pred, render_mode) != 0
        full_metrics = foreground_metrics(gt_foreground_full, pred_foreground_full)
        gt_foreground = gt_foreground_full[:, 1:generated_end]
        pred_foreground = pred_foreground_full[:, 1:generated_end]
        metrics = foreground_metrics(gt_foreground, pred_foreground)
        metrics.update(
            {
                "role_mask_iou": metrics["role_mask_foreground_iou"],
                "role_mask_f1": metrics["role_mask_foreground_f1"],
                "role_mask_precision": metrics["role_mask_foreground_precision"],
                "role_mask_recall": metrics["role_mask_foreground_recall"],
            }
        )
        metrics.update(
            role_trajectory_metrics(
                gt_foreground.long(),
                pred_foreground.long(),
                {1: "foreground"},
                view_layout=view_layout,
                view_names=view_names,
            )
        )
        return add_full_metrics(metrics, full_metrics)

    gt_role_full = rendered_role_ids(gt, render_mode, role_mask_palette=role_mask_palette)
    pred_role_full = rendered_role_ids(pred, render_mode, role_mask_palette=role_mask_palette)
    full_metrics = foreground_metrics(gt_role_full != 0, pred_role_full != 0)
    gt_role = gt_role_full[:, 1:generated_end]
    pred_role = pred_role_full[:, 1:generated_end]
    metrics = foreground_metrics(gt_role != 0, pred_role != 0)
    role_names = {1: "active", 2: "target", 4: "robot"}
    ious = []
    for role_index, role_name in role_names.items():
        gt_current = gt_role == role_index
        pred_current = pred_role == role_index
        intersection = (gt_current & pred_current).sum().float()
        union = (gt_current | pred_current).sum().float()
        if float(union) > 0:
            iou = intersection / union
            ious.append(iou)
            metrics[f"role_mask_{role_name}_iou"] = float(iou.cpu())
    metrics["role_mask_macro_iou"] = float(torch.stack(ious).mean().cpu()) if ious else 0.0
    metrics.update(
        role_trajectory_metrics(
            gt_role,
            pred_role,
            role_names,
            view_layout=view_layout,
            view_names=view_names,
        )
    )
    return add_full_metrics(metrics, full_metrics)


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
            first_view_frames = batch.get("first_view_frames")
            view_video_frames = batch.get("view_video_frames")
            if first_view_frames is not None:
                first_view_frames = first_view_frames.to(model.device, dtype=model.dtype)
            if view_video_frames is not None:
                view_video_frames = view_video_frames.to(model.device, dtype=model.dtype)
            first_role_mask = batch.get("first_role_mask")
            role_mask_frames = batch.get("role_mask_frames")
            if first_role_mask is not None:
                first_role_mask = first_role_mask.to(model.device, dtype=model.dtype)
            if role_mask_frames is not None:
                role_mask_frames = role_mask_frames.to(model.device, dtype=model.dtype)
            language_embeddings = batch["language_embedding"]
            if language_embeddings is not None:
                language_embeddings = language_embeddings.to(model.device, dtype=model.dtype)
            first_state = batch.get("first_state")
            last_state = batch.get("last_state")
            if first_state is not None:
                first_state = first_state.to(model.device, dtype=model.dtype)
            if last_state is not None:
                last_state = last_state.to(model.device, dtype=model.dtype)
            loss_dict = model.training_step(
                first_frame=first_frame,
                video_frames=video_frames,
                first_view_frames=first_view_frames,
                view_video_frames=view_video_frames,
                first_role_mask=first_role_mask,
                role_mask_frames=role_mask_frames,
                language_embeddings=language_embeddings,
                first_state=first_state,
                last_state=last_state,
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
    role_mask_render_mode: str = "binary",
    role_mask_palette: Optional[Dict[Any, Any]] = None,
    view_layout: str = "single",
    view_names: Optional[List[str]] = None,
    dino_encoder: Optional[DinoFrameEncoder] = None,
    lpips_metric: Optional[LPIPSMetric] = None,
) -> Dict[str, float]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    sample_metric_rows: Dict[str, Dict[str, float]] = {}
    sample_index = 0

    for batch_idx, batch in enumerate(batched(windows, batch_size)):
        first_frame = batch["first_frame"].to(model.device, dtype=model.dtype)
        video_frames = batch["video_frames"].to(model.device, dtype=model.dtype)
        first_view_frames = batch.get("first_view_frames")
        view_video_frames = batch.get("view_video_frames")
        if first_view_frames is not None:
            first_view_frames = first_view_frames.to(model.device, dtype=model.dtype)
        if view_video_frames is not None:
            view_video_frames = view_video_frames.to(model.device, dtype=model.dtype)
        tail_n = int(model.config.tail_condition_frames)
        hard_clamped_tail_n = tail_n if model.config.conditioning_mode == "v0" else 0
        last_frame = video_frames[:, -1] if tail_n > 0 else None
        tail_frames = video_frames[:, -tail_n:] if tail_n > 0 else None
        first_role_mask = batch.get("first_role_mask")
        role_mask_frames = batch.get("role_mask_frames")
        if first_role_mask is not None:
            first_role_mask = first_role_mask.to(model.device, dtype=model.dtype)
        if role_mask_frames is not None:
            role_mask_frames = role_mask_frames.to(model.device, dtype=model.dtype)
        use_external_role_prompt = model.requires_role_mask_prompt
        sample_first_role_mask = first_role_mask if use_external_role_prompt else None
        last_role_mask = (
            role_mask_frames[:, -1]
            if use_external_role_prompt and role_mask_frames is not None and tail_n > 0
            else None
        )
        tail_role_masks = (
            role_mask_frames[:, -tail_n:]
            if use_external_role_prompt and role_mask_frames is not None and tail_n > 0
            else None
        )
        language_embeddings = batch["language_embedding"]
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(model.device, dtype=model.dtype)
        first_state = batch.get("first_state")
        last_state = batch.get("last_state")
        if first_state is not None:
            first_state = first_state.to(model.device, dtype=model.dtype)
        if last_state is not None:
            last_state = last_state.to(model.device, dtype=model.dtype)

        generator = torch.Generator(device=model.device).manual_seed(seed + batch_idx)
        sample_output = model.sample_bridge(
            first_frame=first_frame,
            last_frame=last_frame,
            tail_frames=tail_frames,
            first_view_frames=first_view_frames,
            last_view_frames=(view_video_frames[:, -1] if view_video_frames is not None and tail_n > 0 else None),
            tail_view_frames=(view_video_frames[:, -tail_n:] if view_video_frames is not None and tail_n > 0 else None),
            first_role_mask=sample_first_role_mask,
            last_role_mask=last_role_mask,
            tail_role_masks=tail_role_masks,
            language_embeddings=language_embeddings,
            first_state=first_state,
            last_state=last_state,
            num_inference_steps=num_inference_steps,
            generator=generator,
            return_components=model.predict_role_mask or model.uses_separate_views,
        )
        if model.uses_separate_views:
            pred_rgb = sample_output["rgb_video"]
            pred_role = None
            pred_full = pred_rgb
        elif model.predict_role_mask:
            pred_rgb = sample_output["rgb_video"]
            pred_role = sample_output["role_video"]
            pred_full = torch.cat([pred_rgb, pred_role], dim=-1)
        else:
            pred_rgb = sample_output
            pred_role = None
            pred_full = pred_rgb
        gt_rgb = torch.cat([first_frame.unsqueeze(1).float(), video_frames.float()], dim=1).clamp(0, 1)
        if role_mask_frames is not None:
            gt_role = torch.cat([first_role_mask.unsqueeze(1).float(), role_mask_frames.float()], dim=1).clamp(0, 1)
            gt_full = torch.cat([gt_rgb, gt_role], dim=-1) if pred_role is not None else gt_rgb
        else:
            gt_full = gt_rgb

        for local_idx in range(pred_full.shape[0]):
            name = f"sample_{sample_index:03d}"
            if role_mask_frames is not None:
                sample_metrics = {
                    f"rgb_{key}": value
                    for key, value in pixel_metrics(
                        gt_rgb[local_idx : local_idx + 1].float(),
                        pred_rgb[local_idx : local_idx + 1].float(),
                        tail_condition_frames=model.config.tail_condition_frames,
                        hard_clamped_tail_frames=hard_clamped_tail_n,
                    ).items()
                }
                sample_metrics.update(
                    role_region_pixel_metrics(
                        gt_rgb[local_idx : local_idx + 1].float(),
                        pred_rgb[local_idx : local_idx + 1].float(),
                        gt_role[local_idx : local_idx + 1].float(),
                        role_mask_render_mode,
                        role_mask_palette=role_mask_palette,
                        tail_condition_frames=model.config.tail_condition_frames,
                        hard_clamped_tail_frames=hard_clamped_tail_n,
                    )
                )
                if pred_role is not None:
                    role_tail_condition_frames = (
                        model.config.tail_condition_frames
                        if model.role_mask_condition_mode == "legacy_endpoints"
                        else 0
                    )
                    sample_metrics.update(role_mask_metrics(
                        gt_role[local_idx : local_idx + 1].float(),
                        pred_role[local_idx : local_idx + 1].float(),
                        role_mask_render_mode,
                        role_mask_palette=role_mask_palette,
                        tail_condition_frames=role_tail_condition_frames,
                        view_layout=view_layout,
                        view_names=view_names,
                    ))
            else:
                sample_metrics = pixel_metrics(
                    gt_full[local_idx : local_idx + 1].float(),
                    pred_full[local_idx : local_idx + 1].float(),
                    tail_condition_frames=model.config.tail_condition_frames,
                    hard_clamped_tail_frames=hard_clamped_tail_n,
                )
            sample_metrics.update(
                multiview_pixel_metrics(
                    gt_rgb[local_idx : local_idx + 1].float(),
                    pred_rgb[local_idx : local_idx + 1].float(),
                    tail_condition_frames=model.config.tail_condition_frames,
                    hard_clamped_tail_frames=hard_clamped_tail_n,
                    view_layout=view_layout,
                    view_names=view_names,
                )
            )
            sample_metrics.update(
                cross_view_motion_sync_metrics(
                    gt_rgb[local_idx : local_idx + 1].float(),
                    pred_rgb[local_idx : local_idx + 1].float(),
                    view_layout=view_layout,
                    view_names=view_names,
                )
            )
            if view_layout != "single":
                gt_slot_views = split_rgb_views(gt_rgb[local_idx], view_layout, view_names)
                pred_slot_views = split_rgb_views(pred_rgb[local_idx], view_layout, view_names)
                slot_alignments = []
                slot_metric_values: Dict[str, List[float]] = {}
                for (view_name, gt_view), (pred_view_name, pred_view) in zip(
                    gt_slot_views, pred_slot_views
                ):
                    if view_name != pred_view_name:
                        raise RuntimeError(
                            f"GT/pred view names differ: {view_name!r} vs {pred_view_name!r}"
                        )
                    slot_metrics, similarity, alignment = trajectory_slot_diagnostics(
                        gt_view.float(), pred_view.float()
                    )
                    slot_alignments.append(alignment)
                    for key, value in slot_metrics.items():
                        sample_metrics[f"view_{view_name}_{key}"] = value
                        slot_metric_values.setdefault(key, []).append(value)
                    save_similarity_matrix(
                        similarity,
                        alignment,
                        sample_dir / f"{name}_{view_name}_similarity_53x53.png",
                        title=f"{name} {view_name}: generated rows vs GT columns",
                    )
                    frame_count = min(gt_view.shape[0], pred_view.shape[0])
                    generated_end = frame_count - max(
                        0,
                        min(int(hard_clamped_tail_n), frame_count - 1),
                    )
                    if lpips_metric is not None and generated_end > 1:
                        lpips_value = lpips_metric(
                            gt_view[1:generated_end], pred_view[1:generated_end]
                        )
                        sample_metrics[f"view_{view_name}_generated_lpips"] = lpips_value
                        slot_metric_values.setdefault("generated_lpips", []).append(lpips_value)
                    if dino_encoder is not None:
                        gt_dino = dino_encoder(gt_view[:frame_count])
                        pred_dino = dino_encoder(pred_view[:frame_count])
                        if generated_end > 1:
                            same_frame_cosine = (
                                gt_dino[1:generated_end] * pred_dino[1:generated_end]
                            ).sum(dim=-1).mean()
                            dino_distance = float((1.0 - same_frame_cosine).cpu())
                            sample_metrics[f"view_{view_name}_generated_dino_distance"] = dino_distance
                            slot_metric_values.setdefault(
                                "generated_dino_distance", []
                            ).append(dino_distance)
                        dino_metrics, dino_similarity, dino_alignment = (
                            trajectory_slot_diagnostics_from_features(gt_dino, pred_dino)
                        )
                        for key, value in dino_metrics.items():
                            metric_key = f"dino_{key}"
                            sample_metrics[f"view_{view_name}_{metric_key}"] = value
                            slot_metric_values.setdefault(metric_key, []).append(value)
                        save_similarity_matrix(
                            dino_similarity,
                            dino_alignment,
                            sample_dir / f"{name}_{view_name}_dino_similarity_53x53.png",
                            title=f"{name} {view_name} DINO: generated rows vs GT columns",
                        )
                for key, values in slot_metric_values.items():
                    sample_metrics[f"view_macro_{key}"] = float(np.mean(values))
                if len(slot_alignments) == 2:
                    frame_normalizer = float(max(1, slot_alignments[0].numel() - 1))
                    slot_difference = (
                        slot_alignments[0].float() - slot_alignments[1].float()
                    ).abs()
                    sample_metrics["xview_slot_sync_mae"] = float(
                        slot_difference.mean().cpu() / frame_normalizer
                    )
                    sample_metrics["xview_slot_sync_p95"] = float(
                        torch.quantile(slot_difference, 0.95).cpu() / frame_normalizer
                    )
            metric_rows.append(sample_metrics)
            sample_metric_rows[name] = sample_metrics
            # Keep the historical filenames as RGB-only outputs so human review does not
            # shrink each camera frame by horizontally appending the auxiliary RoleMask.
            save_contact_sheet(
                gt_rgb[local_idx],
                pred_rgb[local_idx],
                sample_dir / f"{name}_sheet.png",
            )
            save_comparison_video(
                gt_rgb[local_idx],
                pred_rgb[local_idx],
                sample_dir / f"{name}_gt_pred.mp4",
                fps=fps,
            )
            if view_layout != "single":
                gt_views = split_rgb_views(gt_rgb[local_idx], view_layout, view_names)
                pred_views = split_rgb_views(pred_rgb[local_idx], view_layout, view_names)
                for (view_name, gt_view), (_, pred_view) in zip(gt_views, pred_views):
                    save_contact_sheet(
                        gt_view,
                        pred_view,
                        sample_dir / f"{name}_{view_name}_sheet.png",
                    )
                    save_comparison_video(
                        gt_view,
                        pred_view,
                        sample_dir / f"{name}_{view_name}_gt_pred.mp4",
                        fps=fps,
                    )
            save_contact_sheet(
                gt_rgb[local_idx],
                pred_rgb[local_idx],
                sample_dir / f"{name}_rgb_sheet.png",
            )
            save_comparison_video(
                gt_rgb[local_idx],
                pred_rgb[local_idx],
                sample_dir / f"{name}_rgb_gt_pred.mp4",
                fps=fps,
            )
            if pred_role is not None:
                save_contact_sheet(
                    gt_role[local_idx],
                    pred_role[local_idx],
                    sample_dir / f"{name}_role_sheet.png",
                )
                save_contact_sheet(
                    gt_full[local_idx],
                    pred_full[local_idx],
                    sample_dir / f"{name}_combined_sheet.png",
                )
                save_comparison_video(
                    gt_role[local_idx],
                    pred_role[local_idx],
                    sample_dir / f"{name}_role_gt_pred.mp4",
                    fps=fps,
                )
                save_comparison_video(
                    gt_full[local_idx],
                    pred_full[local_idx],
                    sample_dir / f"{name}_combined_gt_pred.mp4",
                    fps=fps,
                )
            sample_index += 1

    with (output_dir / "sample_metrics.json").open("w") as file:
        json.dump(sample_metric_rows, file, indent=2)
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
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loss_repeats", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--windows_file",
        type=Path,
        default=None,
        help="Reuse an existing windows.json so every model receives exactly the same episodes/frames",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--dino_model_path",
        type=Path,
        default=None,
        help="Optional local Hugging Face DINO/DINOv2 directory for perceptual and 53x53 metrics",
    )
    parser.add_argument("--perceptual_batch_size", type=int, default=32)
    parser.add_argument("--enable_lpips", action="store_true")
    parser.add_argument("--lpips_network", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    if not torch.cuda.is_available():
        raise RuntimeError("eval_vgm_bridge_stage1.py requires CUDA")

    torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config_with_base(args.config)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_step = parse_step(checkpoint_path)
    run_name = checkpoint_path.parent.name if checkpoint_path.parent.name else "checkpoint"
    output_dir = Path(args.output_dir) / run_name / (f"step_{checkpoint_step}" if checkpoint_step else checkpoint_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataset...")
    dataset = build_dataset(config, max_episodes=args.max_episodes)
    if args.windows_file is not None:
        windows = fixed_windows_from_manifest(
            dataset,
            args.windows_file.expanduser().resolve(),
            num_samples=args.num_samples,
        )
    else:
        windows = fixed_windows(dataset, args.num_samples or 16)
    records = window_records(windows)
    with (output_dir / "windows.json").open("w") as file:
        json.dump(records, file, indent=2)

    logger.info("Building model...")
    model = build_model(config, checkpoint_path)
    model.eval()
    dino_encoder = None
    if args.dino_model_path is not None:
        dino_path = args.dino_model_path.expanduser().resolve()
        if not dino_path.exists():
            raise FileNotFoundError(f"DINO model path does not exist: {dino_path}")
        logger.info("Loading local DINO feature model from %s", dino_path)
        dino_encoder = DinoFrameEncoder(
            dino_path,
            model.device,
            batch_size=args.perceptual_batch_size,
        )
    lpips_metric = None
    if args.enable_lpips:
        logger.info("Loading LPIPS network %s", args.lpips_network)
        lpips_metric = LPIPSMetric(
            model.device,
            network=args.lpips_network,
            batch_size=args.perceptual_batch_size,
        )

    logger.info("Evaluating fixed-window training loss...")
    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "num_samples": len(windows),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "windows_fingerprint": window_fingerprint(records),
        "windows_source": str(args.windows_file) if args.windows_file is not None else None,
        "loss_repeats": args.loss_repeats,
        "num_inference_steps": args.num_inference_steps,
        "dino_model_path": str(args.dino_model_path) if args.dino_model_path is not None else None,
        "lpips_enabled": bool(args.enable_lpips),
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
            role_mask_render_mode=config.dataset.get("role_mask_render_mode", "binary"),
            role_mask_palette=config.dataset.get("role_mask_palette", None),
            view_layout=str(config.dataset.get("view_layout", "single")),
            view_names=[str(name) for name in config.dataset.get(
                "view_names",
                config.dataset.get("image_columns", [config.dataset.get("image_column", "image")]),
            )],
            dino_encoder=dino_encoder,
            lpips_metric=lpips_metric,
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
