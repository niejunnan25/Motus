#!/usr/bin/env python3
"""Precompute one VGM trajectory memory and all single-frame queries per episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from data.progress.progress_cache_dataset import SCHEMA_VERSION
from train.eval_vgm_bridge_stage1 import build_dataset, build_model


logger = logging.getLogger(__name__)


def setup_logging(rank: int, level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=f"%(asctime)s - rank={rank} - %(levelname)s - %(message)s",
    )


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha1(f"{base_seed}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def tensor_cache_dtype(name: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in choices:
        raise ValueError(
            f"Unsupported cache dtype={name!r}; expected one of {sorted(choices)}"
        )
    return choices[name]


def frame_indices(total_frames: int, stride: int) -> List[int]:
    if total_frames < 2:
        raise ValueError(
            f"Progress requires at least two source frames, got {total_frames}"
        )
    indices = list(range(0, total_frames, max(1, int(stride))))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def split_for_episode(episode_name: str, validation_fraction: float, seed: int) -> str:
    if validation_fraction <= 0:
        return "train"
    value = stable_seed(f"split|{episode_name}", seed) / float(0xFFFFFFFF)
    return "val" if value < validation_fraction else "train"


def save_language_embedding(
    cache_dir: Path,
    language_embedding: torch.Tensor | None,
    task_index: Any,
) -> str | None:
    if language_embedding is None:
        return None
    embedding = language_embedding.detach().cpu().contiguous().to(torch.float16)
    hasher = hashlib.sha1()
    hasher.update(str(tuple(embedding.shape)).encode("utf-8"))
    hasher.update(str(embedding.dtype).encode("utf-8"))
    hasher.update(embedding.numpy().tobytes())
    content_hash = hasher.hexdigest()[:16]
    prefix = f"task_{int(task_index):06d}" if task_index is not None else "language"
    key = f"{prefix}.{content_hash}"
    path = cache_dir / "language" / f"{key}.pt"
    if not path.exists():
        atomic_torch_save(embedding, path)
    return str(path.relative_to(cache_dir))


def encode_frames_independently(
    model: Any,
    frames: torch.Tensor,
    batch_size: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    """VAE-encode every RGB frame as an independent one-frame video."""
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor [N,C,H,W], got {tuple(frames.shape)}")
    if batch_size < 1:
        raise ValueError("Frame VAE encode batch size must be positive")
    latents = []
    for start in range(0, frames.shape[0], batch_size):
        frame_batch = frames[start : start + batch_size]
        pixels = (
            frame_batch.to(device=model.device, dtype=model.dtype) * 2.0 - 1.0
        ).unsqueeze(2)
        chunk_latents = model.video_model.encode_video(pixels)
        if chunk_latents.ndim != 5 or chunk_latents.shape[2] != 1:
            raise RuntimeError(
                "Independent single-frame VAE encoding must produce [B,C,1,H,W], "
                f"got {tuple(chunk_latents.shape)}"
            )
        latents.append(chunk_latents.detach().cpu().to(cache_dtype))
    return torch.cat(latents, dim=0)


def encode_current_latents(
    model: Any,
    dataset: Any,
    episode_index: int,
    indices: List[int],
    batch_size: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("Current-frame VAE encode batch size must be positive")
    latents = []
    for start in range(0, len(indices), batch_size):
        chunk_indices = indices[start : start + batch_size]
        frames = dataset.load_episode_frames(episode_index, chunk_indices)
        latents.append(
            encode_frames_independently(model, frames, batch_size, cache_dtype)
        )
    return torch.cat(latents, dim=0)


def decode_generated_trajectory_frames(
    model: Any,
    trajectory_latent: torch.Tensor,
    expected_frames: int,
) -> torch.Tensor:
    """Decode the final VGM latent into the explicit frame sequence used as memory."""
    decoded = model.video_model.decode_video(trajectory_latent.to(model.dtype)).float()
    if decoded.ndim != 5 or decoded.shape[0] != 1:
        raise RuntimeError(
            f"Expected decoded trajectory [1,C,F,H,W], got {tuple(decoded.shape)}"
        )
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(0, 2, 1, 3, 4)
    if decoded.shape[1] != expected_frames:
        raise RuntimeError(
            f"VGM decoded {decoded.shape[1]} frames but Progress requires {expected_frames} bins"
        )
    return decoded[0].contiguous()


def cache_episode(
    *,
    model: Any,
    dataset: Any,
    episode_index: int,
    cache_dir: Path,
    config: Any,
    overwrite: bool,
) -> Dict[str, Any]:
    metadata = dataset.get_episode_metadata(episode_index)
    episode_name = str(metadata["episode_name"])
    episode_hash = hashlib.sha1(
        f"{metadata.get('root')}|{episode_name}".encode("utf-8")
    ).hexdigest()[:16]
    safe_name = episode_name.replace("/", "__")
    cache_path = cache_dir / "episodes" / f"{safe_name}.{episode_hash}.pt"

    total_frames = dataset.get_episode_frame_count(episode_index)
    indices = frame_indices(
        total_frames, int(config.cache.get("current_frame_stride", 1))
    )
    episode_seed = stable_seed(episode_name, int(config.cache.get("seed", 0)))
    random_state = random.getstate()
    random.seed(episode_seed)
    try:
        language_embedding = dataset.load_episode_language_embedding(episode_index)
    finally:
        random.setstate(random_state)
    language_file = save_language_embedding(
        cache_dir,
        language_embedding,
        metadata.get("task_index"),
    )

    vgm_config_path = str(Path(config.source.vgm_config).resolve())
    current_frame_stride = int(config.cache.get("current_frame_stride", 1))
    cache_dtype_name = str(config.cache.get("dtype", "float16"))
    num_progress_bins = int(config.progress_model.get("num_progress_bins", 53))
    trajectory_frame_encode_batch_size = int(
        config.cache.get("trajectory_frame_encode_batch_size", 16)
    )

    if not cache_path.exists() or overwrite:
        endpoint_frames = dataset.load_episode_frames(
            episode_index, [0, total_frames - 1]
        )
        first_frame = endpoint_frames[0:1]
        last_frame = endpoint_frames[1:2]
        generator = torch.Generator(device=model.device).manual_seed(episode_seed)
        language_batch = (
            language_embedding.unsqueeze(0) if language_embedding is not None else None
        )

        trajectory_latent = model.sample_bridge(
            first_frame=first_frame,
            last_frame=last_frame,
            language_embeddings=language_batch,
            num_inference_steps=int(config.cache.num_inference_steps),
            generator=generator,
            return_latent=True,
        )
        cache_dtype = tensor_cache_dtype(cache_dtype_name)
        trajectory_frames = decode_generated_trajectory_frames(
            model,
            trajectory_latent,
            num_progress_bins,
        )
        trajectory_frame_latents = encode_frames_independently(
            model,
            trajectory_frames,
            trajectory_frame_encode_batch_size,
            cache_dtype,
        )
        del trajectory_frames
        current_latents = encode_current_latents(
            model,
            dataset,
            episode_index,
            indices,
            int(config.cache.get("current_encode_batch_size", 16)),
            cache_dtype,
        )
        progress = torch.tensor(indices, dtype=torch.float32) / float(total_frames - 1)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_name": episode_name,
            "episode_index": metadata.get("episode_index", episode_index),
            "task_index": metadata.get("task_index"),
            "total_frames": total_frames,
            "num_progress_bins": num_progress_bins,
            "trajectory_latent": trajectory_latent[0].detach().cpu().to(cache_dtype),
            "trajectory_frame_latents": trajectory_frame_latents,
            "current_latents": current_latents,
            "progress": progress,
            "frame_indices": torch.tensor(indices, dtype=torch.long),
            "first_frame": (first_frame[0].clamp(0, 1) * 255.0).round().to(torch.uint8),
            "last_frame": (last_frame[0].clamp(0, 1) * 255.0).round().to(torch.uint8),
            "language_file": language_file,
            "vgm_config": vgm_config_path,
            "vgm_checkpoint": str(config.source.vgm_checkpoint),
            "num_inference_steps": int(config.cache.num_inference_steps),
            "current_frame_stride": current_frame_stride,
            "trajectory_frame_encode_batch_size": trajectory_frame_encode_batch_size,
            "trajectory_frame_encoding": "independent_single_frame_vae",
            "cache_dtype": cache_dtype_name,
            "seed": episode_seed,
        }
        atomic_torch_save(payload, cache_path)
    else:
        try:
            existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            existing = torch.load(cache_path, map_location="cpu")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "vgm_config": vgm_config_path,
            "vgm_checkpoint": str(config.source.vgm_checkpoint),
            "num_inference_steps": int(config.cache.num_inference_steps),
            "current_frame_stride": current_frame_stride,
            "trajectory_frame_encode_batch_size": trajectory_frame_encode_batch_size,
            "trajectory_frame_encoding": "independent_single_frame_vae",
            "cache_dtype": cache_dtype_name,
            "language_file": language_file,
            "total_frames": total_frames,
            "num_progress_bins": num_progress_bins,
            "seed": episode_seed,
        }
        mismatched = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"Existing cache {cache_path} was created with different settings: {mismatched}. "
                "Use --overwrite or a new cache directory."
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_name": episode_name,
        "episode_index": metadata.get("episode_index", episode_index),
        "task_index": metadata.get("task_index"),
        "total_frames": total_frames,
        "num_progress_bins": num_progress_bins,
        "num_queries": len(indices),
        "split": split_for_episode(
            episode_name,
            float(config.cache.get("validation_fraction", 0.1)),
            int(config.cache.get("seed", 0)),
        ),
        "cache_file": str(cache_path.relative_to(cache_dir)),
        "language_file": language_file,
    }


def write_rank_manifest(
    cache_dir: Path, rank: int, entries: Iterable[Dict[str, Any]]
) -> Path:
    path = cache_dir / f"manifest.rank_{rank:03d}.jsonl"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def merge_manifests(cache_dir: Path) -> None:
    merged: Dict[str, Dict[str, Any]] = {}
    existing = cache_dir / "manifest.jsonl"
    sources = ([existing] if existing.exists() else []) + sorted(
        cache_dir.glob("manifest.rank_*.jsonl")
    )
    for source in sources:
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    entry = json.loads(line)
                    merged[str(entry["cache_file"])] = entry
    entries = sorted(merged.values(), key=lambda item: str(item["episode_name"]))
    if not entries:
        raise RuntimeError(f"No episode entries were produced under {cache_dir}")
    progress_bin_counts = {int(entry["num_progress_bins"]) for entry in entries}
    if len(progress_bin_counts) != 1:
        raise RuntimeError(
            f"Manifest mixes incompatible Progress bin counts: {progress_bin_counts}"
        )
    temporary = existing.with_name(f".{existing.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, sort_keys=True) + "\n")
    os.replace(temporary, existing)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "num_progress_bins": next(iter(progress_bin_counts)),
        "episodes": len(entries),
        "train_episodes": sum(entry["split"] == "train" for entry in entries),
        "val_episodes": sum(entry["split"] == "val" for entry in entries),
        "queries": sum(int(entry["num_queries"]) for entry in entries),
    }
    with (cache_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Progress Stage 2 YAML")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--cache_dir", default=None, help="Override cache.cache_dir")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override cache.num_inference_steps (useful for isolated smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    rank, world_size, local_rank = distributed_context()
    setup_logging(rank, args.log_level)
    if not torch.cuda.is_available():
        raise RuntimeError("VGM Progress cache generation requires CUDA")
    torch.cuda.set_device(local_rank)

    config = OmegaConf.load(args.config)
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    if args.num_inference_steps is not None:
        if args.num_inference_steps < 1:
            raise ValueError("--num_inference_steps must be positive")
        config.cache.num_inference_steps = args.num_inference_steps
    vgm_config = OmegaConf.load(config.source.vgm_config)
    if (
        vgm_config.dataset.get("bridge_sampling_mode", "sliding_window")
        != "full_episode_uniform"
    ):
        raise ValueError(
            "Progress cache currently requires a full_episode_uniform VGM source config"
        )
    if vgm_config.common.get("role_mask_fusion_mode", "none") != "none":
        raise NotImplementedError(
            "Initial Progress experiments require an RGB-only V1-proper checkpoint"
        )
    num_progress_bins = int(config.progress_model.get("num_progress_bins", 53))
    generated_frames = int(vgm_config.common.num_video_frames) + 1
    if generated_frames != num_progress_bins:
        raise ValueError(
            "The Progress class count must equal the decoded VGM trajectory length: "
            f"source generates {generated_frames} frames, config requests {num_progress_bins} bins"
        )

    cache_dir = Path(config.cache.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for stale_manifest in cache_dir.glob("manifest.rank_*.jsonl"):
            stale_manifest.unlink()
        OmegaConf.save(config, cache_dir / "cache_config.yaml", resolve=True)
    if world_size > 1:
        dist.barrier()
    dataset = build_dataset(vgm_config, max_episodes=args.max_episodes)
    model = build_model(vgm_config, Path(config.source.vgm_checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    episode_indices = list(range(len(dataset.episodes)))[rank::world_size]
    entries = []
    for position, episode_index in enumerate(episode_indices, start=1):
        try:
            entry = cache_episode(
                model=model,
                dataset=dataset,
                episode_index=episode_index,
                cache_dir=cache_dir,
                config=config,
                overwrite=args.overwrite,
            )
            entries.append(entry)
            logger.info(
                "Cached %s/%s: %s (%s queries)",
                position,
                len(episode_indices),
                entry["episode_name"],
                entry["num_queries"],
            )
        except Exception:
            logger.exception("Failed caching episode index=%s", episode_index)
            raise

    write_rank_manifest(cache_dir, rank, entries)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_manifests(cache_dir)
        logger.info("Progress cache complete: %s", cache_dir)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
