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

# 维度约定：N=一个 episode 中的查询帧数，F=生成轨迹帧数（正式配置为 53），
# C_img=RGB 通道数（3），C_z=VAE latent 通道数（正式配置为 48），
# T_z/H_z/W_z=latent 的时间/空间尺寸，H_img/W_img=输入图像尺寸。

# Progress Stage 2 审阅边界：
# - 首次引入 53-frame Progress 训练的基线为分支 codex/vgm-bridge-stage1、
#   commit e026a6b（Add 53-frame trajectory progress training）。
# - 讨论中的 z_trajectory 在代码里统一命名为 trajectory_latent。
# - V1-proper VGM 先用 first + goal + language 做 50 步采样，只生成一次轨迹；
#   Stage 2 训练不会为每一张在线观测重复执行这 50 步采样。
#
# 本文件负责的完整数据流：
# first/goal/language
# -> trajectory_latent [1,C_z,T_z=14,H_z,W_z]
# -> Wan VAE decode -> 53 张 RGB 图 [F=53,C_img,H_img,W_img]
# -> 每张图单独 Wan VAE encode -> [F=53,C_z,1,H_z,W_z]
# -> 真实 episode 的每张查询图也单独 encode -> [N,C_z,1,H_z,W_z]
# -> 连同 progress [N] 一起写入一次性缓存，供两种 Progress 模型复用。


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
    # 输入 frames: [N, C_img, H_img, W_img]。这里故意不把 N 帧作为连续视频编码，
    # 而是让每张图片都成为独立的单帧视频，从而得到 N 个可单独寻址的 Progress slot。
    # 对生成轨迹调用时 N=F=53；对真实 episode 调用时 N=该 episode 选中的查询帧数。
    # 这里只检查 rank=4，避免把 [B,F,C,H,W] 整段视频误传进来。
    if frames.ndim != 4:
        raise ValueError(f"Expected frame tensor [N,C,H,W], got {tuple(frames.shape)}")
    # batch_size 只是 VAE 编码时的 micro-batch 大小，不改变 N 个帧各自独立编码的语义。
    if batch_size < 1:
        raise ValueError("Frame VAE encode batch size must be positive")
    # latents 中每一项稍后都是一个 CPU tensor: [B_e, C_z, 1, H_z, W_z]。
    latents = []
    # start 依次取 0, B_e, 2*B_e...；最后一个 chunk 可以小于 batch_size。
    for start in range(0, frames.shape[0], batch_size):
        # frame_batch: [B_e, C_img, H_img, W_img]，B_e 是 VAE 编码 micro-batch。
        frame_batch = frames[start : start + batch_size]
        # 仅改变设备和精度，形状仍为 [B_e, C_img, H_img, W_img]。
        frame_batch = frame_batch.to(device=model.device, dtype=model.dtype)
        # Wan VAE 接收 [-1,1] 像素；数值范围由 [0,1] 线性映射到 [-1,1]，形状不变。
        pixels = frame_batch * 2.0 - 1.0
        # 在 dim=2 插入长度为 1 的时间维：
        # [B_e, C_img, H_img, W_img] -> [B_e, C_img, 1, H_img, W_img]。
        pixels = pixels.unsqueeze(2)
        # chunk_latents: [B_e, C_z, 1, H_z, W_z]。
        # 每个输入样本只有一帧，因此每个输出样本必须也只有一个 temporal latent slice。
        chunk_latents = model.video_model.encode_video(pixels)
        # 同时检查 rank=5 和 T_z=1，防止 VAE/API 改动后悄悄破坏 53-slot 对齐定义。
        if chunk_latents.ndim != 5 or chunk_latents.shape[2] != 1:
            raise RuntimeError(
                "Independent single-frame VAE encoding must produce [B,C,1,H,W], "
                f"got {tuple(chunk_latents.shape)}"
            )
        # 缓存不参与 VAE 反向传播：detach 去掉计算图，cpu 搬出显存，to 转成缓存精度。
        # 形状始终保持 [B_e, C_z, 1, H_z, W_z]。
        cached_chunk = chunk_latents.detach().cpu().to(cache_dtype)
        latents.append(cached_chunk)
    # 沿帧/样本维 dim=0 拼回原顺序：若生成轨迹 N=53，则返回 [53,C_z,1,H_z,W_z]。
    return torch.cat(latents, dim=0)


def encode_current_latents(
    model: Any,
    dataset: Any,
    episode_index: int,
    indices: List[int],
    batch_size: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    # indices 长度就是 N；每个元素是当前 episode 中一张真实 source frame 的绝对下标。
    if batch_size < 1:
        raise ValueError("Current-frame VAE encode batch size must be positive")
    # 每个列表元素稍后为 [B_e,C_z,1,H_z,W_z]，最终沿 dim=0 拼成 N 张查询。
    latents = []
    # 真实帧也按 micro-batch 读取和编码，避免一次把完整 episode 塞进显存。
    for start in range(0, len(indices), batch_size):
        # chunk_indices 长度为 B_e（最后一组可能更短），不改变原 episode 顺序。
        chunk_indices = indices[start : start + batch_size]
        # frames: [B_e, C_img, H_img, W_img]，来自真实 source episode。
        frames = dataset.load_episode_frames(episode_index, chunk_indices)
        # 调用与 53 张生成帧完全相同的单帧 VAE 路径，得到 [B_e,C_z,1,H_z,W_z]。
        latents.append(
            encode_frames_independently(model, frames, batch_size, cache_dtype)
        )
    # 沿查询帧维 dim=0 拼接，返回全部在线查询: [N,C_z,1,H_z,W_z]。
    return torch.cat(latents, dim=0)


def decode_generated_trajectory_frames(
    model: Any,
    trajectory_latent: torch.Tensor,
    expected_frames: int,
) -> torch.Tensor:
    """Decode the final VGM latent into the explicit frame sequence used as memory."""
    # 关键设计：这里没有把 14 个 temporal latent slice 插值成 53 个分类位置。
    # 14 个 slice 是 Wan VAE 的时间压缩表示，并不天然等价于 14 张可匹配图片；因此先用
    # 同一个 Wan VAE 完整解码出 53 张图片，再在后续逐帧独立编码，建立 53 个显式 slot。
    # trajectory_latent: [1,C_z,T_z,H_z,W_z]；正式 53F 配置中 T_z=14。
    # 转成 VAE 精度只改变 dtype，不改变形状。
    trajectory_latent = trajectory_latent.to(model.dtype)
    # Wan VAE temporal decoder 把 14 个压缩 slice 还原为完整视频：
    # [1,C_z,T_z=14,H_z,W_z] -> [1,C_img,F=53,H_img,W_img]，像素范围约为 [-1,1]。
    decoded = model.video_model.decode_video(trajectory_latent)
    # 后续像素变换和缓存前重编码统一用 float32；形状仍为 [1,C_img,F,H_img,W_img]。
    decoded = decoded.float()
    # 必须是单 episode 的 5D 视频；这里不允许把多个 episode 混成一条 53-frame memory。
    if decoded.ndim != 5 or decoded.shape[0] != 1:
        raise RuntimeError(
            f"Expected decoded trajectory [1,C,F,H,W], got {tuple(decoded.shape)}"
        )
    # 截断 VAE 可能产生的轻微越界值；形状不变，数值严格落在 [-1,1]。
    decoded = decoded.clamp(-1.0, 1.0)
    # 线性映射到数据集统一使用的 [0,1] RGB 范围；形状仍为 [1,C_img,F,H_img,W_img]。
    decoded = (decoded + 1.0) * 0.5
    # 把时间维移到通道维之前：
    # [1,C_img,F,H_img,W_img] -> [1,F,C_img,H_img,W_img]。
    decoded = decoded.permute(0, 2, 1, 3, 4)
    # Progress 输出空间固定为 F=53；若 VAE 解出的帧数不等于配置 bin 数，立即终止缓存。
    if decoded.shape[1] != expected_frames:
        raise RuntimeError(
            f"VGM decoded {decoded.shape[1]} frames but Progress requires {expected_frames} bins"
        )
    # decoded[0] 去掉唯一的 episode batch 维；contiguous 为后续逐帧 batch 切片整理内存。
    # 返回 [F=53,C_img,H_img,W_img]，这 53 张图接下来会各自独立 VAE encode。
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
        # endpoint_frames: [2, C_img, H_img, W_img]；first/last 各为 [1, C_img, H_img, W_img]。
        endpoint_frames = dataset.load_episode_frames(
            episode_index, [0, total_frames - 1]
        )
        # 保留 batch 维，first_frame: [1,C_img,H_img,W_img]。
        first_frame = endpoint_frames[0:1]
        # 保留 batch 维，last_frame: [1,C_img,H_img,W_img]。
        last_frame = endpoint_frames[1:2]
        # 每个 episode 用稳定 seed 建立独立 CUDA RNG，使同 checkpoint 的缓存可以复现。
        generator = torch.Generator(device=model.device).manual_seed(episode_seed)
        # language_embedding 原为 [L_text,D_text]；增加 batch 后为 [1,L_text,D_text]，也允许 None。
        language_batch = (
            language_embedding.unsqueeze(0) if language_embedding is not None else None
        )

        # 最终去噪后的 z_trajectory/trajectory_latent:
        # [1, C_z, T_z, H_z, W_z]，正式 53F 配置中 T_z=14。
        # first/last 提供首尾视觉条件；language_batch 提供任务语义；generator 固定初始噪声。
        # num_inference_steps 正式配置为 50；return_latent=True 阻止内部 VAE decode。
        trajectory_latent = model.sample_bridge(
            first_frame=first_frame,
            last_frame=last_frame,
            language_embeddings=language_batch,
            num_inference_steps=int(config.cache.num_inference_steps),
            generator=generator,
            return_latent=True,
        )
        # cache_dtype 只决定落盘精度（正式为 float16），不改变上面 VGM 推理使用的精度。
        cache_dtype = tensor_cache_dtype(cache_dtype_name)
        # 先解码成显式生成轨迹: [F=53, C_img, H_img, W_img]。
        trajectory_frames = decode_generated_trajectory_frames(
            model,
            trajectory_latent,
            num_progress_bins,
        )
        # 再逐帧独立编码成 53 个 memory slot:
        # trajectory_frame_latents: [F=53, C_z, 1, H_z, W_z]。
        trajectory_frame_latents = encode_frames_independently(
            model,
            trajectory_frames,
            trajectory_frame_encode_batch_size,
            cache_dtype,
        )
        # RGB 解码结果只是建立 53 个显式 slot 的中间产物。缓存只保留逐帧 latent，
        # 不保留这批 RGB tensor，避免每个 episode 重复占用大量磁盘空间。
        del trajectory_frames
        # 真实 episode 查询帧走完全相同的单帧 VAE 路径:
        # current_latents: [N, C_z, 1, H_z, W_z]。
        current_latents = encode_current_latents(
            model,
            dataset,
            episode_index,
            indices,
            int(config.cache.get("current_encode_batch_size", 16)),
            cache_dtype,
        )
        # indices 先转为 [N] float32 tensor，再逐元素除以同一个标量 total_frames-1。
        # progress: [N]，第 i 个查询的监督值为 source_frame_index/(total_frames-1)。
        # 这是当前最重要的监督假设：标签来自 source episode 的归一化时间，而不是先计算
        # 当前真实图和 53 张生成图的视觉相似度后再选择最近帧。因此 source 进度 0.5 会被
        # 监督到生成轨迹第 26 帧附近，默认两条轨迹在时间上单调且近似线性对齐。
        progress = torch.tensor(indices, dtype=torch.float32) / float(total_frames - 1)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_name": episode_name,
            "episode_index": metadata.get("episode_index", episode_index),
            "task_index": metadata.get("task_index"),
            "total_frames": total_frames,
            "num_progress_bins": num_progress_bins,
            # 去掉 batch 维后保存: [C_z, T_z, H_z, W_z]；仅 Layerwise 分支直接使用。
            "trajectory_latent": trajectory_latent[0].detach().cpu().to(cache_dtype),
            # [F=53, C_z, 1, H_z, W_z]；Serial 与 Layerwise 都用于最终 53-bin 对齐。
            "trajectory_frame_latents": trajectory_frame_latents,
            # [N, C_z, 1, H_z, W_z]；每一项对应一张真实在线观测。
            "current_latents": current_latents,
            # [N]。
            "progress": progress,
            # [N] int64；保留原始 source frame 下标，便于评估和排查标签。
            "frame_indices": torch.tensor(indices, dtype=torch.long),
            # 去掉 batch 后各为 [C_img,H_img,W_img] uint8，仅 Layerwise 重放条件和可视化使用。
            "first_frame": (first_frame[0].clamp(0, 1) * 255.0).round().to(torch.uint8),
            "last_frame": (last_frame[0].clamp(0, 1) * 255.0).round().to(torch.uint8),
            # 字符串路径，不在每个 episode payload 中重复保存 [L_text,D_text] embedding tensor。
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
    if vgm_config.common.get("state_condition_mode", "none") != "none":
        raise NotImplementedError(
            "Progress cache generation currently requires a state-free VGM source. "
            "State-conditioned V2-B checkpoints need an explicit endpoint-state input "
            "policy and must use a separate cache pipeline."
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
