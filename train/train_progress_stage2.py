#!/usr/bin/env python3
"""Train single-frame Progress over a frozen V1-proper trajectory memory."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.progress import ProgressEpisodeCacheDataset, progress_episode_collate_fn
from models.progress_stage2 import (
    ProgressStage2Config,
    build_progress_model,
    compute_progress_loss,
)

logger = logging.getLogger(__name__)

# 维度约定：E=每卡 episode batch，E_micro=一次 forward 的 episode 数，
# N_e=episode e 的真实帧数，Q=每条 episode 采样的 query 数，B_q=sum(Q)，F=53，
# C_z/T_z/H_z/W_z=VAE latent 尺寸，K=每帧 Progress token 数，D=Progress hidden dim。


def setup_logging(rank: int, level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=f"%(asctime)s - rank={rank} - %(levelname)s - %(message)s",
    )


def normalize_report_to(value: Any) -> list[str]:
    if value is None or str(value).lower() in {"", "none"}:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def model_config_from_yaml(config: Any) -> ProgressStage2Config:
    model = config.progress_model
    patch_size = tuple(int(value) for value in model.get("patch_size", [1, 2, 2]))
    return ProgressStage2Config(
        fusion_mode=str(model.fusion_mode),
        num_progress_bins=int(model.get("num_progress_bins", 53)),
        latent_channels=int(model.get("latent_channels", 48)),
        hidden_dim=int(model.get("hidden_dim", 512)),
        num_heads=int(model.get("num_heads", 8)),
        num_layers=int(model.num_layers),
        ffn_multiplier=int(model.get("ffn_multiplier", 4)),
        patch_size=patch_size,
        frame_num_tokens=int(model.get("frame_num_tokens", 16)),
        view_encoding_mode=str(model.get("view_encoding_mode", "joint")),
        num_views=int(model.get("num_views", 1)),
        tokens_per_view=int(model.get("tokens_per_view", 16)),
        alignment_head_mode=str(
            model.get("alignment_head_mode", "pooled_cosine")
        ),
        late_interaction_temperature=float(
            model.get("late_interaction_temperature", 0.07)
        ),
        video_hidden_dim=int(model.get("video_hidden_dim", 3072)),
        dropout=float(model.get("dropout", 0.0)),
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def checkpoint_step(path: str | Path) -> int:
    match = re.search(r"step_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def save_checkpoint(
    accelerator: Accelerator,
    config: Any,
    checkpoint_root: Path,
    global_step: int,
    epoch: int,
) -> None:
    checkpoint_dir = checkpoint_root / f"checkpoint_step_{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        OmegaConf.save(config, checkpoint_dir / "config.yaml", resolve=True)
        with (checkpoint_dir / "progress_training_state.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump({"global_step": global_step, "epoch": epoch}, file, indent=2)
        logger.info("Saved Progress checkpoint: %s", checkpoint_dir)


def load_frozen_vgm(config: Any, device: torch.device):
    # Serial 只消费已经缓存好的 53 帧单帧 latent，不需要在训练时加载 5B VGM。
    if str(config.progress_model.fusion_mode) != "layerwise_wvm":
        return None
    # 延迟导入 VGM/视频数据依赖，使纯 Serial Stage 2 不必在启动时加载 decord 等模块。
    if __package__:
        from .eval_vgm_bridge_stage1 import build_model as build_vgm_model
    else:
        from eval_vgm_bridge_stage1 import build_model as build_vgm_model

    # Layerwise/WVM-style 需要重新运行冻结的 V1-proper WAN，并读取其 30 层 hidden state。
    vgm_config = OmegaConf.load(config.source.vgm_config)
    vgm = build_vgm_model(vgm_config, Path(config.source.vgm_checkpoint))
    vgm.to(device)
    vgm.eval()
    # 冻结 VGM：Stage 2 的梯度只能更新 Progress 模型，不能改动 Stage 1 视频模型。
    for parameter in vgm.parameters():
        parameter.requires_grad = False
    # 当前实现要求一一对应：第 l 个 Progress block 读取第 l 个 WAN block 的输出。
    expected_layers = int(config.progress_model.num_layers)
    actual_layers = len(vgm.video_model.wan_model.blocks)
    if expected_layers != actual_layers:
        raise ValueError(
            f"layerwise_wvm requires one Progress block per WAN block: "
            f"progress={expected_layers}, wan={actual_layers}"
        )
    return vgm


def prepare_episode_queries(
    current_latents: torch.Tensor,
    progress: torch.Tensor,
    queries_per_episode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # current_latents: [N,C_z,1,H_z,W_z]；progress: [N]；二者第 0 维严格对齐。
    count = current_latents.shape[0]
    if count < 1:
        raise ValueError("Progress episode must contain at least one current-frame query")
    # order: [N]，随机打乱同一 episode 内的真实查询帧，避免总按时间顺序训练。
    order = torch.randperm(count, device=current_latents.device)
    if queries_per_episode > 0:
        if queries_per_episode <= count:
            # 只抽取 Q=queries_per_episode 个查询，输出形状变为 [Q,...] 和 [Q]。
            order = order[:queries_per_episode]
        else:
            # Q>N 时循环复用随机排列，确保仍能返回固定 Q 个查询；不会改变标签对应关系。
            repeats = math.ceil(queries_per_episode / count)
            order = order.repeat(repeats)[:queries_per_episode]
    # 同一个 order 同时索引 latent 和标签，保证第 q 个 latent 仍对应第 q 个 progress。
    return current_latents[order], progress[order]


def prepare_episode_micro_batch(
    episodes: Sequence[Dict[str, Any]],
    queries_per_episode: int,
) -> Dict[str, Any]:
    """Stack E memories and flatten Q queries per episode into one model batch."""
    if not episodes:
        raise ValueError("Episode micro-batch must contain at least one episode")

    # trajectory_latent: E * [C_z,T_z,H_z,W_z] -> [E,C_z,T_z,H_z,W_z]。
    trajectory_latent = torch.stack(
        [episode["trajectory_latent"] for episode in episodes], dim=0
    )
    # trajectory_frame_latents: E * [F=53,C_z,1,H_z,W_z]
    # -> [E,F=53,C_z,1,H_z,W_z]。
    trajectory_frame_latents = torch.stack(
        [episode["trajectory_frame_latents"] for episode in episodes], dim=0
    )

    current_latents: List[torch.Tensor] = []
    progress_targets: List[torch.Tensor] = []
    query_episode_indices: List[torch.Tensor] = []
    query_counts: List[int] = []
    for local_episode_index, episode in enumerate(episodes):
        # 每条 episode 独立随机采样 Q 个 query；Q=0 时保留该 episode 的全部 N_e 帧。
        episode_current, episode_progress = prepare_episode_queries(
            episode["current_latents"],
            episode["progress"],
            queries_per_episode,
        )
        query_count = int(episode_current.shape[0])
        if query_count < 1:
            raise ValueError(
                f"Episode {episode.get('episode_name')} contains no Progress queries"
            )
        current_latents.append(episode_current)
        progress_targets.append(episode_progress)
        # 当前 episode 的 Q 个 query 都记录同一个 local_episode_index：
        # E=2,Q=3 时最终得到 [0,0,0,1,1,1]。
        query_episode_indices.append(
            torch.full(
                (query_count,),
                local_episode_index,
                device=episode_current.device,
                dtype=torch.long,
            )
        )
        query_counts.append(query_count)

    return {
        "trajectory_latent": trajectory_latent,
        "trajectory_frame_latents": trajectory_frame_latents,
        # [sum_e Q_e,C_z,1,H_z,W_z]；正式 mixed batch 中 Q_e 都等于配置 Q。
        "current_latents": torch.cat(current_latents, dim=0),
        # [sum_e Q_e]。
        "progress_targets": torch.cat(progress_targets, dim=0),
        # [sum_e Q_e]，显式保证每个 query 只能选择对应的 episode memory。
        "query_episode_indices": torch.cat(query_episode_indices, dim=0),
        "query_counts": query_counts,
        "episodes": list(episodes),
    }


def extract_episode_video_features(
    frozen_vgm: Any,
    config: Any,
    prepared: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Run one batched frozen-WAN feature pass for an episode micro-batch."""
    if frozen_vgm is None:
        return None
    episodes = prepared["episodes"]
    # endpoint 图：E * [C_img,H_img,W_img] -> [E,C_img,H_img,W_img]，并归一化到 [0,1]。
    first_frame = torch.stack([episode["first_frame"] for episode in episodes], dim=0)
    first_frame = first_frame.float().div(255.0)
    last_frame = torch.stack([episode["last_frame"] for episode in episodes], dim=0)
    last_frame = last_frame.float().div(255.0)
    # 每条 UMT5 embedding 是 [L_e,D_text]，L_e 会随任务文本长度变化，不能直接
    # torch.stack。保留长度 E 的 list，VGM._context_list 会逐样本 pad/truncate 到 512。
    language_embeddings = [episode["language_embedding"] for episode in episodes]
    with torch.no_grad():
        # 返回 hidden_states: 30 * [E,L_video,D_video]、grid_sizes: [E,3]。
        return frozen_vgm.extract_bridge_hidden_states(
            trajectory_latent=prepared["trajectory_latent"],
            first_frame=first_frame,
            last_frame=last_frame,
            language_embeddings=language_embeddings,
            feature_timestep=float(
                config.progress_model.get("feature_timestep", 0.0)
            ),
        )


def weighted_metrics(
    accumulator: Dict[str, float], losses: Dict[str, torch.Tensor], weight: int
) -> None:
    for name, value in losses.items():
        accumulator[name] = (
            accumulator.get(name, 0.0) + float(value.detach().float()) * weight
        )
    accumulator["examples"] = accumulator.get("examples", 0.0) + weight


def finalize_metrics(accumulator: Dict[str, float]) -> Dict[str, float]:
    examples = max(1.0, accumulator.pop("examples", 1.0))
    return {name: value / examples for name, value in accumulator.items()}


def forward_progress(
    model: torch.nn.Module,
    fusion_mode: str,
    current_latent: torch.Tensor,
    trajectory_latent: torch.Tensor,
    trajectory_frame_latents: torch.Tensor,
    video_features: Optional[Dict[str, Any]],
    query_episode_indices: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # current_latent: [B_q, C_z, 1, H_z, W_z]
    # trajectory_latent: [E_micro, C_z, T_z=14, H_z, W_z]
    # trajectory_frame_latents: [E_micro,F=53,C_z,1,H_z,W_z]
    # query_episode_indices: [B_q]，值域 [0,E_micro-1]。
    if fusion_mode == "serial_latent":
        # Serial 不直接消费 14-slice trajectory_latent，只使用显式 53 帧 memory。
        # 返回字段：alignment_logits/probabilities [B_q,53]；
        # progress/expected_frame/matched_frame [B_q]。内部 query token 不对外返回。
        outputs = model(
            current_latent=current_latent,
            trajectory_frame_latents=trajectory_frame_latents,
            query_episode_indices=query_episode_indices,
        )
        return outputs
    if fusion_mode == "layerwise_wvm":
        if video_features is None:
            raise RuntimeError("layerwise_wvm requires frozen WAN hidden states")
        # video_hidden_states 中有 30 个 [E_micro,L,D_video] WAN block 输出。
        # grid_sizes 是 [E_micro,3]，每行均为 (G_t,G_h,G_w)，且 L=G_t*G_h*G_w。
        grid_sizes = video_features["grid_sizes"]
        if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
            raise ValueError(
                f"Expected WAN grid_sizes [E,3], got {tuple(grid_sizes.shape)}"
            )
        if not torch.equal(grid_sizes, grid_sizes[:1].expand_as(grid_sizes)):
            raise ValueError("All mixed-batch WAN trajectories must share one token grid")
        video_grid_size = grid_sizes[0]
        # Layerwise 的最终输出字段和 Serial 相同，仍然是对 53 个显式帧位置做匹配。
        outputs = model(
            current_latent=current_latent,
            trajectory_frame_latents=trajectory_frame_latents,
            video_hidden_states=video_features["hidden_states"],
            video_grid_size=video_grid_size,
            query_episode_indices=query_episode_indices,
        )
        return outputs
    raise ValueError(f"Unknown fusion mode={fusion_mode!r}")


def train(config: Any, accelerator: Accelerator, resume_from: Optional[str]) -> None:
    model_config = model_config_from_yaml(config)
    # E 是每个 rank、每个 optimizer step 的 episode 数；Q 是每条 episode 的 query 数。
    # 旧配置没有 episode_batch_size 时默认 E=1，queries_per_episode=0 仍表示使用全部 N 帧。
    episode_batch_size = int(config.training.get("episode_batch_size", 1))
    episode_micro_batch_size = int(
        config.training.get("episode_micro_batch_size", episode_batch_size)
    )
    query_batch_size = int(config.training.query_batch_size)
    queries_per_episode = int(config.training.get("queries_per_episode", 0))
    if episode_batch_size < 1:
        raise ValueError("training.episode_batch_size must be positive")
    if episode_micro_batch_size < 1 or episode_micro_batch_size > episode_batch_size:
        raise ValueError(
            "training.episode_micro_batch_size must be in "
            f"[1,{episode_batch_size}]"
        )
    if query_batch_size < 1:
        raise ValueError("training.query_batch_size must be positive")
    if queries_per_episode < 0:
        raise ValueError("training.queries_per_episode cannot be negative")
    # E>1 时要求固定 Q，避免不同长度 N_e 使一个 optimizer step 的显存和 episode
    # 权重不可控。旧 E=1,Q=0 路径继续支持全 episode 训练。
    if episode_batch_size > 1 and queries_per_episode == 0:
        raise ValueError(
            "Mixed episode batches require training.queries_per_episode > 0"
        )
    # 一个 episode micro-batch 必须完整容纳 E_micro*Q 个 query，这样同一 episode
    # 的 ranking pair 不会被切散，且 53-frame memory 在该 forward 中只编码一次。
    if (
        queries_per_episode > 0
        and episode_micro_batch_size * queries_per_episode > query_batch_size
    ):
        raise ValueError(
            "episode_micro_batch_size * queries_per_episode must not exceed "
            "query_batch_size; increase query_batch_size or reduce the episode micro-batch"
        )

    model = build_progress_model(model_config)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )
    # Dataset 的一个 item 是一整条 episode：一份固定 trajectory memory + N 个真实查询帧。
    dataset = ProgressEpisodeCacheDataset(
        config.cache.cache_dir,
        split="train",
        load_language_embedding=model_config.fusion_mode == "layerwise_wvm",
        expected_num_progress_bins=model_config.num_progress_bins,
        max_episodes=config.training.get("max_episodes", None),
    )
    # DataLoader 的 batch_size=E（每个 rank）；collate 保留长度 E 的 episode list，
    # 因为每条 episode 的真实查询数 N_e 可以不同，不能直接 stack current_latents。
    dataloader = DataLoader(
        dataset,
        batch_size=episode_batch_size,
        shuffle=True,
        num_workers=int(config.system.get("num_workers", 2)),
        pin_memory=bool(config.system.get("pin_memory", True)),
        collate_fn=progress_episode_collate_fn,
        drop_last=False,
    )

    max_steps = int(config.training.max_steps)
    max_epochs = int(config.training.max_epochs)
    # Accelerate 默认不拆分单个 DataLoader batch，而是让不同 rank 消费不同 batch；
    # 因此全局一次更新最多覆盖 world_size*E 条 episode。
    steps_per_epoch = math.ceil(
        len(dataloader) / max(1, accelerator.num_processes)
    )
    planned_steps = min(max_steps, max_epochs * steps_per_epoch)
    scheduler = create_scheduler(
        optimizer,
        int(config.training.get("warmup_steps", 0)),
        planned_steps,
    )

    # 多进程 accelerate launch 下，trainable Progress model 会在这里包装成 DDP；
    # split_batches 默认关闭，因此配置的 episode_batch_size=E 是“每个 rank 的 E”。
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    if len(dataloader) != steps_per_epoch:
        raise RuntimeError(
            f"Unexpected distributed dataloader length: expected={steps_per_epoch}, "
            f"actual={len(dataloader)}"
        )
    # 冻结 VGM 不进入 DDP：每个 rank 在自己的 GPU 上持有一份只读副本，仅为本地
    # episode 提取 hidden memory。它没有梯度，因此不需要参数同步。
    frozen_vgm = load_frozen_vgm(config, accelerator.device)
    global_step = 0
    if resume_from:
        accelerator.load_state(resume_from)
        global_step = checkpoint_step(resume_from)
        logger.info(
            "Resumed Progress training at step=%s from %s", global_step, resume_from
        )

    first_epoch = global_step // steps_per_epoch
    resume_step_in_epoch = global_step % steps_per_epoch

    checkpoint_root = Path(config.system.checkpoint_dir) / str(config.logging.run_name)
    if accelerator.is_main_process:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    save_interval = int(config.system.save_interval)
    log_interval = int(config.system.get("log_interval", 1))
    loss_options = {
        "target_sigma_bins": float(config.loss.get("target_sigma_bins", 2.0)),
        "regression_weight": float(config.loss.get("regression_weight", 1.0)),
        "ranking_weight": float(config.loss.get("ranking_weight", 0.1)),
        "ranking_minimum_gap": float(config.loss.get("ranking_minimum_gap", 0.05)),
        "ranking_temperature": float(config.loss.get("ranking_temperature", 0.1)),
    }

    started_at = time.time()
    logger.info(
        "Progress training plan: episodes=%s world_size=%s episode_batch/rank=%s "
        "episode_micro_batch=%s queries/episode=%s global_queries/step=%s "
        "steps/epoch=%s epochs=%s planned_steps=%s max_steps=%s",
        len(dataset),
        accelerator.num_processes,
        episode_batch_size,
        episode_micro_batch_size,
        queries_per_episode,
        (
            episode_batch_size * queries_per_episode * accelerator.num_processes
            if queries_per_episode > 0
            else "all"
        ),
        steps_per_epoch,
        max_epochs,
        planned_steps,
        max_steps,
    )
    model.train()
    last_epoch = max(0, min(first_epoch, max_epochs - 1))
    for epoch in range(first_epoch, max_epochs):
        last_epoch = epoch
        epoch_dataloader = dataloader
        if epoch == first_epoch and resume_step_in_epoch:
            epoch_dataloader = accelerator.skip_first_batches(
                dataloader,
                resume_step_in_epoch,
            )
            logger.info(
                "Skipping %s already-completed batches in resumed epoch=%s",
                resume_step_in_epoch,
                epoch,
            )
        for episode_batch in epoch_dataloader:
            if global_step >= max_steps:
                break
            if not isinstance(episode_batch, (list, tuple)) or not episode_batch:
                raise ValueError("Progress collate must return a non-empty episode list")
            # 当前 rank 的一个 optimizer batch 含 E_local 条 episode；最后一个 DataLoader
            # batch 可能小于配置 E。所有 episode micro-batch 完成后才更新一次参数。
            episode_count = len(episode_batch)
            optimizer.zero_grad(set_to_none=True)
            metric_accumulator: Dict[str, float] = {}
            query_count_for_step = 0
            micro_starts = list(
                range(0, episode_count, episode_micro_batch_size)
            )
            for micro_number, micro_start in enumerate(micro_starts):
                micro_end = min(
                    micro_start + episode_micro_batch_size,
                    episode_count,
                )
                micro_episodes = episode_batch[micro_start:micro_end]
                # prepared 中 memory 为 [E_micro,...]，query 为 [sum Q_e,...]，并显式保存
                # query_episode_indices [sum Q_e]。正式 mixed 模式中所有 Q_e 都相等。
                prepared = prepare_episode_micro_batch(
                    micro_episodes,
                    queries_per_episode,
                )
                current_latents = prepared["current_latents"]
                progress_targets = prepared["progress_targets"]
                query_episode_indices = prepared["query_episode_indices"]
                query_count = int(current_latents.shape[0])
                query_count_for_step += query_count

                # Mixed 模式由配置校验保证 E_micro*Q <= query_batch_size，因此这里只会
                # 有一个 query chunk；旧 E=1,Q=0 全 episode 路径仍可按 B_q 分块。
                chunk_starts = list(range(0, query_count, query_batch_size))
                video_features = extract_episode_video_features(
                    frozen_vgm,
                    config,
                    prepared,
                )
                for chunk_number, start in enumerate(chunk_starts):
                    end = min(start + query_batch_size, query_count)
                    current_latent_chunk = current_latents[start:end]
                    progress_target_chunk = progress_targets[start:end]
                    query_episode_chunk = query_episode_indices[start:end]
                    chunk_size = end - start
                    is_last_forward = (
                        micro_number == len(micro_starts) - 1
                        and chunk_number == len(chunk_starts) - 1
                    )
                    # DDP 只在本 optimizer batch 的最后一次 backward 同步梯度。
                    sync_context = (
                        contextlib.nullcontext()
                        if is_last_forward
                        else accelerator.no_sync(model)
                    )
                    with sync_context:
                        with accelerator.autocast():
                            # outputs: [B_q,53]；第 q 行只读取 query_episode_chunk[q]
                            # 指定的那条 53-frame trajectory memory。
                            outputs = forward_progress(
                                model,
                                model_config.fusion_mode,
                                current_latent_chunk,
                                prepared["trajectory_latent"],
                                prepared["trajectory_frame_latents"],
                                video_features,
                                query_episode_indices=query_episode_chunk,
                            )
                            losses = compute_progress_loss(
                                outputs,
                                progress_target_chunk,
                                episode_ids=query_episode_chunk,
                                **loss_options,
                            )
                            # E 条 episode 等权。Mixed 模式每个 micro-batch 完整包含 Q 个
                            # query；旧 E=1,Q=0 分块时再乘 chunk_size/N，保持旧梯度口径。
                            episode_fraction = len(micro_episodes) / float(episode_count)
                            query_fraction = (
                                1.0
                                if queries_per_episode > 0
                                else chunk_size / float(query_count)
                            )
                            scaled_loss = (
                                losses["total_loss"]
                                * episode_fraction
                                * query_fraction
                            )
                        accelerator.backward(scaled_loss)

                    # 固定 Q 的 mixed 模式按 episode 数做宏平均；旧全 episode 路径继续
                    # 按 query 数加权，二者都与各自的反向缩放口径一致。
                    metric_weight = (
                        len(micro_episodes)
                        if queries_per_episode > 0
                        else chunk_size
                    )
                    weighted_metrics(metric_accumulator, losses, metric_weight)
                del video_features, prepared

            if float(config.training.get("grad_clip_norm", 0.0)) > 0:
                # 梯度裁剪发生在全部 E 条 episode 都 backward 完成之后。
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    float(config.training.grad_clip_norm),
                )
            optimizer.step()  # E 条 episode 共同产生一次参数更新。
            scheduler.step()  # 学习率调度也只前进一步。
            global_step += 1

            metrics = finalize_metrics(metric_accumulator)
            metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
            metrics["queries"] = float(query_count_for_step)
            metrics["episodes"] = float(episode_count)
            metrics["epoch"] = float(epoch)
            reduced = {
                name: float(
                    accelerator.reduce(
                        torch.tensor(value, device=accelerator.device), reduction="mean"
                    )
                )
                for name, value in metrics.items()
            }
            if global_step % log_interval == 0 and accelerator.is_main_process:
                logger.info(
                    "step=%s epoch=%s loss=%.5f mae=%.5f rmse=%.5f "
                    "episodes/rank=%s queries/rank=%s",
                    global_step,
                    epoch,
                    reduced["total_loss"],
                    reduced["mae"],
                    reduced["rmse"],
                    int(reduced["episodes"]),
                    int(reduced["queries"]),
                )
                accelerator.log(reduced, step=global_step)

            if global_step % save_interval == 0:
                save_checkpoint(
                    accelerator, config, checkpoint_root, global_step, epoch
                )

        if global_step >= max_steps:
            break

    save_checkpoint(accelerator, config, checkpoint_root, global_step, last_epoch)
    if accelerator.is_main_process:
        logger.info(
            "Progress training complete: steps=%s epochs<=%s elapsed=%.1fs",
            global_step,
            max_epochs,
            time.time() - started_at,
        )


def main() -> None:
    # 训练运行时才要求 accelerate；数据准备/混合 Batch helper 可在轻量单测环境导入。
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--cache_dir", default=None, help="Override cache.cache_dir")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--report_to", default=None, help="Override logging.report_to")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max_steps must be positive")
        config.training.max_steps = args.max_steps
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max_epochs must be positive")
        config.training.max_epochs = args.max_epochs
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    report_to = normalize_report_to(config.logging.get("report_to", "tensorboard"))
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=report_to or None,
        step_scheduler_with_optimizer=False,
        project_dir=str(
            Path(config.system.checkpoint_dir) / str(config.logging.run_name)
        ),
        project_config=ProjectConfiguration(
            total_limit=int(config.system.get("checkpoint_limit", 20))
        ),
    )
    setup_logging(accelerator.process_index, args.log_level)
    if not torch.cuda.is_available():
        raise RuntimeError("Progress Stage 2 training requires CUDA")
    torch.cuda.set_device(accelerator.local_process_index)
    set_seed(int(config.training.get("seed", 20260713)))

    if report_to:
        accelerator.init_trackers(
            project_name=str(config.logging.get("wandb_project", "motus")),
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": str(config.logging.run_name)}},
        )
    train(config, accelerator, args.resume_from)
    accelerator.end_training()


if __name__ == "__main__":
    main()
