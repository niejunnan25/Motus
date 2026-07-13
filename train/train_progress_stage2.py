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
from typing import Any, Dict, Iterable, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.progress import ProgressEpisodeCacheDataset, progress_episode_collate_fn
from models.progress_stage2 import (
    ProgressStage2Config,
    build_progress_model,
    compute_progress_loss,
)

if __package__:
    from .eval_vgm_bridge_stage1 import build_model as build_vgm_model
else:
    from eval_vgm_bridge_stage1 import build_model as build_vgm_model


logger = logging.getLogger(__name__)


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
    if str(config.progress_model.fusion_mode) != "layerwise_wvm":
        return None
    vgm_config = OmegaConf.load(config.source.vgm_config)
    vgm = build_vgm_model(vgm_config, Path(config.source.vgm_checkpoint))
    vgm.to(device)
    vgm.eval()
    for parameter in vgm.parameters():
        parameter.requires_grad = False
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
    count = current_latents.shape[0]
    order = torch.randperm(count, device=current_latents.device)
    if queries_per_episode > 0:
        if queries_per_episode <= count:
            order = order[:queries_per_episode]
        else:
            repeats = math.ceil(queries_per_episode / count)
            order = order.repeat(repeats)[:queries_per_episode]
    return current_latents[order], progress[order]


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
) -> Dict[str, torch.Tensor]:
    if fusion_mode == "serial_latent":
        return model(
            current_latent=current_latent,
            trajectory_frame_latents=trajectory_frame_latents,
        )
    if fusion_mode == "layerwise_wvm":
        if video_features is None:
            raise RuntimeError("layerwise_wvm requires frozen WAN hidden states")
        return model(
            current_latent=current_latent,
            trajectory_frame_latents=trajectory_frame_latents,
            video_hidden_states=video_features["hidden_states"],
            video_grid_size=video_features["grid_sizes"][0],
        )
    raise ValueError(f"Unknown fusion mode={fusion_mode!r}")


def train(config: Any, accelerator: Accelerator, resume_from: Optional[str]) -> None:
    model_config = model_config_from_yaml(config)
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
    dataset = ProgressEpisodeCacheDataset(
        config.cache.cache_dir,
        split="train",
        load_language_embedding=model_config.fusion_mode == "layerwise_wvm",
        expected_num_progress_bins=model_config.num_progress_bins,
        max_episodes=config.training.get("max_episodes", None),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config.system.get("num_workers", 2)),
        pin_memory=bool(config.system.get("pin_memory", True)),
        collate_fn=progress_episode_collate_fn,
        drop_last=False,
    )

    max_steps = int(config.training.max_steps)
    max_epochs = int(config.training.max_epochs)
    steps_per_epoch = math.ceil(len(dataset) / accelerator.num_processes)
    planned_steps = min(max_steps, max_epochs * steps_per_epoch)
    scheduler = create_scheduler(
        optimizer,
        int(config.training.get("warmup_steps", 0)),
        planned_steps,
    )

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

    query_batch_size = int(config.training.query_batch_size)
    queries_per_episode = int(config.training.get("queries_per_episode", 0))
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
        "Progress training plan: episodes=%s world_size=%s steps/epoch=%s epochs=%s "
        "planned_steps=%s max_steps=%s",
        len(dataset),
        accelerator.num_processes,
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
        for episode in epoch_dataloader:
            if global_step >= max_steps:
                break
            trajectory_latent = episode["trajectory_latent"].unsqueeze(0)
            trajectory_frame_latents = episode["trajectory_frame_latents"].unsqueeze(0)
            current_latents, progress_targets = prepare_episode_queries(
                episode["current_latents"],
                episode["progress"],
                queries_per_episode,
            )
            if current_latents.shape[0] == 0:
                continue

            video_features = None
            if frozen_vgm is not None:
                first_frame = episode["first_frame"].float().div(255.0).unsqueeze(0)
                last_frame = episode["last_frame"].float().div(255.0).unsqueeze(0)
                language_embedding = episode["language_embedding"].unsqueeze(0)
                with torch.no_grad():
                    video_features = frozen_vgm.extract_bridge_hidden_states(
                        trajectory_latent=trajectory_latent,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        language_embeddings=language_embedding,
                        feature_timestep=float(
                            config.progress_model.get("feature_timestep", 0.0)
                        ),
                    )

            optimizer.zero_grad(set_to_none=True)
            chunk_starts = list(range(0, current_latents.shape[0], query_batch_size))
            metric_accumulator: Dict[str, float] = {}
            for chunk_number, start in enumerate(chunk_starts):
                end = min(start + query_batch_size, current_latents.shape[0])
                is_last_chunk = chunk_number == len(chunk_starts) - 1
                sync_context = (
                    contextlib.nullcontext()
                    if is_last_chunk
                    else accelerator.no_sync(model)
                )
                with sync_context:
                    with accelerator.autocast():
                        outputs = forward_progress(
                            model,
                            model_config.fusion_mode,
                            current_latents[start:end],
                            trajectory_latent,
                            trajectory_frame_latents,
                            video_features,
                        )
                        losses = compute_progress_loss(
                            outputs,
                            progress_targets[start:end],
                            **loss_options,
                        )
                        query_fraction = (end - start) / float(current_latents.shape[0])
                        scaled_loss = losses["total_loss"] * query_fraction
                    accelerator.backward(scaled_loss)
                weighted_metrics(metric_accumulator, losses, end - start)

            if float(config.training.get("grad_clip_norm", 0.0)) > 0:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    float(config.training.grad_clip_norm),
                )
            optimizer.step()
            scheduler.step()
            global_step += 1

            metrics = finalize_metrics(metric_accumulator)
            metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
            metrics["queries"] = float(current_latents.shape[0])
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
                    "step=%s epoch=%s loss=%.5f mae=%.5f rmse=%.5f queries/rank=%s",
                    global_step,
                    epoch,
                    reduced["total_loss"],
                    reduced["mae"],
                    reduced["rmse"],
                    int(reduced["queries"]),
                )
                accelerator.log(reduced, step=global_step)

            if global_step % save_interval == 0:
                save_checkpoint(
                    accelerator, config, checkpoint_root, global_step, epoch
                )

            del video_features

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
