#!/usr/bin/env python3
"""Training entrypoint for VGM bridge stage1 experiments."""

import argparse
import logging
import os
import re
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset, video_bridge_collate_fn
from models.vgm_bridge_stage1 import VGMBridgeStage1, VGMBridgeStage1Config
from utils.scheduler import create_scheduler


logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0, log_level: str = "INFO") -> None:
    """Setup process-aware logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )


def load_config(config_path: str) -> OmegaConf:
    """Load YAML config and populate derived fields used by datasets."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    logger.info("Loaded config from %s", config_path)
    logger.info("Dataset type: %s", config.dataset.type)
    logger.info("Training mode: %s", getattr(config, "training_mode", "unset"))
    logger.info("Video frames: %s", config.common.num_video_frames)
    return config


def normalize_report_to(value: Any) -> list[str]:
    """Normalize logging backend config."""
    if value == "all":
        return ["wandb", "tensorboard"]
    if value in (None, "none"):
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


class VGMBridgeStage1Trainer:
    """Minimal trainer for VGM bridge stage1 training."""

    def __init__(
        self,
        model: VGMBridgeStage1,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints/vgm_bridge_stage1",
        log_interval: int = 100,
        save_interval: int = 1000,
        report_to: Optional[list[str]] = None,
        tb_writer: Optional[SummaryWriter] = None,
        accelerator: Optional[Accelerator] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.report_to = report_to or []
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config
        self.global_step = 0
        self.epoch = 0

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("VGMBridgeStage1 trainer initialized on rank %s/%s", rank, world_size)
        logger.info("Logging backends: %s", self.report_to)

    def save_checkpoint(self, suffix: str = "") -> None:
        """Save full accelerator state and resolved config."""
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(str(checkpoint_dir))
            self.accelerator.wait_for_everyone()
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "global_step": self.global_step,
                },
                checkpoint_dir / "training_state.pt",
            )

        if self.rank == 0:
            if self.config is not None:
                OmegaConf.save(self.config, checkpoint_dir / "config.yaml", resolve=True)
            logger.info("Checkpoint saved to %s", checkpoint_dir)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load an accelerator checkpoint and recover global step from its name."""
        if not os.path.exists(checkpoint_path):
            logger.warning("Checkpoint path %s does not exist", checkpoint_path)
            return

        step_match = re.search(r"step_(\d+)", checkpoint_path)
        if step_match:
            self.global_step = int(step_match.group(1))
            logger.info("Resuming from step %s", self.global_step)

        if self.accelerator is not None:
            self.accelerator.load_state(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = int(state.get("global_step", self.global_step))

        logger.info("Checkpoint loaded from %s", checkpoint_path)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run one VGM bridge optimization step."""
        self.model.train()

        if self.accelerator is not None:
            accumulate_context = self.accelerator.accumulate(self.model)
        else:
            accumulate_context = nullcontext()

        with accumulate_context:
            first_frame = batch["first_frame"].to(self.device, dtype=self.dtype)
            video_frames = batch["video_frames"].to(self.device, dtype=self.dtype)
            language_embeddings = batch["language_embedding"]
            if language_embeddings is not None:
                language_embeddings = language_embeddings.to(self.device, dtype=self.dtype)

            model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
            loss_dict = model.training_step(
                first_frame=first_frame,
                video_frames=video_frames,
                language_embeddings=language_embeddings,
                return_dict=True,
            )
            total_loss = loss_dict["total_loss"]

            if self.accelerator is not None:
                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    grad_clip_norm = getattr(self.config.training, "grad_clip_norm", 1.0)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
            else:
                total_loss.backward()
                grad_clip_norm = getattr(self.config.training, "grad_clip_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}

    def train(self, max_steps: int, resume_from: Optional[str] = None) -> None:
        """Main step-based training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info("Starting VGMBridgeStage1 training for %s steps", max_steps)
        start_time = time.time()
        data_iter = iter(self.train_dataloader)

        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            if batch is None:
                continue

            step_start_time = time.time()
            metrics = self.train_step(batch)

            if self.accelerator is not None and not self.accelerator.sync_gradients:
                continue

            step_time = time.time() - step_start_time
            self.global_step += 1

            if self.global_step % self.log_interval == 0 and self.rank == 0:
                lrs = [group["lr"] for group in self.optimizer.param_groups]
                lr = lrs[0] if lrs else 0.0
                log_str = (
                    f"Step {self.global_step}/{max_steps}, "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"(Video: {metrics['video_loss']:.4f}, Middle: {metrics['middle_loss']:.4f}), "
                    f"LR: {lr:.2e}, Time: {step_time:.2f}s"
                )
                logger.info(log_str)

                log_payload = {
                    **metrics,
                    "learning_rate": lr,
                    "step_time": step_time,
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                }
                if "wandb" in self.report_to:
                    wandb.log(log_payload)
                if self.tb_writer is not None:
                    for key, value in log_payload.items():
                        self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info("VGMBridgeStage1 training completed in %.2fs (%s steps)", total_time, self.global_step)
        self.save_checkpoint()


def create_model_and_optimizer(config: OmegaConf) -> tuple[VGMBridgeStage1, torch.optim.Optimizer, Any]:
    """Create VGM bridge model, optimizer, and scheduler."""
    logger.info(
        "VGM conditioning_mode=%s, tail_condition_frames=%s",
        config.common.get("conditioning_mode", "v0"),
        config.common.get("tail_condition_frames", 1),
    )
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
        load_pretrained_backbones=getattr(config.model, "load_pretrained_backbones", None),
    )
    model = VGMBridgeStage1(model_config)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("VGMBridgeStage1 has no trainable parameters")

    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": float(config.training.learning_rate)}],
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = create_scheduler(optimizer, config)

    trainable_count = sum(param.numel() for param in trainable_params)
    logger.info("Trainable parameters: %s", f"{trainable_count:,}")
    return model, optimizer, scheduler


def create_train_dataloader(config: OmegaConf, rank: int, world_size: int) -> DataLoader:
    """Create train dataloader only; stage1 currently has no validation path."""
    if config.dataset.type != "video_bridge":
        raise ValueError(
            "train_vgm_bridge_stage1.py expects dataset.type='video_bridge'. "
            "Use videos/ plus optional umt5_wan/ embeddings for this stage."
        )

    dataset_dir = config.dataset.get("dataset_dir", None)
    if dataset_dir is None or len(dataset_dir) == 0:
        raise ValueError("Fill dataset.dataset_dir in the selected VGM bridge config with your video dataset root(s).")

    train_dataset = VideoBridgeDataset(
        dataset_dir=[str(path) for path in dataset_dir],
        global_downsample_rate=config.common.global_downsample_rate,
        num_video_frames=config.common.num_video_frames,
        video_size=(config.common.video_height, config.common.video_width),
        max_episodes=config.dataset.get("max_episodes", None),
        require_language_embedding=config.dataset.get("require_language_embedding", False),
        video_extensions=list(config.dataset.get("video_extensions", [".mp4"])),
        data_format=config.dataset.get("data_format", "auto"),
        image_column=config.dataset.get("image_column", "image"),
        image_columns=config.dataset.get("image_columns", None),
        view_layout=config.dataset.get("view_layout", "single"),
        task_language_embedding_dir=config.dataset.get("task_language_embedding_dir", None),
        task_language_embedding_pattern=config.dataset.get("task_language_embedding_pattern", "task_{task_index:06d}.pt"),
        cache_scan=config.dataset.get("cache_scan", True),
        val=False,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    return DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=video_bridge_collate_fn,
        drop_last=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VGM bridge stage1")
    parser.add_argument("--config", type=str, default="configs/vgm_bridge_v0.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "all", "none"],
        help="Logging backends to use",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")
    parser.add_argument("--resume_from", type=str, default=None, help="Override resume checkpoint path")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed launchers")
    args = parser.parse_args()

    config = load_config(args.config)
    if getattr(config, "training_mode", None) not in {"vgm_bridge_v0", "vgm_bridge_stage1"}:
        logger.warning(
            "Expected training_mode=vgm_bridge_v0 or vgm_bridge_stage1, got %s",
            getattr(config, "training_mode", None),
        )

    if args.checkpoint_dir is not None:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.run_name is not None:
        config.logging.run_name = args.run_name
    if args.resume_from is not None:
        config.resume.checkpoint_path = args.resume_from

    if getattr(config.resume, "checkpoint_path", None):
        config.model.load_pretrained_backbones = False

    report_to = normalize_report_to(config.logging.get("report_to", "tensorboard"))

    config_filename = os.path.basename(args.config)
    dataset_name = os.path.splitext(config_filename)[0]
    base_checkpoint_dir = config.system.checkpoint_dir
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, dataset_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.logging.get("run_name", None)
    if not run_name:
        run_name = (
            f"vgm_bridge_stage1_{config.dataset.type}_"
            f"bs{config.training.batch_size}_lr{config.training.learning_rate}_{timestamp}"
        )
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)

    accelerator_project_config = ProjectConfiguration(total_limit=20)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None,
        gradient_accumulation_steps=config.training.get("gradient_accumulation_steps", 1),
        mixed_precision="bf16",
        log_with=report_to if report_to else None,
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_project_config,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level or config.system.get("log_level", "INFO"))
    if not torch.cuda.is_available():
        raise RuntimeError("train_vgm_bridge_stage1.py requires CUDA; run it on the training server.")
    torch.cuda.set_device(accelerator.local_process_index)

    logger.info("Dataset: %s", dataset_name)
    logger.info("Checkpoints will be saved to: %s", config.system.checkpoint_dir)

    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        tb_log_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tb_writer.add_text("config", yaml.dump(OmegaConf.to_container(config, resolve=True)))
        logger.info("TensorBoard logs will be saved to: %s", tb_log_dir)

    if rank == 0 and "wandb" in report_to:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )

    try:
        logger.info("Creating VGMBridgeStage1 model and optimizer...")
        model, optimizer, scheduler = create_model_and_optimizer(config)

        logger.info("Creating train dataloader...")
        train_dataloader = create_train_dataloader(config, rank, world_size)

        logger.info("Preparing model, optimizer, dataloader, and scheduler with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            scheduler,
        )

        trainer = VGMBridgeStage1Trainer(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            report_to=report_to,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
        )
        trainer.train(max_steps=config.training.max_steps, resume_from=config.resume.checkpoint_path)

    except Exception as exc:
        logger.error("Training failed: %s", exc)
        import traceback

        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL ERROR] Training failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
