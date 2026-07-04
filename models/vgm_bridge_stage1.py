import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.utils.fm import FlowMatchScheduler

from .wan_model import WanVideoModel

logger = logging.getLogger(__name__)


@dataclass
class VGMBridgeStage1Config:
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"
    num_video_frames: int = 16
    video_height: int = 384
    video_width: int = 320
    batch_size: int = 1
    tail_condition_frames: int = 1
    load_pretrained_backbones: Optional[bool] = None


class VGMBridgeStage1(nn.Module):
    """Stage1 WAN video-prior training with first-frame and optional tail conditioning."""

    def __init__(self, config: VGMBridgeStage1Config):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("VGMBridgeStage1 requires CUDA; run this entrypoint on the training server.")

        self.config = config
        if config.tail_condition_frames < 0:
            raise ValueError("tail_condition_frames must be >= 0")
        if config.tail_condition_frames > config.num_video_frames:
            raise ValueError("tail_condition_frames must be <= num_video_frames")
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.video_precision]
        cuda_device = f"cuda:{torch.cuda.current_device()}"

        load_backbone = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)
        if load_backbone:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                device=cuda_device,
                precision=config.video_precision,
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device=cuda_device,
                precision=config.video_precision,
            )

        if hasattr(self.video_model.vae, "parameters"):
            for param in self.video_model.vae.parameters():
                param.requires_grad = False

        self.device = next(self.video_model.wan_model.parameters()).device
        self.fm_train_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        logger.info(
            "Initialized VGMBridgeStage1 V0: num_video_frames=%s, tail_condition_frames=%s, video_size=%sx%s",
            config.num_video_frames,
            config.tail_condition_frames,
            config.video_height,
            config.video_width,
        )

    def _context_list(self, language_embeddings: Optional[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
        text_len = self.video_model.wan_model.text_len
        text_dim = self.video_model.wan_model.text_dim

        if language_embeddings is None:
            return [
                torch.zeros(text_len, text_dim, device=self.device, dtype=self.dtype)
                for _ in range(batch_size)
            ]

        if isinstance(language_embeddings, torch.Tensor):
            if language_embeddings.dim() == 2:
                language_embeddings = language_embeddings.unsqueeze(0)
            items = [language_embeddings[i] for i in range(language_embeddings.shape[0])]
        else:
            items = list(language_embeddings)

        context = []
        for emb in items:
            emb = emb.to(device=self.device, dtype=self.dtype)
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            if emb.shape[0] > text_len:
                emb = emb[:text_len]
            context.append(emb)
        return context

    def _wan_seq_len(self, latent: torch.Tensor) -> int:
        _, _, t, h, w = latent.shape
        pt, ph, pw = self.video_model.wan_model.patch_size
        return (t // pt) * (h // ph) * (w // pw)

    def _timestep_tokens(
        self,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        known_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        tokens = timestep.unsqueeze(1).expand(timestep.shape[0], seq_len).clone()

        _, _, t, h, w = latent.shape
        pt, ph, pw = self.video_model.wan_model.patch_size
        if t % pt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                f"Latent shape {(t, h, w)} is not divisible by WAN patch_size {(pt, ph, pw)}"
            )

        frame_known = known_mask[:, 0, :, 0, 0].bool()
        if pt > 1:
            frame_known = frame_known.view(frame_known.shape[0], t // pt, pt).any(dim=2)
        patch_known = frame_known[:, :, None, None].expand(-1, -1, h // ph, w // pw)
        patch_known = patch_known.reshape(frame_known.shape[0], -1)
        if patch_known.shape[1] < seq_len:
            pad = patch_known.new_zeros(patch_known.shape[0], seq_len - patch_known.shape[1])
            patch_known = torch.cat([patch_known, pad], dim=1)
        elif patch_known.shape[1] > seq_len:
            patch_known = patch_known[:, :seq_len]

        return tokens.masked_fill(patch_known, 0)

    def _known_condition_mask(self, latent: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(
            latent.shape[0],
            1,
            latent.shape[2],
            1,
            1,
            device=latent.device,
            dtype=latent.dtype,
        )
        mask[:, :, 0:1] = 1
        if int(self.config.tail_condition_frames) > 0:
            mask[:, :, -1:] = 1
        return mask

    def _make_condition_video(
        self,
        first_frame: torch.Tensor,
        last_frame: Optional[torch.Tensor] = None,
        tail_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = first_frame.shape[0]
        tail_n = int(self.config.tail_condition_frames)
        if tail_n > 0:
            if tail_frames is None:
                if last_frame is None:
                    raise ValueError("Either last_frame or tail_frames must be provided")
                tail_frames = last_frame.unsqueeze(1).expand(-1, tail_n, -1, -1, -1)
            elif tail_frames.shape[1] != tail_n:
                raise ValueError(
                    f"Expected tail_frames with {tail_n} frames, got shape={tuple(tail_frames.shape)}"
                )

        condition_video = first_frame.new_zeros(
            batch_size,
            first_frame.shape[1],
            self.config.num_video_frames + 1,
            first_frame.shape[2],
            first_frame.shape[3],
        )
        condition_video[:, :, 0] = first_frame * 2.0 - 1.0
        if tail_n > 0:
            condition_video[:, :, -tail_n:] = (tail_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        return condition_video

    def training_step(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        language_embeddings: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Train V0: first latent clamp plus optional tail latent clamp."""
        batch_size = video_frames.shape[0]

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

        condition_video = torch.zeros_like(full_video)
        condition_video[:, :, 0:1] = full_video[:, :, 0:1]
        tail_n = int(self.config.tail_condition_frames)
        if tail_n > 0:
            condition_video[:, :, -tail_n:] = full_video[:, :, -tail_n:]

        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
            if tail_n > 0:
                condition_latent = self.video_model.encode_video(condition_video.to(self.dtype))
            else:
                first_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
                condition_latent = torch.zeros_like(clean_full_latent)
                condition_latent[:, :, 0:1] = first_frame_latent

        timestep_id = torch.randint(
            0,
            self.fm_train_scheduler.num_train_timesteps,
            (batch_size,),
        )
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(
            dtype=self.dtype,
            device=self.device,
        ).view(batch_size, 1, 1, 1, 1)

        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma

        known_mask = self._known_condition_mask(clean_full_latent)
        noisy_video_latent = noisy_video_latent * (1 - known_mask) + condition_latent * known_mask

        video_target = video_noise - clean_full_latent
        video_target = video_target * (1 - known_mask)

        seq_len = self._wan_seq_len(noisy_video_latent)
        timestep_tokens = self._timestep_tokens(
            timestep=video_t_embed,
            latent=noisy_video_latent,
            known_mask=known_mask,
            seq_len=seq_len,
        )
        context = self._context_list(language_embeddings, batch_size)
        latent_list = [noisy_video_latent[i] for i in range(batch_size)]

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            pred_list = self.video_model.wan_model(
                latent_list,
                t=timestep_tokens,
                context=context,
                seq_len=seq_len,
            )
        video_pred = torch.stack(pred_list, dim=0)

        loss_mask = 1 - known_mask
        sq_error = (video_pred.float() - video_target.float()).pow(2) * loss_mask.float()
        denom = loss_mask.expand_as(video_pred).float().sum().clamp_min(1.0)
        video_loss = sq_error.sum() / denom
        zero = video_loss.detach().new_zeros(())

        if return_dict:
            return {
                "total_loss": video_loss,
                "video_loss": video_loss,
                "middle_loss": video_loss,
                "action_loss": zero,
            }
        return {"total_loss": video_loss}

    @torch.no_grad()
    def sample_bridge(
        self,
        first_frame: torch.Tensor,
        last_frame: Optional[torch.Tensor] = None,
        tail_frames: Optional[torch.Tensor] = None,
        language_embeddings: Optional[torch.Tensor] = None,
        num_inference_steps: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate a conditioned video in pixel space [B, T, C, H, W]."""
        was_training = self.training
        self.eval()
        try:
            first_frame = first_frame.to(device=self.device, dtype=self.dtype)
            if last_frame is not None:
                last_frame = last_frame.to(device=self.device, dtype=self.dtype)
            if tail_frames is not None:
                tail_frames = tail_frames.to(device=self.device, dtype=self.dtype)
            batch_size = first_frame.shape[0]

            condition_video = self._make_condition_video(
                first_frame=first_frame,
                last_frame=last_frame,
                tail_frames=tail_frames,
            )
            condition_latent = self.video_model.encode_video(condition_video.to(self.dtype))
            if int(self.config.tail_condition_frames) == 0:
                first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
                first_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
                condition_latent.zero_()
                condition_latent[:, :, 0:1] = first_frame_latent
            known_mask = self._known_condition_mask(condition_latent)

            latent = torch.randn(
                condition_latent.shape,
                device=condition_latent.device,
                dtype=self.dtype,
                generator=generator,
            )
            latent = latent * (1 - known_mask) + condition_latent * known_mask

            scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000,
            )
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
            sigmas = scheduler.sigmas.to(device=self.device, dtype=self.dtype)
            timesteps = scheduler.timesteps.to(device=self.device, dtype=self.dtype)

            seq_len = self._wan_seq_len(latent)
            context = self._context_list(language_embeddings, batch_size)

            for step_idx, timestep in enumerate(timesteps):
                timestep_tokens = self._timestep_tokens(
                    timestep=timestep.expand(batch_size),
                    latent=latent,
                    known_mask=known_mask,
                    seq_len=seq_len,
                )
                latent_list = [latent[i] for i in range(batch_size)]
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    pred_list = self.video_model.wan_model(
                        latent_list,
                        t=timestep_tokens,
                        context=context,
                        seq_len=seq_len,
                    )
                pred = torch.stack(pred_list, dim=0)
                sigma = sigmas[step_idx]
                sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else sigmas.new_zeros(())
                latent = latent + pred * (sigma_next - sigma)
                latent = latent * (1 - known_mask) + condition_latent * known_mask

            decoded = self.video_model.decode_video(latent.to(self.dtype)).float()
            decoded = (decoded.clamp(-1.0, 1.0) + 1.0) * 0.5
            return decoded.permute(0, 2, 1, 3, 4).contiguous()
        finally:
            self.train(was_training)
