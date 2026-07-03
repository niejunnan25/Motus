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
    load_pretrained_backbones: Optional[bool] = None


class VGMBridgeStage1(nn.Module):
    """V0 first-last-frame bridge training for the WAN video prior only."""

    def __init__(self, config: VGMBridgeStage1Config):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("VGMBridgeStage1 requires CUDA; run this entrypoint on the training server.")

        self.config = config
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
            "Initialized VGMBridgeStage1 V0: num_video_frames=%s, video_size=%sx%s",
            config.num_video_frames,
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

    def _known_endpoint_mask(self, latent: torch.Tensor) -> torch.Tensor:
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
        mask[:, :, -1:] = 1
        return mask

    def training_step(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        language_embeddings: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Train V0: endpoint latent clamp + middle-only flow matching loss."""
        batch_size = video_frames.shape[0]

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

        condition_video = torch.zeros_like(full_video)
        condition_video[:, :, 0:1] = full_video[:, :, 0:1]
        condition_video[:, :, -1:] = full_video[:, :, -1:]

        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
            condition_latent = self.video_model.encode_video(condition_video.to(self.dtype))

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

        known_mask = self._known_endpoint_mask(clean_full_latent)
        noisy_video_latent = noisy_video_latent * (1 - known_mask) + condition_latent * known_mask

        video_target = video_noise - clean_full_latent
        video_target = video_target * (1 - known_mask)

        seq_len = self._wan_seq_len(noisy_video_latent)
        timestep_tokens = video_t_embed.unsqueeze(1).expand(batch_size, seq_len)
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
