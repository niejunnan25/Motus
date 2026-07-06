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

from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm import FlowMatchScheduler

from .wan_model import WanVideoModel

logger = logging.getLogger(__name__)


class EndpointTokenAdapter(nn.Module):
    """Encode a single endpoint frame latent into WAN token residuals."""

    def __init__(self, latent_channels: int, hidden_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.proj = nn.Conv3d(
            latent_channels + 2,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, endpoint_latent: torch.Tensor, target_latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        _, _, latent_t, latent_h, latent_w = target_latent.shape
        proj_weight = self.proj.weight
        endpoint_latent = endpoint_latent.to(device=proj_weight.device, dtype=proj_weight.dtype)
        endpoint_volume = endpoint_latent.expand(-1, -1, latent_t, -1, -1)

        ramp = torch.linspace(
            0.0,
            1.0,
            latent_t,
            device=proj_weight.device,
            dtype=proj_weight.dtype,
        ).view(1, 1, latent_t, 1, 1)
        ramp = ramp.expand(endpoint_latent.shape[0], -1, -1, latent_h, latent_w)
        end_indicator = torch.zeros_like(ramp)
        end_indicator[:, :, -1:] = 1
        endpoint_input = torch.cat([endpoint_volume, ramp, end_indicator], dim=1)

        tokens = self.proj(endpoint_input).flatten(2).transpose(1, 2)
        if tokens.shape[1] < seq_len:
            pad = tokens.new_zeros(tokens.shape[0], seq_len - tokens.shape[1], tokens.shape[2])
            tokens = torch.cat([tokens, pad], dim=1)
        elif tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len]
        return tokens


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
    conditioning_mode: str = "v0"
    mask_channels: int = 4
    load_pretrained_backbones: Optional[bool] = None


class VGMBridgeStage1(nn.Module):
    """Stage1 WAN video-prior training with first-frame and optional tail conditioning."""

    VALID_CONDITIONING_MODES = {"v0", "v0_5_endpoint", "v1_mask", "v1_mask_endpoint"}

    def __init__(self, config: VGMBridgeStage1Config):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("VGMBridgeStage1 requires CUDA; run this entrypoint on the training server.")

        self.config = config
        if config.conditioning_mode not in self.VALID_CONDITIONING_MODES:
            raise ValueError(
                f"Unknown conditioning_mode={config.conditioning_mode!r}; "
                f"expected one of {sorted(self.VALID_CONDITIONING_MODES)}"
            )
        if config.tail_condition_frames < 0:
            raise ValueError("tail_condition_frames must be >= 0")
        if config.tail_condition_frames > config.num_video_frames:
            raise ValueError("tail_condition_frames must be <= num_video_frames")
        if config.conditioning_mode in {"v0_5_endpoint", "v1_mask_endpoint"} and config.tail_condition_frames < 1:
            raise ValueError("Endpoint conditioning modes require tail_condition_frames >= 1")
        if config.mask_channels < 1:
            raise ValueError("mask_channels must be >= 1")
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
        self.latent_channels = self.video_model.wan_model.in_dim
        self.use_mask_condition = config.conditioning_mode in {"v1_mask", "v1_mask_endpoint"}
        self.use_endpoint_adapter = config.conditioning_mode in {"v0_5_endpoint", "v1_mask_endpoint"}

        if self.use_mask_condition:
            self._expand_patch_embedding(self.latent_channels * 2 + config.mask_channels)
        if self.use_endpoint_adapter:
            self.endpoint_adapter = EndpointTokenAdapter(
                latent_channels=self.latent_channels,
                hidden_dim=self.video_model.wan_model.dim,
                patch_size=self.video_model.wan_model.patch_size,
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.endpoint_adapter = None

        self.fm_train_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        logger.info(
            "Initialized VGMBridgeStage1: conditioning_mode=%s, num_video_frames=%s, "
            "tail_condition_frames=%s, video_size=%sx%s",
            config.conditioning_mode,
            config.num_video_frames,
            config.tail_condition_frames,
            config.video_height,
            config.video_width,
        )

    def _expand_patch_embedding(self, new_in_channels: int) -> None:
        """Extend WAN patch embedding for condition latent and mask channels."""
        patch_embedding = self.video_model.wan_model.patch_embedding
        if patch_embedding.in_channels == new_in_channels:
            return
        if patch_embedding.in_channels != self.latent_channels:
            raise ValueError(
                f"Cannot expand patch_embedding with in_channels={patch_embedding.in_channels}; "
                f"expected base latent_channels={self.latent_channels}"
            )
        if new_in_channels <= self.latent_channels:
            raise ValueError(
                f"new_in_channels={new_in_channels} must be greater than base latent_channels={self.latent_channels}"
            )

        expanded = nn.Conv3d(
            new_in_channels,
            patch_embedding.out_channels,
            kernel_size=patch_embedding.kernel_size,
            stride=patch_embedding.stride,
            padding=patch_embedding.padding,
            dilation=patch_embedding.dilation,
            groups=patch_embedding.groups,
            bias=patch_embedding.bias is not None,
            padding_mode=patch_embedding.padding_mode,
        ).to(device=patch_embedding.weight.device, dtype=patch_embedding.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : self.latent_channels].copy_(patch_embedding.weight)
            if patch_embedding.bias is not None:
                expanded.bias.copy_(patch_embedding.bias)

        self.video_model.wan_model.patch_embedding = expanded
        self.video_model.wan_model.in_dim = new_in_channels
        logger.info(
            "Expanded WAN patch_embedding from %s to %s input channels",
            self.latent_channels,
            new_in_channels,
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

    def _forward_wan(
        self,
        latent_list: List[torch.Tensor],
        timestep_tokens: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        token_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run WAN with an optional token residual injected after patch embedding."""
        wan_model = self.video_model.wan_model
        device = wan_model.patch_embedding.weight.device
        if wan_model.freqs.device != device:
            wan_model.freqs = wan_model.freqs.to(device)

        x = [wan_model.patch_embedding(item.unsqueeze(0)) for item in latent_list]
        grid_sizes = torch.stack(
            [torch.tensor(item.shape[2:], dtype=torch.long, device=device) for item in x]
        )
        x = [item.flatten(2).transpose(1, 2) for item in x]
        seq_lens = torch.tensor([item.size(1) for item in x], dtype=torch.long, device=device)
        if seq_lens.max() > seq_len:
            raise ValueError(f"WAN seq_len={seq_len} is smaller than max sequence length={seq_lens.max().item()}")
        x = torch.cat(
            [
                torch.cat([item, item.new_zeros(1, seq_len - item.size(1), item.size(2))], dim=1)
                for item in x
            ]
        )
        if token_residual is not None:
            x = x + token_residual.to(device=x.device, dtype=x.dtype)

        if timestep_tokens.dim() == 1:
            timestep_tokens = timestep_tokens.unsqueeze(1).expand(timestep_tokens.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = timestep_tokens.size(0)
            timestep_flat = timestep_tokens.flatten()
            e = wan_model.time_embedding(
                sinusoidal_embedding_1d(wan_model.freq_dim, timestep_flat)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = wan_model.time_projection(e).unflatten(2, (6, wan_model.dim))

        context_emb = wan_model.text_embedding(
            torch.stack(
                [
                    torch.cat([item, item.new_zeros(wan_model.text_len - item.size(0), item.size(1))])
                    for item in context
                ]
            )
        )
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=wan_model.freqs,
            context=context_emb,
            context_lens=None,
        )

        for block in wan_model.blocks:
            x = block(x, **kwargs)

        x = wan_model.head(x, e)
        x = wan_model.unpatchify(x, grid_sizes)
        return torch.stack([item.float() for item in x], dim=0)

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

    def _first_condition_mask(self, latent: torch.Tensor) -> torch.Tensor:
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
        return mask

    def _frame_condition_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        frame_count = self.config.num_video_frames + 1
        mask = torch.zeros(batch_size, frame_count, device=device, dtype=dtype)
        mask[:, 0] = 1
        tail_n = int(self.config.tail_condition_frames)
        if tail_n > 0:
            mask[:, -tail_n:] = 1
        return mask

    def _rearrange_frame_mask(self, frame_mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch_size, _, latent_t, latent_h, latent_w = latent.shape
        mask_channels = int(self.config.mask_channels)
        latent_mask = latent.new_zeros(batch_size, mask_channels, latent_t, 1, 1)

        for latent_idx in range(latent_t):
            if latent_idx == 0:
                frame_indices = [0]
            else:
                start = 1 + mask_channels * (latent_idx - 1)
                frame_indices = list(range(start, start + mask_channels))
            for channel_idx, frame_idx in enumerate(frame_indices[:mask_channels]):
                if frame_idx < frame_mask.shape[1]:
                    latent_mask[:, channel_idx, latent_idx, 0, 0] = frame_mask[:, frame_idx]

        return latent_mask.expand(-1, -1, -1, latent_h, latent_w)

    def _build_model_input(
        self,
        noisy_video_latent: torch.Tensor,
        condition_latent: Optional[torch.Tensor],
        frame_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_mask_condition:
            return noisy_video_latent
        if condition_latent is None or frame_mask is None:
            raise ValueError("condition_latent and frame_mask are required for mask-conditioned modes")
        mask_volume = self._rearrange_frame_mask(frame_mask, noisy_video_latent)
        return torch.cat([noisy_video_latent, condition_latent, mask_volume], dim=1)

    def _endpoint_token_residual(
        self,
        last_frame: Optional[torch.Tensor],
        target_latent: torch.Tensor,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        if not self.use_endpoint_adapter:
            return None
        if last_frame is None:
            raise ValueError("last_frame is required when conditioning_mode uses the endpoint adapter")
        last_frame_norm = (last_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            endpoint_latent = self.video_model.encode_video(last_frame_norm.to(self.dtype))
        return self.endpoint_adapter(endpoint_latent, target_latent, seq_len)

    def _encode_condition_latent(
        self,
        first_frame_norm: torch.Tensor,
        condition_video: torch.Tensor,
        latent_template: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the latent used for hard clamps and optional mask conditioning."""
        should_encode_condition_video = (
            (self.config.conditioning_mode == "v0" and int(self.config.tail_condition_frames) > 0)
            or self.use_mask_condition
        )
        if should_encode_condition_video:
            condition_latent = self.video_model.encode_video(condition_video.to(self.dtype))
            if self.use_mask_condition:
                first_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
                condition_latent[:, :, 0:1] = first_frame_latent
            return condition_latent

        first_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
        if latent_template is None:
            latent_t = 1 + self.config.num_video_frames // 4
            condition_latent = first_frame_latent.new_zeros(
                first_frame_latent.shape[0],
                first_frame_latent.shape[1],
                latent_t,
                first_frame_latent.shape[3],
                first_frame_latent.shape[4],
            )
        else:
            condition_latent = torch.zeros_like(latent_template)
        condition_latent[:, :, 0:1] = first_frame_latent
        return condition_latent

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
        """Run one bridge training step for the configured conditioning mode."""
        batch_size = video_frames.shape[0]

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)

        last_frame = video_frames[:, -1] if int(self.config.tail_condition_frames) > 0 else None
        tail_n = int(self.config.tail_condition_frames)
        condition_video = torch.zeros_like(full_video)
        condition_video[:, :, 0:1] = full_video[:, :, 0:1]
        if tail_n > 0:
            condition_video[:, :, -tail_n:] = full_video[:, :, -tail_n:]

        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
            condition_latent = self._encode_condition_latent(
                first_frame_norm=first_frame_norm,
                condition_video=condition_video,
                latent_template=clean_full_latent,
            )

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

        if self.config.conditioning_mode == "v0":
            known_mask = self._known_condition_mask(clean_full_latent)
        else:
            known_mask = self._first_condition_mask(clean_full_latent)
        noisy_video_latent = noisy_video_latent * (1 - known_mask) + condition_latent * known_mask

        video_target = video_noise - clean_full_latent
        video_target = video_target * (1 - known_mask)

        seq_len = self._wan_seq_len(noisy_video_latent)
        frame_mask = self._frame_condition_mask(batch_size, clean_full_latent.device, clean_full_latent.dtype)
        model_input = self._build_model_input(
            noisy_video_latent=noisy_video_latent,
            condition_latent=condition_latent,
            frame_mask=frame_mask,
        )
        token_residual = self._endpoint_token_residual(last_frame=last_frame, target_latent=noisy_video_latent, seq_len=seq_len)
        timestep_tokens = self._timestep_tokens(
            timestep=video_t_embed,
            latent=noisy_video_latent,
            known_mask=known_mask,
            seq_len=seq_len,
        )
        context = self._context_list(language_embeddings, batch_size)
        latent_list = [model_input[i] for i in range(batch_size)]

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            video_pred = self._forward_wan(
                latent_list=latent_list,
                timestep_tokens=timestep_tokens,
                context=context,
                seq_len=seq_len,
                token_residual=token_residual,
            )

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
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
            condition_latent = self._encode_condition_latent(
                first_frame_norm=first_frame_norm,
                condition_video=condition_video,
            )
            if self.config.conditioning_mode == "v0":
                known_mask = self._known_condition_mask(condition_latent)
            else:
                known_mask = self._first_condition_mask(condition_latent)
            frame_mask = self._frame_condition_mask(batch_size, condition_latent.device, condition_latent.dtype)

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
            endpoint_frame = last_frame
            if endpoint_frame is None and tail_frames is not None and tail_frames.shape[1] > 0:
                endpoint_frame = tail_frames[:, -1]
            token_residual = self._endpoint_token_residual(
                last_frame=endpoint_frame,
                target_latent=latent,
                seq_len=seq_len,
            )

            for step_idx, timestep in enumerate(timesteps):
                model_input = self._build_model_input(
                    noisy_video_latent=latent,
                    condition_latent=condition_latent,
                    frame_mask=frame_mask,
                )
                timestep_tokens = self._timestep_tokens(
                    timestep=timestep.expand(batch_size),
                    latent=latent,
                    known_mask=known_mask,
                    seq_len=seq_len,
                )
                latent_list = [model_input[i] for i in range(batch_size)]
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    pred = self._forward_wan(
                        latent_list=latent_list,
                        timestep_tokens=timestep_tokens,
                        context=context,
                        seq_len=seq_len,
                        token_residual=token_residual,
                    )
                sigma = sigmas[step_idx]
                sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else sigmas.new_zeros(())
                latent = latent + pred * (sigma_next - sigma)
                latent = latent * (1 - known_mask) + condition_latent * known_mask

            decoded = self.video_model.decode_video(latent.to(self.dtype)).float()
            decoded = (decoded.clamp(-1.0, 1.0) + 1.0) * 0.5
            return decoded.permute(0, 2, 1, 3, 4).contiguous()
        finally:
            self.train(was_training)
