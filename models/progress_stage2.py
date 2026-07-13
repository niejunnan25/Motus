"""Progress models over a frozen endpoint-conditioned VGM trajectory memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProgressStage2Config:
    fusion_mode: str = "serial_latent"
    num_progress_bins: int = 53
    latent_channels: int = 48
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_multiplier: int = 4
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    frame_num_tokens: int = 16
    video_hidden_dim: int = 3072
    dropout: float = 0.0


def _axis_sincos(
    length: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if dim <= 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    half = max(1, dim // 2)
    denominator = max(1, half - 1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / denominator
    )
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    embedding = torch.cat(
        [torch.sin(positions * frequencies), torch.cos(positions * frequencies)],
        dim=1,
    )
    if embedding.shape[1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding[:, :dim].to(dtype=dtype)


def factorized_3d_position_embedding(
    time: int,
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic [T, H, W, D] temporal-spatial positions."""
    time_dim = dim // 3
    height_dim = dim // 3
    width_dim = dim - time_dim - height_dim
    time_pos = _axis_sincos(time, time_dim, device, dtype)[:, None, None, :]
    height_pos = _axis_sincos(height, height_dim, device, dtype)[None, :, None, :]
    width_pos = _axis_sincos(width, width_dim, device, dtype)[None, None, :, :]
    return torch.cat(
        [
            time_pos.expand(time, height, width, -1),
            height_pos.expand(time, height, width, -1),
            width_pos.expand(time, height, width, -1),
        ],
        dim=-1,
    )


class LatentPatchTokenizer(nn.Module):
    """Patchify VAE latents while preserving explicit temporal groups."""

    def __init__(
        self, in_channels: int, hidden_dim: int, patch_size: Tuple[int, int, int]
    ):
        super().__init__()
        self.patch_size = tuple(int(value) for value in patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        if latent.ndim != 5:
            raise ValueError(
                f"Expected [B, C, T, H, W] latent, got {tuple(latent.shape)}"
            )
        _, _, time, height, width = latent.shape
        patch_t, patch_h, patch_w = self.patch_size
        if time % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                f"Latent shape {(time, height, width)} is not divisible by patch size {self.patch_size}"
            )

        features = self.proj(latent)
        _, hidden_dim, grid_t, grid_h, grid_w = features.shape
        tokens = features.permute(0, 2, 3, 4, 1)
        positions = factorized_3d_position_embedding(
            grid_t,
            grid_h,
            grid_w,
            hidden_dim,
            tokens.device,
            tokens.dtype,
        )
        tokens = self.norm(tokens + positions.unsqueeze(0))
        return tokens.reshape(tokens.shape[0], grid_t, grid_h * grid_w, hidden_dim), (
            grid_t,
            grid_h,
            grid_w,
        )


class AttentionResampler(nn.Module):
    """Compress a variable token set into a small number of learned query tokens."""

    def __init__(
        self, hidden_dim: int, num_heads: int, num_queries: int, dropout: float
    ):
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        normalized = self.input_norm(tokens)
        output, _ = self.attn(queries, normalized, normalized, need_weights=False)
        return self.output_norm(queries + output)


class ProgressCrossAttentionBlock(nn.Module):
    """Update Progress tokens without writing back into trajectory tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        memory_dim = hidden_dim if memory_dim is None else int(memory_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=memory_dim,
            vdim=memory_dim,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * int(ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self, progress_tokens: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        self_input = self.self_norm(progress_tokens)
        self_output, _ = self.self_attn(
            self_input, self_input, self_input, need_weights=False
        )
        progress_tokens = progress_tokens + self_output

        query = self.cross_norm(progress_tokens)
        memory = self.memory_norm(memory_tokens).to(dtype=query.dtype)
        cross_output, _ = self.cross_attn(query, memory, memory, need_weights=False)
        progress_tokens = progress_tokens + cross_output
        return progress_tokens + self.ffn(self.ffn_norm(progress_tokens))


class AsymmetricJointProgressBlock(nn.Module):
    """WVM-style masked joint attention: Progress reads Video, never vice versa."""

    def __init__(
        self,
        hidden_dim: int,
        video_hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.progress_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(video_hidden_dim)
        self.video_projection = nn.Linear(video_hidden_dim, hidden_dim)
        self.joint_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * int(ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self, progress_tokens: torch.Tensor, video_tokens: torch.Tensor
    ) -> torch.Tensor:
        progress_input = self.progress_norm(progress_tokens)
        # Project the shared trajectory once before broadcasting it across all
        # current-frame queries from the same episode.
        video_input = self.video_projection(self.video_norm(video_tokens))
        if video_input.shape[0] == 1 and progress_input.shape[0] > 1:
            video_input = video_input.expand(progress_input.shape[0], -1, -1)
        elif video_input.shape[0] != progress_input.shape[0]:
            raise ValueError(
                "Video token batch must be one or match the Progress query batch"
            )
        key_value = torch.cat(
            [video_input.to(progress_input.dtype), progress_input], dim=1
        )
        output, _ = self.joint_attention(
            progress_input,
            key_value,
            key_value,
            need_weights=False,
        )
        progress_tokens = progress_tokens + output
        return progress_tokens + self.ffn(self.ffn_norm(progress_tokens))


class TemporalAlignmentHead(nn.Module):
    """Score one current observation against explicit generated-frame slots."""

    def __init__(self, query_dim: int, memory_dim: int, hidden_dim: int, num_bins: int):
        super().__init__()
        if num_bins < 2:
            raise ValueError("Progress matching requires at least two frame bins")
        self.num_bins = int(num_bins)
        self.query_norm = nn.LayerNorm(query_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self, query_tokens: torch.Tensor, temporal_memory: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if temporal_memory.ndim != 3 or temporal_memory.shape[1] != self.num_bins:
            raise ValueError(
                "Temporal memory must contain one slot per generated frame: "
                f"expected [B,{self.num_bins},D], got {tuple(temporal_memory.shape)}"
            )
        query = self.query_proj(self.query_norm(query_tokens.mean(dim=1)))
        memory = self.memory_proj(self.memory_norm(temporal_memory))
        query = F.normalize(query.float(), dim=-1)
        memory = F.normalize(memory.float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("bd,btd->bt", query, memory) * scale
        probabilities = logits.softmax(dim=-1)
        frame_indices = torch.arange(
            self.num_bins,
            device=logits.device,
            dtype=probabilities.dtype,
        )
        positions = frame_indices / float(self.num_bins - 1)
        progress = (probabilities * positions.unsqueeze(0)).sum(dim=-1)
        expected_frame = (probabilities * frame_indices.unsqueeze(0)).sum(dim=-1)
        return {
            "progress": progress,
            "expected_frame": expected_frame,
            "matched_frame": logits.argmax(dim=-1),
            "alignment_logits": logits,
            "alignment_probabilities": probabilities,
            "memory_positions": positions,
            "memory_frame_indices": frame_indices,
        }


class SharedFrameLatentEncoder(nn.Module):
    """Encode current and generated RGB-frame latents with shared weights."""

    def __init__(self, config: ProgressStage2Config):
        super().__init__()
        self.tokenizer = LatentPatchTokenizer(
            config.latent_channels,
            config.hidden_dim,
            config.patch_size,
        )
        self.resampler = AttentionResampler(
            config.hidden_dim,
            config.num_heads,
            config.frame_num_tokens,
            config.dropout,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.tokenizer(latent)
        if grid[0] != 1:
            raise ValueError(
                f"Current observation must encode to one temporal latent, got grid={grid}"
            )
        return self.resampler(tokens.flatten(1, 2))


def encode_trajectory_frame_memory(
    frame_encoder: SharedFrameLatentEncoder,
    trajectory_frame_latents: torch.Tensor,
    *,
    num_progress_bins: int,
    type_embedding: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode independently-VAE-encoded generated frames into 53 ordered slots."""
    if trajectory_frame_latents.ndim != 6:
        raise ValueError(
            "trajectory_frame_latents must be [B,F,C,1,H,W], got "
            f"{tuple(trajectory_frame_latents.shape)}"
        )
    batch_size, frames, channels, latent_time, height, width = (
        trajectory_frame_latents.shape
    )
    if frames != num_progress_bins:
        raise ValueError(
            f"Expected exactly {num_progress_bins} generated frame slots, got {frames}"
        )
    if latent_time != 1:
        raise ValueError(
            "Each generated frame must be VAE-encoded independently to one temporal latent, "
            f"got latent_time={latent_time}"
        )

    frame_tokens = frame_encoder(
        trajectory_frame_latents.reshape(
            batch_size * frames,
            channels,
            latent_time,
            height,
            width,
        )
    )
    tokens_per_frame = frame_tokens.shape[1]
    hidden_dim = frame_tokens.shape[2]
    frame_tokens = frame_tokens.reshape(
        batch_size,
        frames,
        tokens_per_frame,
        hidden_dim,
    )
    frame_positions = _axis_sincos(
        frames,
        hidden_dim,
        frame_tokens.device,
        frame_tokens.dtype,
    )[None, :, None, :]
    frame_tokens = frame_tokens + frame_positions + type_embedding[:, None, :, :]
    return frame_tokens.flatten(1, 2), frame_tokens.mean(dim=2)


class SerialLatentProgressModel(nn.Module):
    """Post-generation matcher over 53 independently encoded trajectory frames."""

    def __init__(self, config: ProgressStage2Config):
        super().__init__()
        self.config = config
        self.frame_encoder = SharedFrameLatentEncoder(config)
        self.current_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.memory_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        nn.init.normal_(self.current_type_embedding, std=0.02)
        nn.init.normal_(self.memory_type_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [
                ProgressCrossAttentionBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.ffn_multiplier,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.alignment_head = TemporalAlignmentHead(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.num_progress_bins,
        )

    def forward(
        self,
        current_latent: torch.Tensor,
        trajectory_frame_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        progress_tokens = (
            self.frame_encoder(current_latent) + self.current_type_embedding
        )
        memory_tokens, temporal_memory = encode_trajectory_frame_memory(
            self.frame_encoder,
            trajectory_frame_latents,
            num_progress_bins=self.config.num_progress_bins,
            type_embedding=self.memory_type_embedding,
        )
        if memory_tokens.shape[0] == 1 and progress_tokens.shape[0] > 1:
            memory_tokens = memory_tokens.expand(progress_tokens.shape[0], -1, -1)
            temporal_memory = temporal_memory.expand(progress_tokens.shape[0], -1, -1)
        elif memory_tokens.shape[0] != progress_tokens.shape[0]:
            raise ValueError(
                "Trajectory batch must be one or match the current-observation batch"
            )

        for block in self.blocks:
            progress_tokens = block(progress_tokens, memory_tokens)
        return self.alignment_head(progress_tokens, temporal_memory)


class LayerwiseProgressModel(nn.Module):
    """WVM-style Progress expert that reads every frozen WAN hidden level."""

    def __init__(self, config: ProgressStage2Config):
        super().__init__()
        self.config = config
        self.frame_encoder = SharedFrameLatentEncoder(config)
        self.current_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.memory_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        nn.init.normal_(self.current_type_embedding, std=0.02)
        nn.init.normal_(self.memory_type_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [
                AsymmetricJointProgressBlock(
                    config.hidden_dim,
                    config.video_hidden_dim,
                    config.num_heads,
                    config.ffn_multiplier,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.alignment_head = TemporalAlignmentHead(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.num_progress_bins,
        )

    def forward(
        self,
        current_latent: torch.Tensor,
        trajectory_frame_latents: torch.Tensor,
        video_hidden_states: Sequence[torch.Tensor],
        video_grid_size: Sequence[int] | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if len(video_hidden_states) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} video hidden states, got {len(video_hidden_states)}"
            )
        if isinstance(video_grid_size, torch.Tensor):
            grid = tuple(
                int(value) for value in video_grid_size.detach().cpu().tolist()
            )
        else:
            grid = tuple(int(value) for value in video_grid_size)
        if len(grid) != 3:
            raise ValueError(f"video_grid_size must contain (T, H, W), got {grid}")

        progress_tokens = (
            self.frame_encoder(current_latent) + self.current_type_embedding
        )
        query_batch = progress_tokens.shape[0]
        valid_length = math.prod(grid)
        saw_video_hidden = False
        for block, video_hidden in zip(self.blocks, video_hidden_states):
            if video_hidden.shape[1] < valid_length:
                raise ValueError(
                    f"Video hidden has {video_hidden.shape[1]} tokens but grid={grid} requires {valid_length}"
                )
            video_hidden = video_hidden[:, :valid_length].detach()
            if video_hidden.shape[0] not in {1, query_batch}:
                raise ValueError(
                    "Video hidden batch must be one or match the current-observation batch"
                )
            progress_tokens = block(progress_tokens, video_hidden)
            saw_video_hidden = True

        if not saw_video_hidden:
            raise RuntimeError(
                "Layerwise Progress requires at least one Video hidden state"
            )
        _, temporal_memory = encode_trajectory_frame_memory(
            self.frame_encoder,
            trajectory_frame_latents,
            num_progress_bins=self.config.num_progress_bins,
            type_embedding=self.memory_type_embedding,
        )
        if temporal_memory.shape[0] == 1 and query_batch > 1:
            temporal_memory = temporal_memory.expand(query_batch, -1, -1)
        elif temporal_memory.shape[0] != query_batch:
            raise ValueError(
                "Trajectory-frame batch must be one or match the current-observation batch"
            )
        return self.alignment_head(progress_tokens, temporal_memory)


def build_progress_model(config: ProgressStage2Config) -> nn.Module:
    if config.num_progress_bins < 2:
        raise ValueError("num_progress_bins must be at least two")
    if config.fusion_mode == "serial_latent":
        return SerialLatentProgressModel(config)
    if config.fusion_mode == "layerwise_wvm":
        return LayerwiseProgressModel(config)
    raise ValueError(f"Unknown Progress fusion_mode={config.fusion_mode!r}")


def _pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_gap: float,
    temperature: float,
) -> torch.Tensor:
    target_delta = target[:, None] - target[None, :]
    valid = target_delta.abs() >= float(minimum_gap)
    if not valid.any():
        return prediction.new_zeros(())
    prediction_delta = prediction[:, None] - prediction[None, :]
    signed_delta = target_delta.sign() * prediction_delta
    return F.softplus(-signed_delta[valid] / max(float(temperature), 1e-6)).mean()


def build_progress_target_distribution(
    target: torch.Tensor,
    num_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    """Build Gaussian soft labels centered at (num_bins - 1) * progress."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    target = target.float().clamp(0.0, 1.0)
    target_frame = target * float(num_bins - 1)
    frame_indices = torch.arange(
        num_bins,
        device=target.device,
        dtype=target.dtype,
    )
    sigma = max(float(sigma_bins), 1e-6)
    distribution = torch.exp(
        -0.5 * ((frame_indices.unsqueeze(0) - target_frame.unsqueeze(1)) / sigma).pow(2)
    )
    return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def compute_progress_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    target_sigma_bins: float = 2.0,
    regression_weight: float = 1.0,
    ranking_weight: float = 0.1,
    ranking_minimum_gap: float = 0.05,
    ranking_temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    target = target.float().clamp(0.0, 1.0)
    logits = outputs["alignment_logits"].float()
    if logits.ndim != 2 or logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"Expected logits [B,F] aligned with target [B], got {tuple(logits.shape)} and {tuple(target.shape)}"
        )
    target_distribution = build_progress_target_distribution(
        target,
        num_bins=logits.shape[1],
        sigma_bins=target_sigma_bins,
    )
    alignment_loss = (
        -(target_distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    )

    prediction = outputs["progress"].float()
    regression_loss = F.smooth_l1_loss(prediction, target)
    ranking_loss = _pairwise_ranking_loss(
        prediction,
        target,
        minimum_gap=ranking_minimum_gap,
        temperature=ranking_temperature,
    )
    total_loss = (
        alignment_loss
        + float(regression_weight) * regression_loss
        + float(ranking_weight) * ranking_loss
    )
    target_frame = target * float(logits.shape[1] - 1)
    expected_frame = outputs.get(
        "expected_frame", prediction * float(logits.shape[1] - 1)
    ).float()
    matched_frame = outputs.get("matched_frame", logits.argmax(dim=-1)).float()
    return {
        "total_loss": total_loss,
        "alignment_loss": alignment_loss,
        "regression_loss": regression_loss,
        "ranking_loss": ranking_loss,
        "mae": (prediction - target).abs().mean().detach(),
        "rmse": (prediction - target).pow(2).mean().sqrt().detach(),
        "expected_frame_mae": (expected_frame - target_frame).abs().mean().detach(),
        "matched_frame_mae": (matched_frame - target_frame).abs().mean().detach(),
        "matched_within_one": ((matched_frame - target_frame).abs() <= 1.0)
        .float()
        .mean()
        .detach(),
    }
