"""Multi-view token adapters and WAN execution paths for Stage 1 bridge experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.attention import flash_attention
from wan.modules.model import rope_apply, sinusoidal_embedding_1d


MULTIVIEW_MODES = {
    "legacy",
    "rgb_mosaic_view_embedding",
    "rgb_mosaic_adapter_control",
    "latent_spatial",
    "latent_channel",
    "independent",
    "independent_consistency",
    "independent_adapter_control",
    "joint_attention",
    "cross_view_attention",
    "scene_tokens",
    "endpoint_scene_context",
    "autoregressive_high_to_wrist",
}

ARCHITECTURE_MODES = {
    "independent_adapter_control",
    "joint_attention",
    "cross_view_attention",
    "scene_tokens",
    "endpoint_scene_context",
    "autoregressive_high_to_wrist",
}

SEPARATE_VIEW_MODES = {
    "latent_spatial",
    "latent_channel",
    "independent",
    "independent_consistency",
    *ARCHITECTURE_MODES,
}


def add_token_residuals(*values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    result = None
    for value in values:
        if value is not None:
            result = value if result is None else result + value
    return result


def matched_control_bottleneck_dim(hidden_dim: int, adapter_dim: int) -> int:
    """Choose a plain MLP adapter width that matches one cross-view adapter."""
    hidden_dim = int(hidden_dim)
    adapter_dim = int(adapter_dim)
    if hidden_dim < 1 or adapter_dim < 1:
        raise ValueError("hidden_dim and adapter_dim must be positive")
    # CrossViewAdapter has 2*H*A + 4*A^2 + 3*H + 5*A parameters.
    # BottleneckResidual with width D has 2*H*D + 3*H + D parameters.
    numerator = 2 * hidden_dim * adapter_dim + 4 * adapter_dim**2 + 5 * adapter_dim
    return max(1, int(round(numerator / (2 * hidden_dim + 1))))


class MosaicViewEmbedding(nn.Module):
    """Add an explicit camera id to tokens produced from a legacy RGB mosaic."""

    def __init__(self, num_views: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.embedding = nn.Parameter(torch.zeros(self.num_views, hidden_dim))

    def forward(
        self,
        grid_sizes: torch.Tensor,
        seq_len: int,
        layout: str,
    ) -> torch.Tensor:
        if layout not in {"vertical", "horizontal"}:
            raise ValueError("Mosaic view embedding requires vertical or horizontal layout")
        outputs = []
        for grid in grid_sizes.tolist():
            grid_t, grid_h, grid_w = (int(value) for value in grid)
            if layout == "vertical":
                positions = torch.arange(grid_h, device=self.embedding.device)
                view_ids = torch.div(positions * self.num_views, grid_h, rounding_mode="floor")
                view_ids = view_ids.clamp_max(self.num_views - 1)
                ids = view_ids.view(1, grid_h, 1).expand(grid_t, grid_h, grid_w)
            else:
                positions = torch.arange(grid_w, device=self.embedding.device)
                view_ids = torch.div(positions * self.num_views, grid_w, rounding_mode="floor")
                view_ids = view_ids.clamp_max(self.num_views - 1)
                ids = view_ids.view(1, 1, grid_w).expand(grid_t, grid_h, grid_w)
            residual = F.embedding(ids.reshape(-1), self.embedding)
            if residual.shape[0] < seq_len:
                residual = torch.cat(
                    [residual, residual.new_zeros(seq_len - residual.shape[0], residual.shape[1])],
                    dim=0,
                )
            outputs.append(residual[:seq_len])
        return torch.stack(outputs, dim=0)


class BottleneckResidual(nn.Module):
    """Zero-initialized view-specific residual adapter."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.silu(self.down(self.norm(value))))


class ViewSpecificAdapters(nn.Module):
    def __init__(self, num_views: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [BottleneckResidual(hidden_dim, bottleneck_dim) for _ in range(num_views)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != len(self.adapters):
            raise ValueError(
                f"Expected {len(self.adapters)} views, got token shape {tuple(tokens.shape)}"
            )
        return torch.stack(
            [adapter(tokens[:, view_index]) for view_index, adapter in enumerate(self.adapters)],
            dim=1,
        )


class CrossViewAdapter(nn.Module):
    """Exchange information only between cameras at the same latent time."""

    def __init__(self, hidden_dim: int, adapter_dim: int, num_heads: int) -> None:
        super().__init__()
        if adapter_dim % num_heads != 0:
            raise ValueError("cross-view adapter_dim must be divisible by num_heads")
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.attention = nn.MultiheadAttention(adapter_dim, num_heads, batch_first=True)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        grid_t, grid_h, grid_w = (int(value) for value in grid_size)
        spatial_tokens = grid_h * grid_w
        if seq_len != grid_t * spatial_tokens:
            raise ValueError(
                f"Token length {seq_len} does not match grid {tuple(grid_size)}"
            )
        value = tokens.view(batch_size, num_views, grid_t, spatial_tokens, hidden_dim)
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch_size * grid_t, num_views * spatial_tokens, hidden_dim
        )
        reduced = self.down(self.norm(value))
        view_ids = torch.arange(num_views, device=tokens.device).repeat_interleave(spatial_tokens)
        same_view_mask = view_ids[:, None].eq(view_ids[None, :])
        attended, _ = self.attention(
            reduced,
            reduced,
            reduced,
            attn_mask=same_view_mask,
            need_weights=False,
        )
        value = value + self.up(attended)
        value = value.view(batch_size, grid_t, num_views, spatial_tokens, hidden_dim)
        return value.permute(0, 2, 1, 3, 4).reshape(batch_size, num_views, seq_len, hidden_dim)


class SceneTokenAdapter(nn.Module):
    """Use a small per-time scene bottleneck to communicate across cameras."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int,
        num_heads: int,
        num_scene_tokens: int,
    ) -> None:
        super().__init__()
        if adapter_dim % num_heads != 0:
            raise ValueError("scene adapter_dim must be divisible by num_heads")
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.scene_tokens = nn.Parameter(torch.randn(num_scene_tokens, adapter_dim) / math.sqrt(adapter_dim))
        self.scene_attention = nn.MultiheadAttention(adapter_dim, num_heads, batch_first=True)
        self.token_attention = nn.MultiheadAttention(adapter_dim, num_heads, batch_first=True)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        grid_t, grid_h, grid_w = (int(value) for value in grid_size)
        spatial_tokens = grid_h * grid_w
        if seq_len != grid_t * spatial_tokens:
            raise ValueError(
                f"Token length {seq_len} does not match grid {tuple(grid_size)}"
            )
        value = tokens.view(batch_size, num_views, grid_t, spatial_tokens, hidden_dim)
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch_size * grid_t, num_views * spatial_tokens, hidden_dim
        )
        reduced = self.down(self.norm(value))
        scene = self.scene_tokens.unsqueeze(0).expand(reduced.shape[0], -1, -1)
        scene, _ = self.scene_attention(scene, reduced, reduced, need_weights=False)
        attended, _ = self.token_attention(reduced, scene, scene, need_weights=False)
        value = value + self.up(attended)
        value = value.view(batch_size, grid_t, num_views, spatial_tokens, hidden_dim)
        return value.permute(0, 2, 1, 3, 4).reshape(batch_size, num_views, seq_len, hidden_dim)


class ContextTokenAdapter(nn.Module):
    """Inject endpoint or generated-high trajectory tokens through a zero-init cross adapter."""

    def __init__(self, hidden_dim: int, adapter_dim: int, num_heads: int) -> None:
        super().__init__()
        if adapter_dim % num_heads != 0:
            raise ValueError("context adapter_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.query_down = nn.Linear(hidden_dim, adapter_dim)
        self.key_down = nn.Linear(hidden_dim, adapter_dim)
        self.attention = nn.MultiheadAttention(adapter_dim, num_heads, batch_first=True)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_down(self.query_norm(tokens))
        context = self.key_down(self.context_norm(context_tokens))
        attended, _ = self.attention(query, context, context, need_weights=False)
        return tokens + self.up(attended)


class CrossViewConsistencyHead(nn.Module):
    """Match the same temporal stage across high/wrist predictions using symmetric InfoNCE."""

    def __init__(
        self,
        latent_channels: int,
        num_views: int,
        projection_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if num_views != 2:
            raise ValueError("The current consistency loss is defined for exactly two views")
        if temperature <= 0:
            raise ValueError("consistency temperature must be positive")
        self.projectors = nn.ModuleList(
            [nn.Linear(latent_channels, projection_dim) for _ in range(num_views)]
        )
        self.temperature = float(temperature)

    def forward(self, clean_prediction: torch.Tensor, unknown_mask: torch.Tensor) -> torch.Tensor:
        if clean_prediction.ndim != 6:
            raise ValueError(
                f"clean_prediction must be [B,V,C,T,H,W], got {tuple(clean_prediction.shape)}"
            )
        pooled = clean_prediction.mean(dim=(-1, -2)).permute(0, 1, 3, 2)
        if unknown_mask.ndim == 5:
            valid = unknown_mask[:, 0, :, 0, 0].bool()
        elif unknown_mask.ndim == 2:
            valid = unknown_mask.bool()
        else:
            raise ValueError(
                "unknown_mask must be [B,1,T,1,1] or [B,T], got "
                f"{tuple(unknown_mask.shape)}"
            )
        if valid.shape != pooled.shape[:1] + pooled.shape[2:3]:
            raise ValueError(
                f"unknown_mask temporal shape {tuple(valid.shape)} does not match "
                f"prediction {tuple(pooled.shape)}"
            )
        features = []
        for view_index, projector in enumerate(self.projectors):
            projected = projector(
                pooled[:, view_index].to(dtype=projector.weight.dtype)
            ).float()
            features.append(F.normalize(projected, dim=-1))

        # Negatives are temporal stages from the same episode. Treating another
        # episode at the same stage as a negative teaches task identity instead
        # of the intended high/wrist temporal alignment.
        episode_losses = []
        for batch_index in range(clean_prediction.shape[0]):
            episode_valid = valid[batch_index]
            high = features[0][batch_index, episode_valid]
            wrist = features[1][batch_index, episode_valid]
            if high.shape[0] == 0:
                continue
            if high.shape[0] == 1:
                episode_losses.append(1.0 - (high * wrist).sum(dim=-1).mean())
                continue
            logits = high @ wrist.transpose(0, 1) / self.temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            episode_losses.append(
                0.5
                * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.transpose(0, 1), labels)
                )
            )
        if not episode_losses:
            return sum(feature.sum() * 0.0 for feature in features)
        return torch.stack(episode_losses).mean()


class MultiViewWanRunner(nn.Module):
    """Execute a shared WAN with explicit camera-token structure."""

    def __init__(
        self,
        *,
        mode: str,
        num_views: int,
        hidden_dim: int,
        adapter_dim: int,
        adapter_heads: int,
        layer_indices: Iterable[int],
        num_scene_tokens: int,
        capacity_mode: str,
    ) -> None:
        super().__init__()
        if mode not in ARCHITECTURE_MODES:
            raise ValueError(f"Unsupported architecture mode {mode!r}")
        if capacity_mode not in {"shared", "view_adapter", "separate"}:
            raise ValueError(
                "Token-level multiview modes support shared, view_adapter, or separate capacity"
            )
        if capacity_mode == "separate" and mode != "cross_view_attention":
            raise ValueError("Separate token-level WAN capacity is currently defined only for MV7")
        self.mode = mode
        self.num_views = int(num_views)
        self.layer_indices = tuple(sorted({int(index) for index in layer_indices}))
        self.view_embedding = nn.Parameter(torch.zeros(self.num_views, hidden_dim))

        self.cross_adapters = nn.ModuleDict()
        self.scene_adapters = nn.ModuleDict()
        self.context_adapters = nn.ModuleDict()
        self.view_adapters = nn.ModuleDict()
        self.control_adapters = nn.ModuleDict()
        for layer_index in self.layer_indices:
            key = str(layer_index)
            if mode == "cross_view_attention":
                self.cross_adapters[key] = CrossViewAdapter(hidden_dim, adapter_dim, adapter_heads)
            elif mode == "scene_tokens":
                self.scene_adapters[key] = SceneTokenAdapter(
                    hidden_dim,
                    adapter_dim,
                    adapter_heads,
                    num_scene_tokens,
                )
            elif mode in {"endpoint_scene_context", "autoregressive_high_to_wrist"}:
                self.context_adapters[key] = ContextTokenAdapter(hidden_dim, adapter_dim, adapter_heads)
            elif mode == "independent_adapter_control":
                # Match the parameter count of down/up + bottleneck MHA in CrossViewAdapter.
                control_dim = matched_control_bottleneck_dim(hidden_dim, adapter_dim)
                self.control_adapters[key] = BottleneckResidual(hidden_dim, control_dim)
            if capacity_mode == "view_adapter":
                self.view_adapters[key] = ViewSpecificAdapters(
                    self.num_views,
                    hidden_dim,
                    adapter_dim,
                )

    @staticmethod
    def _uniform_grid(grid_sizes: torch.Tensor) -> tuple[int, int, int]:
        flat = grid_sizes.reshape(-1, 3)
        if not torch.equal(flat, flat[0:1].expand_as(flat)):
            raise ValueError("Multi-view batches currently require one shared latent grid size")
        return tuple(int(value) for value in flat[0].tolist())

    def _patchify(self, wan_model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 6:
            raise ValueError(f"Multi-view WAN input must be [B,V,C,T,H,W], got {tuple(inputs.shape)}")
        batch_size, num_views = inputs.shape[:2]
        if num_views > self.num_views:
            raise ValueError(f"Configured for {self.num_views} views, got {num_views}")
        flat = inputs.reshape(batch_size * num_views, *inputs.shape[2:])
        patches = wan_model.patch_embedding(flat)
        grid = torch.tensor(patches.shape[2:], device=patches.device, dtype=torch.long)
        grid_sizes = grid.view(1, 1, 3).expand(batch_size, num_views, 3).clone()
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2]), grid_sizes

    def _patchify_separate(
        self,
        wan_models: Sequence[nn.Module],
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 6 or inputs.shape[1] != len(wan_models):
            raise ValueError(
                "Separate-WAN input must be [B,V,C,T,H,W] with one model per view, got "
                f"inputs={tuple(inputs.shape)}, models={len(wan_models)}"
            )
        view_tokens = []
        view_grids = []
        for view_index, model in enumerate(wan_models):
            patches = model.patch_embedding(inputs[:, view_index])
            view_grids.append(
                torch.tensor(patches.shape[2:], device=patches.device, dtype=torch.long)
                .view(1, 3)
                .expand(inputs.shape[0], 3)
                .clone()
            )
            view_tokens.append(patches.flatten(2).transpose(1, 2))
        shapes = {tuple(tokens.shape[1:]) for tokens in view_tokens}
        if len(shapes) != 1:
            raise ValueError(f"Separate WANs produced incompatible token shapes: {sorted(shapes)}")
        return torch.stack(view_tokens, dim=1), torch.stack(view_grids, dim=1)

    @staticmethod
    def _context_embedding(wan_model: nn.Module, context: Sequence[torch.Tensor]) -> torch.Tensor:
        padded = []
        for item in context:
            item = item[: wan_model.text_len]
            if item.shape[0] < wan_model.text_len:
                item = torch.cat(
                    [item, item.new_zeros(wan_model.text_len - item.shape[0], item.shape[1])],
                    dim=0,
                )
            padded.append(item)
        return wan_model.text_embedding(torch.stack(padded))

    @staticmethod
    def _time_embedding(
        wan_model: nn.Module,
        timestep_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = timestep_tokens.shape
        with torch.amp.autocast("cuda", dtype=torch.float32):
            embedded = wan_model.time_embedding(
                sinusoidal_embedding_1d(wan_model.freq_dim, timestep_tokens.reshape(-1))
                .unflatten(0, shape)
                .float()
            )
            modulation = wan_model.time_projection(embedded).unflatten(-1, (6, wan_model.dim))
        return embedded, modulation

    @staticmethod
    def _select_endpoint_tokens(tokens: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
        batch_size, num_views, _, hidden_dim = tokens.shape
        grid_t, grid_h, grid_w = (int(value) for value in grid_size)
        spatial_tokens = grid_h * grid_w
        tokens = tokens.view(batch_size, num_views, grid_t, spatial_tokens, hidden_dim)
        temporal_indices = [0] if grid_t == 1 else [0, grid_t - 1]
        return tokens[:, :, temporal_indices].reshape(batch_size, -1, hidden_dim)

    @staticmethod
    def _head_unpatchify(
        wan_model: nn.Module,
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        flat_tokens = tokens.reshape(batch_size * num_views, seq_len, hidden_dim)
        flat_time = time_embedding.reshape(batch_size * num_views, seq_len, hidden_dim)
        prediction = wan_model.head(flat_tokens, flat_time)
        flat_grids = grid_sizes.reshape(batch_size * num_views, 3)
        prediction = wan_model.unpatchify(prediction, flat_grids)
        return torch.stack([item.float() for item in prediction], dim=0).view(
            batch_size, num_views, *prediction[0].shape
        )

    @staticmethod
    def _head_unpatchify_separate(
        wan_models: Sequence[nn.Module],
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for view_index, model in enumerate(wan_models):
            prediction = model.head(tokens[:, view_index], time_embedding[:, view_index])
            prediction = model.unpatchify(prediction, grid_sizes[:, view_index])
            outputs.append(torch.stack([item.float() for item in prediction], dim=0))
        return torch.stack(outputs, dim=1)

    def _apply_view_adapter(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        adapter = self.view_adapters[str(layer_index)] if str(layer_index) in self.view_adapters else None
        return adapter(tokens) if adapter is not None else tokens

    def forward_factorized(
        self,
        *,
        wan_model: nn.Module,
        inputs: torch.Tensor,
        timestep_tokens: torch.Tensor,
        context: Sequence[torch.Tensor],
        token_residual: Optional[torch.Tensor] = None,
        context_inputs: Optional[torch.Tensor] = None,
        view_offset: int = 0,
        wan_models: Optional[Sequence[nn.Module]] = None,
    ) -> torch.Tensor:
        separate_models = list(wan_models) if wan_models is not None else None
        if separate_models is not None:
            if self.mode != "cross_view_attention":
                raise ValueError("Separate WAN execution is currently supported only for MV7")
            if context_inputs is not None or view_offset != 0:
                raise ValueError("Separate MV7 requires all views in one pass without context_inputs")
            tokens, grid_sizes = self._patchify_separate(separate_models, inputs)
        else:
            tokens, grid_sizes = self._patchify(wan_model, inputs)
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        grid_size = self._uniform_grid(grid_sizes)
        if view_offset < 0 or view_offset + num_views > self.num_views:
            raise ValueError(
                f"view_offset={view_offset} with {num_views} inputs exceeds {self.num_views} configured views"
            )
        tokens = tokens + self.view_embedding[
            view_offset : view_offset + num_views
        ].view(1, num_views, 1, hidden_dim)
        if token_residual is not None:
            tokens = tokens + token_residual.to(device=tokens.device, dtype=tokens.dtype)

        if timestep_tokens.ndim == 1:
            timestep_tokens = timestep_tokens.view(batch_size, 1, 1).expand(batch_size, num_views, seq_len)
        if timestep_tokens.shape != (batch_size, num_views, seq_len):
            raise ValueError(
                "factorized timestep tokens must be [B,V,L], got "
                f"{tuple(timestep_tokens.shape)} for {(batch_size, num_views, seq_len)}"
            )
        if separate_models is None:
            time_embedding, time_modulation = self._time_embedding(wan_model, timestep_tokens)
            context_embedding = self._context_embedding(wan_model, context)
            flat_context = context_embedding.repeat_interleave(num_views, dim=0)
            separate_context = None
        else:
            time_parts = []
            modulation_parts = []
            context_parts = []
            for view_index, model in enumerate(separate_models):
                embedded, modulation = self._time_embedding(
                    model, timestep_tokens[:, view_index]
                )
                time_parts.append(embedded)
                modulation_parts.append(modulation)
                context_parts.append(self._context_embedding(model, context))
            time_embedding = torch.stack(time_parts, dim=1)
            time_modulation = torch.stack(modulation_parts, dim=1)
            separate_context = context_parts
            flat_context = None

        context_tokens = None
        if context_inputs is not None:
            context_tokens, context_grids = self._patchify(wan_model, context_inputs)
            context_grid = self._uniform_grid(context_grids)
            if self.mode == "endpoint_scene_context":
                context_tokens = self._select_endpoint_tokens(context_tokens, context_grid)
            else:
                context_tokens = context_tokens.reshape(batch_size, -1, hidden_dim)

        flat_grid_sizes = grid_sizes.reshape(batch_size * num_views, 3)
        seq_lens = torch.full(
            (batch_size * num_views,), seq_len, device=tokens.device, dtype=torch.long
        )
        flat_tokens = tokens.reshape(batch_size * num_views, seq_len, hidden_dim)
        flat_modulation = time_modulation.reshape(batch_size * num_views, seq_len, 6, hidden_dim)
        layer_count = len(wan_model.blocks) if separate_models is None else len(separate_models[0].blocks)
        if separate_models is not None and any(len(model.blocks) != layer_count for model in separate_models):
            raise ValueError("Separate WANs must have the same number of blocks")
        for layer_index in range(layer_count):
            if separate_models is None:
                block = wan_model.blocks[layer_index]
                flat_tokens = block(
                    flat_tokens,
                    e=flat_modulation,
                    seq_lens=seq_lens,
                    grid_sizes=flat_grid_sizes,
                    freqs=wan_model.freqs,
                    context=flat_context,
                    context_lens=None,
                )
                tokens = flat_tokens.view(batch_size, num_views, seq_len, hidden_dim)
            else:
                updated_views = []
                for view_index, model in enumerate(separate_models):
                    if model.freqs.device != model.patch_embedding.weight.device:
                        model.freqs = model.freqs.to(model.patch_embedding.weight.device)
                    view_seq_lens = torch.full(
                        (batch_size,), seq_len, device=tokens.device, dtype=torch.long
                    )
                    updated_views.append(
                        model.blocks[layer_index](
                            tokens[:, view_index],
                            e=time_modulation[:, view_index],
                            seq_lens=view_seq_lens,
                            grid_sizes=grid_sizes[:, view_index],
                            freqs=model.freqs,
                            context=separate_context[view_index],
                            context_lens=None,
                        )
                    )
                tokens = torch.stack(updated_views, dim=1)
            key = str(layer_index)
            if key in self.cross_adapters:
                tokens = self.cross_adapters[key](tokens, grid_size)
            if key in self.scene_adapters:
                tokens = self.scene_adapters[key](tokens, grid_size)
            if key in self.context_adapters:
                if context_tokens is None and self.mode == "endpoint_scene_context":
                    raise ValueError(f"{self.mode} requires context_inputs")
                if context_tokens is not None:
                    flattened = tokens.reshape(batch_size, num_views * seq_len, hidden_dim)
                    flattened = self.context_adapters[key](flattened, context_tokens)
                    tokens = flattened.view(batch_size, num_views, seq_len, hidden_dim)
            if key in self.control_adapters:
                flattened = tokens.reshape(batch_size * num_views, seq_len, hidden_dim)
                flattened = self.control_adapters[key](flattened)
                tokens = flattened.view(batch_size, num_views, seq_len, hidden_dim)
            if self.view_adapters and (view_offset != 0 or num_views != self.num_views):
                raise ValueError("view-specific capacity adapters require all views in one factorized pass")
            tokens = self._apply_view_adapter(layer_index, tokens)
            flat_tokens = tokens.reshape(batch_size * num_views, seq_len, hidden_dim)

        if separate_models is not None:
            return self._head_unpatchify_separate(
                separate_models, tokens, time_embedding, grid_sizes
            )
        return self._head_unpatchify(wan_model, tokens, time_embedding, grid_sizes)

    @staticmethod
    def _joint_block(
        block: nn.Module,
        tokens: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        combined = tokens.reshape(batch_size, num_views * seq_len, hidden_dim)
        combined_modulation = modulation.reshape(batch_size, num_views * seq_len, 6, hidden_dim)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            parts = (block.modulation.unsqueeze(0) + combined_modulation).chunk(6, dim=2)
        normalized = block.norm1(combined).float() * (1 + parts[1].squeeze(2)) + parts[0].squeeze(2)
        attention = block.self_attn
        num_heads = attention.num_heads
        head_dim = attention.head_dim
        q = attention.norm_q(attention.q(normalized)).view(
            batch_size, num_views, seq_len, num_heads, head_dim
        )
        k = attention.norm_k(attention.k(normalized)).view(
            batch_size, num_views, seq_len, num_heads, head_dim
        )
        v = attention.v(normalized).view(
            batch_size, num_views * seq_len, num_heads, head_dim
        )
        flat_grids = grid_sizes.reshape(batch_size * num_views, 3)
        q = rope_apply(q.reshape(batch_size * num_views, seq_len, num_heads, head_dim), flat_grids, freqs)
        k = rope_apply(k.reshape(batch_size * num_views, seq_len, num_heads, head_dim), flat_grids, freqs)
        q = q.view(batch_size, num_views * seq_len, num_heads, head_dim)
        k = k.view(batch_size, num_views * seq_len, num_heads, head_dim)
        combined_lens = torch.full(
            (batch_size,), num_views * seq_len, device=combined.device, dtype=torch.long
        )
        attended = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=combined_lens,
            window_size=attention.window_size,
        )
        attended = attention.o(attended.flatten(2))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            combined = combined + attended * parts[2].squeeze(2)
        combined = combined + block.cross_attn(block.norm3(combined), context, None)
        ffn = block.ffn(
            block.norm2(combined).float() * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2)
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            combined = combined + ffn * parts[5].squeeze(2)
        return combined.view(batch_size, num_views, seq_len, hidden_dim)

    def forward_joint(
        self,
        *,
        wan_model: nn.Module,
        inputs: torch.Tensor,
        timestep_tokens: torch.Tensor,
        context: Sequence[torch.Tensor],
        token_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, grid_sizes = self._patchify(wan_model, inputs)
        batch_size, num_views, seq_len, hidden_dim = tokens.shape
        tokens = tokens + self.view_embedding[:num_views].view(1, num_views, 1, hidden_dim)
        if token_residual is not None:
            tokens = tokens + token_residual.to(device=tokens.device, dtype=tokens.dtype)
        if timestep_tokens.ndim == 1:
            timestep_tokens = timestep_tokens.view(batch_size, 1, 1).expand(batch_size, num_views, seq_len)
        if timestep_tokens.shape != (batch_size, num_views, seq_len):
            raise ValueError(
                f"joint timestep tokens must be [B,V,L], got {tuple(timestep_tokens.shape)}"
            )
        time_embedding, modulation = self._time_embedding(wan_model, timestep_tokens)
        context_embedding = self._context_embedding(wan_model, context)
        for layer_index, block in enumerate(wan_model.blocks):
            tokens = self._joint_block(
                block,
                tokens,
                modulation,
                context_embedding,
                grid_sizes,
                wan_model.freqs,
            )
            tokens = self._apply_view_adapter(layer_index, tokens)
        return self._head_unpatchify(wan_model, tokens, time_embedding, grid_sizes)
