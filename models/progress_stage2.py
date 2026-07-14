"""Progress models over a frozen endpoint-conditioned VGM trajectory memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 维度约定：E=episode/trajectory batch，Q=每条 episode 的 query 数，B_q=sum(Q)，
# F=轨迹帧数（正式配置为 53），C_z=VAE latent 通道，T_z/H_z/W_z=latent
# 时间/空间尺寸，K=每帧重采样 token 数（正式配置为 16），D=Progress hidden dim
# （正式配置为 512），L=WAN 视频 token 数。

# 两个模型共享同一个任务定义：输入一张当前观测及同一条 53-frame 生成轨迹，
# 输出当前观测对 0...52 每个轨迹位置的匹配 logits [B,53]。它们的区别只在于
# 当前观测与轨迹信息如何融合，不在标签、输出空间或最终 53-frame memory。


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
    view_encoding_mode: str = "joint"
    num_views: int = 1
    tokens_per_view: int = 16
    alignment_head_mode: str = "pooled_cosine"
    late_interaction_temperature: float = 0.07
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

        # latent: [B, C_z, T_z, H_z, W_z]
        # features: [B, D, G_t, G_h, G_w]
        features = self.proj(latent)
        _, hidden_dim, grid_t, grid_h, grid_w = features.shape
        # tokens: [B, G_t, G_h, G_w, D]
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
        # 返回 tokens: [B, G_t, G_h*G_w, D]。
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
        # 输入 tokens: [B, S, D]；learned queries: [B, K, D]；输出: [B, K, D]。
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
        # progress_tokens: [B, K, D]；memory_tokens: [B, F*K, D]。
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
        self,
        progress_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        query_episode_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # progress_tokens: [B_q,K,D]；video_tokens: [E,L,D_video]。
        # 只有 Progress query 被更新，WAN video token 不会被改写。
        progress_input = self.progress_norm(progress_tokens)
        # 先按 E 条 episode 做昂贵的 D_video->D 投影：[E,L,D_video] -> [E,L,D]，
        # 再映射到 B_q 个 query；不能先 expand/index_select 再重复执行 Q 次投影。
        video_input = self.video_projection(self.video_norm(video_tokens))
        video_input = _select_episode_memory_for_queries(
            video_input,
            query_batch_size=progress_input.shape[0],
            query_episode_indices=query_episode_indices,
            name="Projected WAN hidden memory",
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


def _alignment_outputs_from_logits(
    logits: torch.Tensor, num_bins: int
) -> Dict[str, torch.Tensor]:
    """Convert frame-alignment logits into the shared Progress output contract."""
    probabilities = logits.softmax(dim=-1)
    frame_indices = torch.arange(
        num_bins,
        device=logits.device,
        dtype=probabilities.dtype,
    )
    positions = frame_indices / float(num_bins - 1)
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
        # query_tokens: [B, K, D_query]；temporal_memory: [B, F=53, D_memory]。
        # 必须保证 memory 的第二维就是 53；否则 logits 的列不再对应明确的第 0...52 帧。
        if temporal_memory.ndim != 3 or temporal_memory.shape[1] != self.num_bins:
            raise ValueError(
                "Temporal memory must contain one slot per generated frame: "
                f"expected [B,{self.num_bins},D], got {tuple(temporal_memory.shape)}"
            )
        # 沿 K 个 query token 求均值：[B,K,D_query] -> [B,D_query]。
        query = query_tokens.mean(dim=1)
        # 只在最后一维做 LayerNorm，形状保持 [B,D_query]。
        query = self.query_norm(query)
        # 投影到共同匹配空间：[B,D_query] -> [B,D_hidden]。
        query = self.query_proj(query)
        # 对 53 个 memory slot 分别做 LayerNorm，形状保持 [B,F,D_memory]。
        memory = self.memory_norm(temporal_memory)
        # 把每个 memory slot 投影到同一匹配空间：[B,F,D_memory] -> [B,F,D_hidden]。
        memory = self.memory_proj(memory)
        # 转 float32 后沿最后一维做 L2 normalize；query 形状保持 [B,D_hidden]。
        query = F.normalize(query.float(), dim=-1)
        # memory 形状保持 [B,F,D_hidden]；之后点积就是 cosine similarity。
        memory = F.normalize(memory.float(), dim=-1)
        # logit_scale 是一个可学习标量；exp 保证为正，clamp 防止训练后数值过大。
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        # 对每个 query，先把 K 个 query token 平均成一个向量，再分别和 53 个逐帧
        # memory 向量计算归一化点积；因此每一列都具有明确的“生成轨迹第 k 帧”语义。
        # logits/probabilities: [B, F=53]，每一列对应生成轨迹中的一张图片。
        logits = torch.einsum("bd,btd->bt", query, memory) * scale
        return _alignment_outputs_from_logits(logits, self.num_bins)


class TokenLateInteractionAlignmentHead(nn.Module):
    """Match local query tokens to the best local token in every frame slot."""

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        hidden_dim: int,
        num_bins: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("Progress matching requires at least two frame bins")
        if temperature <= 0.0:
            raise ValueError("late_interaction_temperature must be positive")
        self.num_bins = int(num_bins)
        self.temperature = float(temperature)
        self.query_norm = nn.LayerNorm(query_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.query_importance = nn.Linear(query_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self, query_tokens: torch.Tensor, frame_memory_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # query_tokens: [B,K_q,D_q]；frame_memory_tokens: [B,F,K_m,D_m]。
        if query_tokens.ndim != 3:
            raise ValueError(
                f"Query tokens must be [B,K,D], got {tuple(query_tokens.shape)}"
            )
        if (
            frame_memory_tokens.ndim != 4
            or frame_memory_tokens.shape[1] != self.num_bins
        ):
            raise ValueError(
                "Token memory must preserve every frame and local token: "
                f"expected [B,{self.num_bins},K,D], got "
                f"{tuple(frame_memory_tokens.shape)}"
            )
        if frame_memory_tokens.shape[0] != query_tokens.shape[0]:
            raise ValueError(
                "Query and frame-token memory batches must match, got "
                f"{query_tokens.shape[0]} and {frame_memory_tokens.shape[0]}"
            )

        normalized_query = self.query_norm(query_tokens)
        normalized_memory = self.memory_norm(frame_memory_tokens)
        query = F.normalize(self.query_proj(normalized_query).float(), dim=-1)
        memory = F.normalize(self.memory_proj(normalized_memory).float(), dim=-1)

        # 每个 query token 和每个候选帧中的全部 local token 比较：
        # [B,K_q,D] x [B,F,K_m,D] -> [B,F,K_q,K_m]。
        similarities = torch.einsum("bqd,bfkd->bfqk", query, memory)
        memory_token_count = similarities.shape[-1]
        temperature = self.temperature
        # 归一化 smooth-max 近似“该 query 局部特征在这一帧中最匹配的位置”。
        # 减去 log(K_m) 只去除 token 数带来的常数偏置，不改变帧间排序。
        local_scores = temperature * (
            torch.logsumexp(similarities / temperature, dim=-1)
            - math.log(memory_token_count)
        )
        # 学习哪些 query token 更值得信任，而不是先把夹爪/物体等局部 token 平均掉。
        query_weights = self.query_importance(normalized_query).squeeze(-1).float()
        query_weights = query_weights.softmax(dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = torch.einsum("bfq,bq->bf", local_scores, query_weights) * scale

        outputs = _alignment_outputs_from_logits(logits, self.num_bins)
        # 诊断字段不参与 loss，可用于检查模型最终关注了哪些局部 query token。
        outputs["query_token_weights"] = query_weights
        outputs["late_interaction_scores"] = local_scores
        return outputs


class SharedFrameLatentEncoder(nn.Module):
    """Encode current and generated RGB-frame latents with shared weights."""

    def __init__(self, config: ProgressStage2Config):
        super().__init__()
        self.view_encoding_mode = str(config.view_encoding_mode)
        self.num_views = int(config.num_views)
        if self.view_encoding_mode not in {"joint", "split_height"}:
            raise ValueError(
                "view_encoding_mode must be 'joint' or 'split_height', got "
                f"{self.view_encoding_mode!r}"
            )
        if self.view_encoding_mode == "split_height" and self.num_views < 2:
            raise ValueError("split_height requires num_views >= 2")
        self.tokenizer = LatentPatchTokenizer(
            config.latent_channels,
            config.hidden_dim,
            config.patch_size,
        )
        resampler_tokens = (
            int(config.frame_num_tokens)
            if self.view_encoding_mode == "joint"
            else int(config.tokens_per_view)
        )
        self.resampler = AttentionResampler(
            config.hidden_dim,
            config.num_heads,
            resampler_tokens,
            config.dropout,
        )
        if self.view_encoding_mode == "split_height":
            self.view_embeddings = nn.Parameter(
                torch.empty(1, self.num_views, 1, config.hidden_dim)
            )
            nn.init.normal_(self.view_embeddings, std=0.02)
            self.output_num_tokens = self.num_views * resampler_tokens
        else:
            # Baseline mode intentionally creates no additional parameter, so old
            # pooled-cosine checkpoints remain strict-load compatible.
            self.output_num_tokens = resampler_tokens

    def _encode_one_view(self, latent: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.tokenizer(latent)
        if grid[0] != 1:
            raise ValueError(
                f"Current observation must encode to one temporal latent, got grid={grid}"
            )
        return self.resampler(tokens.flatten(1, 2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # 单帧 latent 输入: [B, C_z, 1, H_z, W_z]。
        # 当前真实帧和 53 张生成帧都调用这个同权重 encoder，避免两侧使用不同特征空间；
        # VAE 已在缓存阶段冻结执行，这里训练的是 patch tokenizer 和 attention resampler。
        # tokenizer 输出 tokens: [B,G_t,S_spatial,D] 和 grid=(G_t,G_h,G_w)；
        # 单帧输入且 patch_t=1 时 G_t=1，S_spatial=G_h*G_w。
        if latent.ndim != 5:
            raise ValueError(
                f"Expected [B,C,1,H,W] frame latent, got {tuple(latent.shape)}"
            )
        if self.view_encoding_mode == "joint":
            # Baseline path is unchanged: the vertically concatenated views compete
            # for one shared set of K learned resampler tokens.
            return self._encode_one_view(latent)

        height = latent.shape[-2]
        if height % self.num_views:
            raise ValueError(
                f"Latent height {height} is not divisible by num_views={self.num_views}"
            )
        # The LIBERO composite is vertically concatenated. Encode each view with
        # the same tokenizer/resampler, but guarantee K_view tokens for every view.
        view_batch = torch.cat(latent.chunk(self.num_views, dim=-2), dim=0)
        encoded_views = self._encode_one_view(view_batch)
        batch_size = latent.shape[0]
        tokens_per_view = encoded_views.shape[1]
        encoded_views = encoded_views.reshape(
            self.num_views,
            batch_size,
            tokens_per_view,
            encoded_views.shape[-1],
        ).permute(1, 0, 2, 3)
        encoded_views = encoded_views + self.view_embeddings
        return encoded_views.reshape(
            batch_size,
            self.num_views * tokens_per_view,
            encoded_views.shape[-1],
        )


def encode_trajectory_frame_memory(
    frame_encoder: SharedFrameLatentEncoder,
    trajectory_frame_latents: torch.Tensor,
    *,
    num_progress_bins: int,
    type_embedding: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode independently-VAE-encoded generated frames into 53 ordered slots."""
    # 输入 trajectory_frame_latents: [B, F=53, C_z, 1, H_z, W_z]。
    # rank 必须为 6：episode batch、53 帧、latent 通道、单帧时间维、两维空间。
    if trajectory_frame_latents.ndim != 6:
        raise ValueError(
            "trajectory_frame_latents must be [B,F,C,1,H,W], got "
            f"{tuple(trajectory_frame_latents.shape)}"
        )
    # 仅给各维命名，不改变 tensor：B,F,C_z,T_single,H_z,W_z。
    batch_size, frames, channels, latent_time, height, width = (
        trajectory_frame_latents.shape
    )
    # 第二维必须等于分类 bin 数；正式任务要求 F=num_progress_bins=53。
    if frames != num_progress_bins:
        raise ValueError(
            f"Expected exactly {num_progress_bins} generated frame slots, got {frames}"
        )
    # 每个 slot 必须来自“单张图片独立 VAE encode”，所以 T_single 必须严格等于 1。
    if latent_time != 1:
        raise ValueError(
            "Each generated frame must be VAE-encoded independently to one temporal latent, "
            f"got latent_time={latent_time}"
        )

    # 合并 B 和 F，方便一次调用共享帧编码器：
    # [B,F,C_z,1,H_z,W_z] -> [B*F,C_z,1,H_z,W_z]。
    flat_frame_latents = trajectory_frame_latents.reshape(
        batch_size * frames,
        channels,
        latent_time,
        height,
        width,
    )
    # 53 张生成帧使用和 current query 相同的 frame_encoder；输出 [B*F,K,D]。
    frame_tokens = frame_encoder(flat_frame_latents)
    # K=tokens_per_frame，正式配置为 16。
    tokens_per_frame = frame_tokens.shape[1]
    # D=hidden_dim，正式配置为 512。
    hidden_dim = frame_tokens.shape[2]
    # 恢复 53 帧结构: [B, F, K, D]。
    frame_tokens = frame_tokens.reshape(
        batch_size,
        frames,
        tokens_per_frame,
        hidden_dim,
    )
    # 先生成 [F,D] 的确定性 sin/cos 帧位置，再增加 batch/token 广播维得到 [1,F,1,D]。
    frame_positions = _axis_sincos(
        frames,
        hidden_dim,
        frame_tokens.device,
        frame_tokens.dtype,
    )[None, :, None, :]
    # frame_positions: [1, F, 1, D]，明确区分轨迹中的第 0...52 帧。
    # type_embedding 原为 [1,1,D]，增加帧维后 [1,1,1,D]；标记这些 token 来自 memory。
    # 两种 embedding 都通过广播加入 [B,F,K,D]，形状保持不变。
    frame_tokens = frame_tokens + frame_positions + type_embedding[:, None, :, :]
    # 合并 F 和 K，供 cross-attention 使用：all_memory_tokens [B,F*K,D]。
    all_memory_tokens = frame_tokens.flatten(1, 2)
    # 对每帧的 K 个 token 求均值，保留 53 个显式 slot：temporal_memory [B,F,D]。
    temporal_memory = frame_tokens.mean(dim=2)
    return all_memory_tokens, temporal_memory


def _select_episode_memory_for_queries(
    memory: torch.Tensor,
    *,
    query_batch_size: int,
    query_episode_indices: Optional[torch.Tensor],
    name: str,
) -> torch.Tensor:
    """Map E episode memories to B_q queries without cross-episode mixing."""
    episode_batch_size = memory.shape[0]
    if query_episode_indices is None:
        # 兼容旧的 1:N 路径：一条 episode memory 广播给当前 forward 的全部 query。
        if episode_batch_size == 1 and query_batch_size > 1:
            return memory.expand(query_batch_size, *([-1] * (memory.ndim - 1)))
        # 也兼容严格的一一对应 E:E，例如 64 条轨迹分别对应 64 个 query。
        if episode_batch_size == query_batch_size:
            return memory
        raise ValueError(
            f"{name} batch must be one or match the query batch when "
            "query_episode_indices is omitted"
        )

    # query_episode_indices: [B_q]，第 q 个整数指定 query q 应读取第几条 episode memory。
    if query_episode_indices.ndim != 1 or query_episode_indices.shape[0] != query_batch_size:
        raise ValueError(
            "query_episode_indices must be [B_q], got "
            f"{tuple(query_episode_indices.shape)} for B_q={query_batch_size}"
        )
    indices = query_episode_indices.to(device=memory.device, dtype=torch.long)
    if indices.numel() > 0 and (
        int(indices.min()) < 0 or int(indices.max()) >= episode_batch_size
    ):
        raise ValueError(
            f"query_episode_indices must be in [0,{episode_batch_size - 1}] for {name}"
        )
    # [E,...] -> [B_q,...]。index_select 只按 episode 维取对应 memory；不同 query
    # 在 Transformer 的 batch 维仍彼此独立，但同一 episode 的 Q 个 query 会把梯度
    # 累加回同一份 memory encoding。
    return memory.index_select(0, indices)


def _build_alignment_head(config: ProgressStage2Config) -> nn.Module:
    if config.alignment_head_mode == "pooled_cosine":
        return TemporalAlignmentHead(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.num_progress_bins,
        )
    if config.alignment_head_mode == "token_late_interaction":
        return TokenLateInteractionAlignmentHead(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.num_progress_bins,
            config.late_interaction_temperature,
        )
    raise ValueError(
        "alignment_head_mode must be 'pooled_cosine' or "
        f"'token_late_interaction', got {config.alignment_head_mode!r}"
    )


def _alignment_memory(
    config: ProgressStage2Config,
    memory_tokens: torch.Tensor,
    temporal_memory: torch.Tensor,
) -> torch.Tensor:
    if config.alignment_head_mode == "pooled_cosine":
        return temporal_memory
    if memory_tokens.shape[1] % config.num_progress_bins:
        raise ValueError(
            "Flattened trajectory memory cannot be restored to frame slots: "
            f"shape={tuple(memory_tokens.shape)}, bins={config.num_progress_bins}"
        )
    tokens_per_frame = memory_tokens.shape[1] // config.num_progress_bins
    return memory_tokens.reshape(
        memory_tokens.shape[0],
        config.num_progress_bins,
        tokens_per_frame,
        memory_tokens.shape[-1],
    )


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
        self.alignment_head = _build_alignment_head(config)

    def forward(
        self,
        current_latent: torch.Tensor,
        trajectory_frame_latents: torch.Tensor,
        query_episode_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Serial 数据流：
        # current 单帧 -> SharedFrameLatentEncoder -> [B,K,D]
        # 53-frame memory -> 同一个 SharedFrameLatentEncoder -> [B,F*K,D]
        # -> 6 层 self-attention + cross-attention -> TemporalAlignmentHead -> [B,53]。
        # 原始 [B,C_z,T_z=14,H_z,W_z] trajectory_latent 不进入本模型。
        # current_latent: [B_q, C_z, 1, H_z, W_z]
        # trajectory_frame_latents: [E, F=53, C_z, 1, H_z, W_z]
        # query_episode_indices: [B_q]；例如 E=2,Q=3 时为 [0,0,0,1,1,1]。
        # 同一个共享帧编码器把 B 张真实 current observation 编为 [B,K,D]。
        progress_tokens = self.frame_encoder(current_latent)
        # current_type_embedding: [1,1,D]，广播到 [B,K,D]，标记这些 token 属于查询侧。
        progress_tokens = progress_tokens + self.current_type_embedding
        # 返回 E 条轨迹的 memory_tokens [E,F*K,D] 与 temporal_memory [E,F,D]。
        memory_tokens, temporal_memory = encode_trajectory_frame_memory(
            self.frame_encoder,
            trajectory_frame_latents,
            num_progress_bins=self.config.num_progress_bins,
            type_embedding=self.memory_type_embedding,
        )
        # 按 [B_q] 显式索引把 E 条轨迹映射到 E*Q 个 query，得到 [B_q,F*K,D]
        # 和 [B_q,F,D]。这一步保证 query(e,q) 只读取 memory(e)。
        memory_tokens = _select_episode_memory_for_queries(
            memory_tokens,
            query_batch_size=progress_tokens.shape[0],
            query_episode_indices=query_episode_indices,
            name="Trajectory token memory",
        )
        temporal_memory = _select_episode_memory_for_queries(
            temporal_memory,
            query_batch_size=progress_tokens.shape[0],
            query_episode_indices=query_episode_indices,
            name="Trajectory temporal memory",
        )

        # 6 层 Serial Transformer: [B, K, D] cross-attend [B, F*K, D]。
        # 每层输入/输出 progress_tokens 都保持 [B,K,D]；memory_tokens 始终只读、不被更新。
        for block in self.blocks:
            progress_tokens = block(progress_tokens, memory_tokens)
        # Baseline 对每帧 K 个 token 先平均；late-interaction ablation 则恢复
        # [B,F,K,D]，让局部 query token 分别寻找每帧内最匹配的局部 token。
        alignment_memory = _alignment_memory(
            self.config,
            memory_tokens,
            temporal_memory,
        )
        outputs = self.alignment_head(progress_tokens, alignment_memory)
        return outputs


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
        self.alignment_head = _build_alignment_head(config)

    def forward(
        self,
        current_latent: torch.Tensor,
        trajectory_frame_latents: torch.Tensor,
        video_hidden_states: Sequence[torch.Tensor],
        video_grid_size: Sequence[int] | torch.Tensor,
        query_episode_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Layerwise/WVM-style 数据流：
        # current 单帧 -> SharedFrameLatentEncoder -> [B,K,D]
        # -> 第 j 个 Progress block 读取冻结 WAN 第 j 层 hidden [1或B,L,D_video]
        # -> 读完 30 层后，再和显式 53-frame memory [B,F,D] 做最终匹配 -> [B,53]。
        # 信息流是非对称的：Progress 可以读 WAN，WAN token 不读 Progress，冻结 WAN
        # 也不会收到梯度。它借鉴 WVM 的逐层读取方式，但不是 WVM 4F 模型的参数级复刻。
        # current_latent: [B_q, C_z, 1, H_z, W_z]
        # trajectory_frame_latents: [E, F=53, C_z, 1, H_z, W_z]
        # video_hidden_states: 长度为 30 的列表，每项 [E,L,D_video]。
        # query_episode_indices: [B_q]，负责把每个 query 映射回对应的 episode。
        # 一层 Progress block 必须对应一层 WAN hidden；正式配置二者都为 30。
        if len(video_hidden_states) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} video hidden states, got {len(video_hidden_states)}"
            )
        # video_grid_size 若为 tensor，形状应为 [3]，值依次是 (G_t,G_h,G_w)。
        if isinstance(video_grid_size, torch.Tensor):
            # detach/cpu/tolist 只为得到 Python tuple，不参与梯度。
            grid = tuple(
                int(value) for value in video_grid_size.detach().cpu().tolist()
            )
        else:
            # 已是 sequence 时逐项转 int，得到同样的三元 tuple。
            grid = tuple(int(value) for value in video_grid_size)
        # 三个网格轴缺一不可；后面 L=G_t*G_h*G_w。
        if len(grid) != 3:
            raise ValueError(f"video_grid_size must contain (T, H, W), got {grid}")

        # 当前真实单帧：[B,C_z,1,H_z,W_z] -> frame_encoder -> [B,K,D]。
        progress_tokens = self.frame_encoder(current_latent)
        # 加 current 类型标记，形状仍为 [B,K,D]。
        progress_tokens = progress_tokens + self.current_type_embedding
        # query_batch=B_q；query_episode_indices 决定它们分别读取哪条 WAN memory。
        query_batch = progress_tokens.shape[0]
        # valid_length=L=G_t*G_h*G_w，只保留网格内真实 token，排除 seq_len padding。
        valid_length = math.prod(grid)
        # 用于防止空 hidden-state 列表意外跳过整个 Layerwise 融合过程。
        saw_video_hidden = False
        # 第 j 个 Progress block 读取第 j 个冻结 WAN block 的 hidden state。
        for block, video_hidden in zip(self.blocks, video_hidden_states):
            # video_hidden: [E,seq_len,D_video]；seq_len 至少应覆盖 valid_length=L。
            if video_hidden.shape[1] < valid_length:
                raise ValueError(
                    f"Video hidden has {video_hidden.shape[1]} tokens but grid={grid} requires {valid_length}"
                )
            # 去掉右侧 padding token：[1或B,seq_len,D_video] -> [1或B,L,D_video]；
            # detach 明确阻断梯度写回冻结 WAN。
            video_hidden = video_hidden[:, :valid_length].detach()
            # block 内先把 E 条 WAN hidden 各投影一次，再按 query_episode_indices 映射到
            # B_q 个 query；输出仍为 [B_q,K,D]，video_hidden 本身不被修改。
            progress_tokens = block(
                progress_tokens,
                video_hidden,
                query_episode_indices=query_episode_indices,
            )
            saw_video_hidden = True

        if not saw_video_hidden:
            raise RuntimeError(
                "Layerwise Progress requires at least one Video hidden state"
            )
        # Layerwise 读取完 WAN 特征后，编码同一份显式 53-frame memory。
        memory_tokens, temporal_memory = encode_trajectory_frame_memory(
            self.frame_encoder,
            trajectory_frame_latents,
            num_progress_bins=self.config.num_progress_bins,
            type_embedding=self.memory_type_embedding,
        )
        if self.config.alignment_head_mode == "token_late_interaction":
            memory_tokens = _select_episode_memory_for_queries(
                memory_tokens,
                query_batch_size=query_batch,
                query_episode_indices=query_episode_indices,
                name="Trajectory-frame token memory",
            )
        # [E,F,D] -> [B_q,F,D]，和 WAN hidden 使用完全相同的 query-to-episode 配对。
        temporal_memory = _select_episode_memory_for_queries(
            temporal_memory,
            query_batch_size=query_batch,
            query_episode_indices=query_episode_indices,
            name="Trajectory-frame temporal memory",
        )
        alignment_memory = _alignment_memory(
            self.config,
            memory_tokens,
            temporal_memory,
        )
        outputs = self.alignment_head(progress_tokens, alignment_memory)
        return outputs


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
    episode_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if episode_ids is not None:
        if episode_ids.ndim != 1 or episode_ids.shape[0] != target.shape[0]:
            raise ValueError(
                "episode_ids must be [B] and align with target, got "
                f"{tuple(episode_ids.shape)} and {tuple(target.shape)}"
            )
        episode_ids = episode_ids.to(device=target.device, dtype=torch.long)
        # 混合 batch 中只在同一 episode 内构造先后 pair。每个 episode 独立求均值后
        # 再对 E 条 episode 求均值，避免 query 数或有效 pair 数更多的轨迹占更大权重。
        group_losses = []
        for episode_id in torch.unique(episode_ids):
            group_mask = episode_ids == episode_id
            group_losses.append(
                _pairwise_ranking_loss(
                    prediction[group_mask],
                    target[group_mask],
                    minimum_gap=minimum_gap,
                    temperature=temperature,
                    episode_ids=None,
                )
            )
        if not group_losses:
            return prediction.new_zeros(())
        return torch.stack(group_losses).mean()

    # target: [B]；两次增加广播维后 target_delta: [B,B]，元素 (i,j)=target_i-target_j。
    target_delta = target[:, None] - target[None, :]
    # valid: [B,B] bool；只比较真实进度差至少为 minimum_gap 的 query 对。
    valid = target_delta.abs() >= float(minimum_gap)
    # 当前 micro-batch 没有有效先后关系时返回标量 0，避免对空 tensor 求 mean 得到 NaN。
    if not valid.any():
        return prediction.new_zeros(())
    # prediction_delta: [B,B]，与 target_delta 的每个 pair 一一对应。
    prediction_delta = prediction[:, None] - prediction[None, :]
    # target_delta.sign 决定正确排序方向；正确顺序时 signed_delta 应为正。
    signed_delta = target_delta.sign() * prediction_delta
    # softplus(-margin/temperature) 是平滑 pairwise ranking loss；最终对有效 pair 求标量均值。
    return F.softplus(-signed_delta[valid] / max(float(temperature), 1e-6)).mean()


def build_progress_target_distribution(
    target: torch.Tensor,
    num_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    """Build Gaussian soft labels centered at (num_bins - 1) * progress."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    # 转 float32 并限制到合法进度区间；target 形状保持 [B]。
    target = target.float().clamp(0.0, 1.0)
    # target 来自 source episode 的归一化时间，不是视觉 nearest-neighbor 标签。
    # target: [B]、范围 [0,1]；target_frame: [B]、范围 [0,F-1]。
    # F=53 时，progress=0/0.5/1 分别对应第 0/26/52 帧。
    target_frame = target * float(num_bins - 1)
    # frame_indices: [F] = [0,1,...,F-1]，放在和 target 相同的 device/dtype。
    frame_indices = torch.arange(
        num_bins,
        device=target.device,
        dtype=target.dtype,
    )
    # sigma 是以“帧/bin”为单位的标量；下限防止除零。正式配置 sigma=2。
    sigma = max(float(sigma_bins), 1e-6)
    # distribution: [B, F]，每行是以 target_frame 为中心的 Gaussian soft label。
    # frame_indices[None,:]: [1,F]，target_frame[:,None]: [B,1]，广播差值为 [B,F]。
    distribution = torch.exp(
        -0.5 * ((frame_indices.unsqueeze(0) - target_frame.unsqueeze(1)) / sigma).pow(2)
    )
    # 每行沿 F 维归一化，使 soft target 总和为 1；normalizer: [B,1]。
    normalizer = distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    distribution = distribution / normalizer
    return distribution


def compute_progress_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    episode_ids: Optional[torch.Tensor] = None,
    target_sigma_bins: float = 2.0,
    regression_weight: float = 1.0,
    ranking_weight: float = 0.1,
    ranking_minimum_gap: float = 0.05,
    ranking_temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    # outputs["alignment_logits"]: [B,F=53]；target: [B]。
    # target 转 float32 并截断到 [0,1]，形状仍为 [B]。
    target = target.float().clamp(0.0, 1.0)
    # logits 转 float32 提高 loss 数值稳定性，形状保持 [B,F]。
    logits = outputs["alignment_logits"].float()
    # 同时检查 rank=2 和 batch 对齐；F 可由配置决定，但正式值必须为 53。
    if logits.ndim != 2 or logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"Expected logits [B,F] aligned with target [B], got {tuple(logits.shape)} and {tuple(target.shape)}"
        )
    # 根据每个 target 构造 Gaussian soft label: [B] -> [B,F]。
    target_distribution = build_progress_target_distribution(
        target,
        num_bins=logits.shape[1],
        sigma_bins=target_sigma_bins,
    )
    # 在 F=53 维做 log_softmax，log_probabilities: [B,F]。
    log_probabilities = F.log_softmax(logits, dim=-1)
    # soft-label cross entropy 先沿 F 求和得到 [B]，再对 B 求均值得到标量。
    alignment_loss = -(target_distribution * log_probabilities).sum(dim=-1).mean()

    # prediction 是 alignment distribution 的期望 Progress，形状 [B]、范围 [0,1]。
    prediction = outputs["progress"].float()
    # Smooth-L1 直接约束连续 progress prediction 与 target，默认 reduction=mean，输出标量。
    regression_loss = F.smooth_l1_loss(prediction, target)
    ranking_weight = float(ranking_weight)
    if ranking_weight < 0.0:
        raise ValueError("ranking_weight cannot be negative")
    # 当前正式实验关闭 ranking。权重为 0 时直接跳过 O(B^2) pairwise 计算，
    # 同时保留返回字段，保证训练日志和旧调用接口兼容。
    if ranking_weight == 0.0:
        ranking_loss = prediction.new_zeros(())
    else:
        ranking_loss = _pairwise_ranking_loss(
            prediction,
            target,
            minimum_gap=ranking_minimum_gap,
            temperature=ranking_temperature,
            episode_ids=episode_ids,
        )
    # 两项正式监督和一项可选监督的语义：
    # 1. alignment_loss 让 [B,53] 分布集中在目标帧附近（sigma=2 的软标签）；
    # 2. regression_loss 约束分布期望得到的连续 Progress；
    # 3. ranking_loss 是可选项，权重为 0 时不参与当前实验。
    # total_loss 是标量：53-bin 对齐交叉熵 + 连续 Progress Smooth-L1 + 可选排序损失。
    total_loss = (
        alignment_loss
        + float(regression_weight) * regression_loss
        + ranking_weight * ranking_loss
    )
    # target_frame: [B]，把归一化 GT 重新换算为 0...52 的连续帧号，仅用于日志指标。
    target_frame = target * float(logits.shape[1] - 1)
    # expected_frame: [B]；正常由模型直接返回，fallback 与 prediction*52 等价。
    expected_frame = outputs.get(
        "expected_frame", prediction * float(logits.shape[1] - 1)
    ).float()
    # matched_frame: [B] float；正常是 argmax 离散帧号，fallback 从 logits 现场计算。
    matched_frame = outputs.get("matched_frame", logits.argmax(dim=-1)).float()
    return {
        # 以下四项均为可反向/记录的标量 loss。
        "total_loss": total_loss,
        "alignment_loss": alignment_loss,
        "regression_loss": regression_loss,
        "ranking_loss": ranking_loss,
        # 连续 Progress MAE/RMSE：标量，只用于日志，所以 detach。
        "mae": (prediction - target).abs().mean().detach(),
        "rmse": (prediction - target).pow(2).mean().sqrt().detach(),
        # 期望帧号和 argmax 帧号相对连续 GT 帧号的 MAE：标量。
        "expected_frame_mae": (expected_frame - target_frame).abs().mean().detach(),
        "matched_frame_mae": (matched_frame - target_frame).abs().mean().detach(),
        # 每个 query 先判断 argmax 是否落在 GT 帧号 ±1 内，得到 [B] bool，再转 float 求均值。
        "matched_within_one": ((matched_frame - target_frame).abs() <= 1.0)
        .float()
        .mean()
        .detach(),
    }
