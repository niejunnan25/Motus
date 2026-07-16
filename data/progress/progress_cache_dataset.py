"""Load one cached trajectory memory with all of its single-frame queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

SCHEMA_VERSION = 2

# 维度约定：F=53 个生成轨迹位置，N=一个 episode 中的真实查询帧数，
# C_z/T_z/H_z/W_z=VAE latent 的通道/时间/空间尺寸。
# Schema v2 同时保存原始 14-slice trajectory_latent 和显式 53-frame memory；
# 旧 Schema v1 只有 14 个 VAE temporal slice，无法表示“查询属于 53 帧中的哪一帧”，
# 因而本 Dataset 会主动拒绝旧缓存，而不是静默插值或降级成 14-bin 任务。


def _load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


class ProgressEpisodeCacheDataset(Dataset):
    """Each item owns one 53-frame trajectory memory and all episode queries."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str = "train",
        manifest_name: str = "manifest.jsonl",
        load_language_embedding: bool = False,
        expected_num_progress_bins: Optional[int] = None,
        require_trajectory_role_latent: bool = False,
        max_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.load_language_embedding = bool(load_language_embedding)
        self.expected_num_progress_bins = (
            int(expected_num_progress_bins)
            if expected_num_progress_bins is not None
            else None
        )
        self.require_trajectory_role_latent = bool(require_trajectory_role_latent)
        manifest_path = self.cache_dir / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Progress cache manifest does not exist: {manifest_path}. "
                "Run scripts/cache_vgm_progress_latents.py first."
            )

        entries: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if int(entry.get("schema_version", -1)) != SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported Progress cache schema at {manifest_path}:{line_number}: "
                        f"{entry.get('schema_version')}"
                    )
                if split != "all" and entry.get("split") != split:
                    continue
                entries.append(entry)

        entries.sort(
            key=lambda item: (
                str(item.get("episode_name")),
                str(item.get("cache_file")),
            )
        )
        if max_episodes is not None and int(max_episodes) > 0:
            entries = entries[: int(max_episodes)]
        if not entries:
            raise RuntimeError(
                f"No Progress cache entries found for split={split!r} in {manifest_path}"
            )
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.cache_dir / path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # index 选择一条 episode manifest 记录；一个 Dataset item 始终对应一整条 episode。
        entry = self.entries[index]
        # cache_path 指向该 episode 的 .pt payload，不是单帧文件。
        cache_path = self._resolve(str(entry["cache_file"]))
        # payload 在 CPU 上加载，避免 DataLoader worker 提前占用 GPU。
        payload = _load_torch_file(cache_path)
        # Schema v2 才包含显式 53-frame memory；v1 的 14-slice cache 在这里直接拒绝。
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache payload schema in {cache_path}")

        # trajectory_latent: [C_z, T_z, H_z, W_z]，正式 53F 缓存中 T_z=14。
        # Serial 不使用它；Layerwise 用它重放一次冻结 WAN，以提取 30 层 hidden state。
        trajectory_latent = payload["trajectory_latent"]
        trajectory_role_latent = payload.get("trajectory_role_latent")
        # trajectory_frame_latents: [F=53, C_z, 1, H_z, W_z]。
        # 两种模型都使用它作为最终 53-bin 匹配的显式轨迹 memory。
        trajectory_frame_latents = payload["trajectory_frame_latents"]
        # current_latents: [N, C_z, 1, H_z, W_z]；progress: [N]。
        # 每个 current_latent 都只来自一张真实在线观测，不含未来帧。
        current_latents = payload["current_latents"]
        progress = payload["progress"]
        # frame_indices: [N]，保存每个 query 在 source episode 中的原始整数帧号。
        frame_indices = payload["frame_indices"]
        # trajectory_latent 去掉 cache 生成时的 batch 维后必须是 rank 4: [C_z,T_z,H_z,W_z]。
        if trajectory_latent.ndim != 4:
            raise ValueError(f"trajectory_latent must be [C,T,H,W] in {cache_path}")
        if trajectory_role_latent is not None:
            if trajectory_role_latent.ndim != 4:
                raise ValueError(
                    f"trajectory_role_latent must be [C,T,H,W] in {cache_path}"
                )
            if trajectory_role_latent.shape != trajectory_latent.shape:
                raise ValueError(
                    "RGB and RoleMask trajectory latents must have identical shapes, got "
                    f"{tuple(trajectory_latent.shape)} and "
                    f"{tuple(trajectory_role_latent.shape)} in {cache_path}"
                )
        if self.require_trajectory_role_latent and trajectory_role_latent is None:
            raise ValueError(
                f"Layerwise joint RoleMask Progress requires trajectory_role_latent in {cache_path}; "
                "regenerate this cache with the updated Stage 1 cache script"
            )
        # 显式帧 memory 必须是 rank 5，且每个 slot 的单帧 temporal latent 维必须 T=1。
        if trajectory_frame_latents.ndim != 5 or trajectory_frame_latents.shape[2] != 1:
            raise ValueError(
                f"trajectory_frame_latents must be [F,C,1,H,W] in {cache_path}"
            )
        # 在线 query 同样必须是独立单帧 VAE latent: [N,C_z,1,H_z,W_z]。
        if current_latents.ndim != 5 or current_latents.shape[2] != 1:
            raise ValueError(f"current_latents must be [N,C,1,H,W] in {cache_path}")
        # num_progress_bins 是标量；旧 payload 若缺字段，才从 memory 第二维 F 回退推断。
        num_progress_bins = int(
            payload.get("num_progress_bins", trajectory_frame_latents.shape[0])
        )
        # payload 声明的 F 必须等于实际 trajectory_frame_latents.shape[0]。
        if trajectory_frame_latents.shape[0] != num_progress_bins:
            raise ValueError(
                f"Cache declares {num_progress_bins} bins but stores "
                f"{trajectory_frame_latents.shape[0]} frame latents in {cache_path}"
            )
        # 正式模型配置要求 53 bins；这里阻止误把其他 F 的 cache 交给 53-class head。
        if (
            self.expected_num_progress_bins is not None
            and num_progress_bins != self.expected_num_progress_bins
        ):
            raise ValueError(
                f"Progress config expects {self.expected_num_progress_bins} bins but cache has "
                f"{num_progress_bins} in {cache_path}"
            )
        # 去掉各自的帧数维 F/N 后，两边必须共享完全相同的 [C_z,1,H_z,W_z]；
        # 这是 SharedFrameLatentEncoder 可以同权重处理两侧的前提。
        if trajectory_frame_latents.shape[1:] != current_latents.shape[1:]:
            raise ValueError(
                "Generated-frame and current-frame latents must share [C,1,H,W], got "
                f"{tuple(trajectory_frame_latents.shape[1:])} and {tuple(current_latents.shape[1:])} "
                f"in {cache_path}"
            )
        if progress.ndim != 1 or frame_indices.ndim != 1:
            raise ValueError(
                f"progress and frame_indices must both be [N] in {cache_path}"
            )
        # 三个 query 级 tensor 的首维必须都是 N，保证 current_latents[i]、progress[i]、
        # frame_indices[i] 描述同一张 source frame。
        if (
            current_latents.shape[0] != progress.shape[0]
            or progress.shape[0] != frame_indices.shape[0]
        ):
            raise ValueError(f"Query tensors have inconsistent lengths in {cache_path}")
        if current_latents.shape[0] < 1:
            raise ValueError(f"Progress cache contains no query frames in {cache_path}")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if frame_indices.dtype not in integer_dtypes:
            raise ValueError(
                f"frame_indices must be an integer [N] tensor in {cache_path}"
            )
        total_frames = int(payload["total_frames"])
        if bool((frame_indices < 0).any()) or bool(
            (frame_indices >= total_frames).any()
        ):
            raise ValueError(
                f"frame_indices must lie in [0,{total_frames - 1}] in {cache_path}"
            )
        if frame_indices.numel() > 1 and not bool(
            (frame_indices[1:] > frame_indices[:-1]).all()
        ):
            raise ValueError(
                f"frame_indices must be strictly increasing in {cache_path}"
            )
        # progress: [N] 中不允许 NaN/Inf，也不允许超出归一化区间 [0,1]。
        if not torch.isfinite(progress).all() or bool(
            ((progress < 0) | (progress > 1)).any()
        ):
            raise ValueError(
                f"Progress targets must be finite values in [0,1] in {cache_path}"
            )
        if progress.numel() > 1 and not bool((progress[1:] > progress[:-1]).all()):
            raise ValueError(
                f"Progress targets must be strictly increasing in {cache_path}"
            )

        sample: Dict[str, Any] = {
            # Dataset item 保留 episode 维度结构，不在这里添加 batch 维。
            # [C_z,T_z=14,H_z,W_z]。
            "trajectory_latent": trajectory_latent,
            # Optional [C_z,T_z,H_z,W_z] generated RoleMask latent for joint-VGM
            # Layerwise replay. Serial Progress intentionally ignores this field.
            "trajectory_role_latent": trajectory_role_latent,
            # [F=53,C_z,1,H_z,W_z]。
            "trajectory_frame_latents": trajectory_frame_latents,
            # [N,C_z,1,H_z,W_z]。
            "current_latents": current_latents,
            # [N] float32。
            "progress": progress.float(),
            # [N] int64。
            "frame_indices": frame_indices.long(),
            # 各为 [C_img,H_img,W_img] uint8；Trainer 再恢复 batch 维。
            "first_frame": payload["first_frame"],
            "last_frame": payload["last_frame"],
            "episode_name": payload.get("episode_name", entry.get("episode_name")),
            "task_index": payload.get("task_index", entry.get("task_index")),
            "total_frames": total_frames,
            "num_progress_bins": num_progress_bins,
            "cache_path": str(cache_path),
        }
        if self.load_language_embedding:
            # 只有 Layerwise 分支需要重放冻结 WAN，因而才加载语言 embedding。
            language_file = payload.get("language_file", entry.get("language_file"))
            if language_file is None:
                raise ValueError(
                    f"Layerwise Progress requires a language embedding for {cache_path}"
                )
            # 加载后 language_embedding: [L_text,D_text] float32；Trainer 再增加 batch 维。
            sample["language_embedding"] = _load_torch_file(
                self._resolve(str(language_file))
            ).float()
        else:
            # Serial 只读取缓存的单帧 latent，不运行 WAN，所以语言字段保持 None。
            sample["language_embedding"] = None
        # 返回一条完整 episode；不会在这里随机抽一张 current query。
        return sample


def progress_episode_collate_fn(
    batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep E variable-length episodes intact for episode-query mixed batching."""
    if not batch:
        raise ValueError("Progress episode batch must contain at least one episode")
    # 返回长度 E 的 episode list，不尝试 stack 具有不同 N 的 current_latents/progress。
    # 每个 item 内部仍保持：trajectory_frame_latents [F,C_z,1,H_z,W_z]、
    # current_latents [N_e,C_z,1,H_z,W_z]、progress [N_e]。Trainer 再从每条
    # episode 独立采样 Q 个 query，并建立 query -> episode 的显式索引。
    return batch
