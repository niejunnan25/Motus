"""Load one cached trajectory memory with all of its single-frame queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 2


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
        entry = self.entries[index]
        cache_path = self._resolve(str(entry["cache_file"]))
        payload = _load_torch_file(cache_path)
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache payload schema in {cache_path}")

        trajectory_latent = payload["trajectory_latent"]
        trajectory_frame_latents = payload["trajectory_frame_latents"]
        current_latents = payload["current_latents"]
        progress = payload["progress"]
        frame_indices = payload["frame_indices"]
        if trajectory_latent.ndim != 4:
            raise ValueError(f"trajectory_latent must be [C,T,H,W] in {cache_path}")
        if trajectory_frame_latents.ndim != 5 or trajectory_frame_latents.shape[2] != 1:
            raise ValueError(
                f"trajectory_frame_latents must be [F,C,1,H,W] in {cache_path}"
            )
        if current_latents.ndim != 5 or current_latents.shape[2] != 1:
            raise ValueError(f"current_latents must be [N,C,1,H,W] in {cache_path}")
        num_progress_bins = int(
            payload.get("num_progress_bins", trajectory_frame_latents.shape[0])
        )
        if trajectory_frame_latents.shape[0] != num_progress_bins:
            raise ValueError(
                f"Cache declares {num_progress_bins} bins but stores "
                f"{trajectory_frame_latents.shape[0]} frame latents in {cache_path}"
            )
        if (
            self.expected_num_progress_bins is not None
            and num_progress_bins != self.expected_num_progress_bins
        ):
            raise ValueError(
                f"Progress config expects {self.expected_num_progress_bins} bins but cache has "
                f"{num_progress_bins} in {cache_path}"
            )
        if trajectory_frame_latents.shape[1:] != current_latents.shape[1:]:
            raise ValueError(
                "Generated-frame and current-frame latents must share [C,1,H,W], got "
                f"{tuple(trajectory_frame_latents.shape[1:])} and {tuple(current_latents.shape[1:])} "
                f"in {cache_path}"
            )
        if (
            current_latents.shape[0] != progress.shape[0]
            or progress.shape[0] != frame_indices.shape[0]
        ):
            raise ValueError(f"Query tensors have inconsistent lengths in {cache_path}")
        if not torch.isfinite(progress).all() or bool(
            ((progress < 0) | (progress > 1)).any()
        ):
            raise ValueError(
                f"Progress targets must be finite values in [0,1] in {cache_path}"
            )

        sample: Dict[str, Any] = {
            "trajectory_latent": trajectory_latent,
            "trajectory_frame_latents": trajectory_frame_latents,
            "current_latents": current_latents,
            "progress": progress.float(),
            "frame_indices": frame_indices.long(),
            "first_frame": payload["first_frame"],
            "last_frame": payload["last_frame"],
            "episode_name": payload.get("episode_name", entry.get("episode_name")),
            "task_index": payload.get("task_index", entry.get("task_index")),
            "total_frames": int(payload["total_frames"]),
            "num_progress_bins": num_progress_bins,
            "cache_path": str(cache_path),
        }
        if self.load_language_embedding:
            language_file = payload.get("language_file", entry.get("language_file"))
            if language_file is None:
                raise ValueError(
                    f"Layerwise Progress requires a language embedding for {cache_path}"
                )
            sample["language_embedding"] = _load_torch_file(
                self._resolve(str(language_file))
            ).float()
        else:
            sample["language_embedding"] = None
        return sample


def progress_episode_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep variable-length episodes intact; one trajectory is one loader item."""
    if len(batch) != 1:
        raise ValueError(
            "Progress episode batches must use batch_size=1. Query batch size is controlled "
            "inside the trainer so one trajectory memory is reused across many frames."
        )
    return batch[0]
