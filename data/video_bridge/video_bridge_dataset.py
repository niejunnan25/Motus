import logging
import os
import pickle
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
from tqdm import tqdm

from data.utils.image_utils import get_video_frame_count, load_video_frames


logger = logging.getLogger(__name__)


def _has_video_subdir(directory: Path) -> bool:
    return (directory / "videos").exists()


def _find_leaf_video_dirs(root: Path) -> List[Path]:
    """Recursively find directories that contain a videos/ subdirectory."""
    results: List[Path] = []
    try:
        if _has_video_subdir(root):
            results.append(root)
            return results
        for file in os.listdir(root):
            current = root / file
            if current.is_dir():
                results += _find_leaf_video_dirs(current)
    except Exception as exc:
        logger.warning("Failed scanning %s: %s", root, exc)
    return results


def _process_language_embeddings_batch(
    language_embeddings: List[Optional[torch.Tensor]],
    text_len: int = 512,
) -> Optional[torch.Tensor]:
    """Pad language embeddings to [B, text_len, dim], filling missing items with zeros."""
    template = next((emb for emb in language_embeddings if emb is not None), None)
    if template is None:
        return None

    if template.dim() == 3:
        template = template.squeeze(0)
    text_dim = template.shape[1]
    padded_embeddings = []

    for emb in language_embeddings:
        if emb is None:
            padded_embeddings.append(template.new_zeros(text_len, text_dim))
            continue

        if emb.dim() == 3:
            emb = emb.squeeze(0)
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)

    return torch.stack(padded_embeddings, dim=0)


def video_bridge_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Collate pure video bridge samples."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None

    return {
        "first_frame": torch.stack([sample["first_frame"] for sample in batch]),
        "video_frames": torch.stack([sample["video_frames"] for sample in batch]),
        "language_embedding": _process_language_embeddings_batch(
            [sample.get("language_embedding") for sample in batch]
        ),
        "episode_name": [sample.get("episode_name") for sample in batch],
        "video_path": [sample.get("video_path") for sample in batch],
    }


class VideoBridgeDataset(data.Dataset):
    """
    Pure video dataset for first/last-frame bridge training.

    Expected leaf layout:
      <leaf>/videos/<episode>.mp4
      <leaf>/umt5_wan/<episode>.pt   optional unless require_language_embedding=True
    """

    def __init__(
        self,
        dataset_dir: List[str],
        *,
        global_downsample_rate: int = 3,
        num_video_frames: int = 16,
        video_size: Tuple[int, int] = (384, 320),
        max_episodes: Optional[int] = None,
        require_language_embedding: bool = False,
        video_extensions: Optional[List[str]] = None,
        cache_scan: bool = True,
        val: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not dataset_dir:
            raise ValueError("dataset.dataset_dir must contain at least one video dataset root")

        self.dataset_dir = [Path(path) for path in dataset_dir]
        self.global_downsample_rate = int(global_downsample_rate)
        self.num_video_frames = int(num_video_frames)
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.require_language_embedding = bool(require_language_embedding)
        self.video_extensions = tuple(video_extensions or [".mp4"])
        self.cache_scan = bool(cache_scan)
        self.val = val

        self.episodes = self._scan_all_episodes()
        if self.max_episodes is not None and self.max_episodes > 0:
            self.episodes = self.episodes[: min(self.max_episodes, len(self.episodes))]

        logger.info(
            "VideoBridgeDataset initialized with %s episodes from %s roots; require_language_embedding=%s",
            len(self.episodes),
            len(self.dataset_dir),
            self.require_language_embedding,
        )

    def _scan_all_episodes(self) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        cache_suffix = "lang" if self.require_language_embedding else "video"

        for root in self.dataset_dir:
            cache_file = root / f"cached_video_bridge_episodes.{cache_suffix}.v1.pkl"
            if self.cache_scan and cache_file.exists():
                with open(cache_file, "rb") as file:
                    cached_episodes = pickle.load(file)
                logger.info("Loaded %s cached episodes from %s", len(cached_episodes), cache_file)
                episodes.extend(cached_episodes)
                continue

            cur_episodes: List[Dict[str, Any]] = []
            leaf_dirs = _find_leaf_video_dirs(root)
            if not leaf_dirs:
                logger.warning("No valid video leaf dataset dirs under %s", root)
                continue

            pbar = tqdm(leaf_dirs)
            for leaf_dir in pbar:
                pbar.set_description(f"Scanning {leaf_dir}")
                videos_dir = leaf_dir / "videos"
                umt5_dir = leaf_dir / "umt5_wan"
                has_umt5 = umt5_dir.exists()

                if self.require_language_embedding and not has_umt5:
                    logger.warning("Skipping %s because require_language_embedding=True but umt5_wan/ is missing", leaf_dir)
                    continue

                lang_stems = set()
                if has_umt5:
                    lang_stems = {Path(file).stem for file in os.listdir(umt5_dir) if file.endswith(".pt")}

                for video_file in sorted(os.listdir(videos_dir)):
                    video_path = videos_dir / video_file
                    if video_path.suffix not in self.video_extensions:
                        continue

                    stem = video_path.stem
                    lang_path = umt5_dir / f"{stem}.pt" if stem in lang_stems else None
                    if self.require_language_embedding and lang_path is None:
                        continue

                    cur_episodes.append(
                        {
                            "video_path": str(video_path),
                            "lang_path": str(lang_path) if lang_path is not None else None,
                            "root": str(leaf_dir),
                            "episode_name": stem,
                        }
                    )

            if self.cache_scan:
                try:
                    tmp_path = root / f"cached_video_bridge_episodes.{uuid.uuid4().hex}.pkl"
                    with open(tmp_path, "wb") as file:
                        pickle.dump(cur_episodes, file)
                    os.replace(tmp_path, cache_file)
                    logger.info("Cached %s episodes to %s", len(cur_episodes), cache_file)
                except Exception as exc:
                    logger.warning("Failed to cache episodes for %s: %s", root, exc)

            episodes.extend(cur_episodes)

        return episodes

    def __len__(self) -> int:
        return len(self.episodes) * 100

    def _select_indices(self, total_frames: int) -> Tuple[int, List[int]]:
        step = self.global_downsample_rate
        max_cond = total_frames - 1 - self.num_video_frames * step
        if max_cond < 0:
            raise ValueError(
                f"Video is too short: total_frames={total_frames}, "
                f"needs at least {1 + self.num_video_frames * step}"
            )

        condition_idx = random.randint(0, max_cond)
        video_indices = [condition_idx + (i + 1) * step for i in range(self.num_video_frames)]
        return condition_idx, video_indices

    def _load_language_embedding(self, lang_path: Optional[str]) -> Optional[torch.Tensor]:
        if lang_path is None:
            return None

        embedding_data = torch.load(lang_path, map_location="cpu")
        if isinstance(embedding_data, list):
            embeddings = random.choice(embedding_data)
        else:
            embeddings = embedding_data

        if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Language embedding must be a Tensor or list of Tensors: {lang_path}")
        return embeddings

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.episodes:
            return None

        for _ in range(20):
            episode = random.choice(self.episodes)
            try:
                total_frames = get_video_frame_count(episode["video_path"])
                condition_idx, video_indices = self._select_indices(total_frames)
                frames = load_video_frames(
                    episode["video_path"],
                    [condition_idx] + video_indices,
                    self.video_size,
                )
                language_embedding = self._load_language_embedding(episode.get("lang_path"))

                return {
                    "first_frame": frames[0],
                    "video_frames": frames[1:],
                    "language_embedding": language_embedding,
                    "episode_name": episode["episode_name"],
                    "video_path": episode["video_path"],
                }
            except Exception as exc:
                logger.warning(
                    "Retry due to video bridge sample error (%s): %s",
                    episode.get("episode_name", "unknown"),
                    exc,
                )

        return None
