import logging
import os
import pickle
import random
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from data.utils.image_utils import get_video_frame_count, load_video_frames, resize_with_padding


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


def _has_lerobot_parquet_layout(directory: Path) -> bool:
    return (directory / "data").exists() and (directory / "meta").exists()


def _find_lerobot_parquet_dirs(root: Path) -> List[Path]:
    """Recursively find LeRobot task directories with data/ and meta/ subdirectories."""
    results: List[Path] = []
    try:
        if _has_lerobot_parquet_layout(root):
            results.append(root)
            return results
        for file in os.listdir(root):
            current = root / file
            if current.is_dir():
                results += _find_lerobot_parquet_dirs(current)
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
        data_format: str = "auto",
        image_column: str = "image",
        image_columns: Optional[List[str]] = None,
        view_layout: str = "single",
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
        self.data_format = data_format
        self.image_column = image_column
        self.image_columns = list(image_columns) if image_columns else [image_column]
        self.view_layout = view_layout
        self.cache_scan = bool(cache_scan)
        self.val = val

        self.episodes = self._scan_all_episodes()
        if self.max_episodes is not None and self.max_episodes > 0:
            self.episodes = self.episodes[: min(self.max_episodes, len(self.episodes))]

        logger.info(
            "VideoBridgeDataset initialized with %s episodes from %s roots; "
            "data_format=%s, image_columns=%s, view_layout=%s, require_language_embedding=%s",
            len(self.episodes),
            len(self.dataset_dir),
            self.data_format,
            self.image_columns,
            self.view_layout,
            self.require_language_embedding,
        )

    def _scan_all_episodes(self) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        image_key = "-".join(self.image_columns)
        cache_suffix = f"{self.data_format}.{image_key}.{self.view_layout}"
        if self.require_language_embedding:
            cache_suffix += ".lang"

        for root in self.dataset_dir:
            cache_file = root / f"cached_video_bridge_episodes.{cache_suffix}.v1.pkl"
            if self.cache_scan and cache_file.exists():
                with open(cache_file, "rb") as file:
                    cached_episodes = pickle.load(file)
                logger.info("Loaded %s cached episodes from %s", len(cached_episodes), cache_file)
                episodes.extend(cached_episodes)
                continue

            cur_episodes = self._scan_root(root)

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

    def _scan_root(self, root: Path) -> List[Dict[str, Any]]:
        cur_episodes: List[Dict[str, Any]] = []

        if self.data_format in ("auto", "video"):
            cur_episodes.extend(self._scan_video_root(root))

        if self.data_format in ("auto", "lerobot_parquet"):
            cur_episodes.extend(self._scan_lerobot_parquet_root(root))

        if not cur_episodes:
            logger.warning("No valid video bridge episodes found under %s", root)

        return cur_episodes

    def _scan_video_root(self, root: Path) -> List[Dict[str, Any]]:
        cur_episodes: List[Dict[str, Any]] = []
        leaf_dirs = _find_leaf_video_dirs(root)
        if not leaf_dirs:
            return cur_episodes

        pbar = tqdm(leaf_dirs)
        for leaf_dir in pbar:
            pbar.set_description(f"Scanning videos {leaf_dir}")
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
                        "format": "video",
                        "video_path": str(video_path),
                        "lang_path": str(lang_path) if lang_path is not None else None,
                        "root": str(leaf_dir),
                        "episode_name": stem,
                    }
                )

        return cur_episodes

    def _scan_lerobot_parquet_root(self, root: Path) -> List[Dict[str, Any]]:
        cur_episodes: List[Dict[str, Any]] = []
        if self.require_language_embedding:
            logger.warning("LeRobot parquet scan ignores roots requiring language embeddings: %s", root)
            return cur_episodes

        leaf_dirs = _find_lerobot_parquet_dirs(root)
        if not leaf_dirs:
            return cur_episodes

        pbar = tqdm(leaf_dirs)
        for leaf_dir in pbar:
            pbar.set_description(f"Scanning parquet {leaf_dir}")
            parquet_files = sorted((leaf_dir / "data").glob("chunk-*/episode_*.parquet"))
            for parquet_path in parquet_files:
                cur_episodes.append(
                    {
                        "format": "lerobot_parquet",
                        "parquet_path": str(parquet_path),
                        "root": str(leaf_dir),
                        "episode_name": f"{leaf_dir.name}/{parquet_path.stem}",
                    }
                )

        return cur_episodes

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

    def _parquet_frame_count(self, parquet_path: str) -> int:
        import pyarrow.parquet as pq

        return pq.ParquetFile(parquet_path).metadata.num_rows

    def _decode_lerobot_image_cell(self, cell: Any, parquet_path: str) -> np.ndarray:
        if isinstance(cell, dict):
            image_bytes = cell.get("bytes")
            image_path = cell.get("path")
            if image_bytes is not None:
                return np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
            if image_path:
                resolved = Path(parquet_path).parent / image_path
                return np.asarray(Image.open(resolved).convert("RGB"))
            raise ValueError(f"Unsupported image dict in {parquet_path}: missing bytes/path")

        if isinstance(cell, (bytes, bytearray)):
            return np.asarray(Image.open(BytesIO(cell)).convert("RGB"))

        if isinstance(cell, np.ndarray):
            return cell

        raise TypeError(f"Unsupported LeRobot image cell type in {parquet_path}: {type(cell)}")

    def _compose_lerobot_views(self, view_frames: List[np.ndarray]) -> np.ndarray:
        if len(view_frames) == 1 or self.view_layout == "single":
            return view_frames[0]

        if self.view_layout not in ("vertical", "horizontal"):
            raise ValueError(f"Unsupported view_layout={self.view_layout!r}; expected single/vertical/horizontal")

        target_h = max(frame.shape[0] for frame in view_frames)
        target_w = max(frame.shape[1] for frame in view_frames)
        aligned = [
            frame if frame.shape[:2] == (target_h, target_w) else resize_with_padding(frame, (target_h, target_w))
            for frame in view_frames
        ]

        axis = 0 if self.view_layout == "vertical" else 1
        return np.concatenate(aligned, axis=axis)

    def _load_lerobot_parquet_frames(self, parquet_path: str, frame_indices: List[int]) -> torch.Tensor:
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=self.image_columns)
        frames = []
        for idx in frame_indices:
            view_frames = [
                self._decode_lerobot_image_cell(df[column].iloc[idx], parquet_path)
                for column in self.image_columns
            ]
            frame_np = self._compose_lerobot_views(view_frames)
            if self.video_size is not None and frame_np.shape[:2] != tuple(self.video_size):
                frame_np = resize_with_padding(frame_np, self.video_size)
            frames.append(frame_np)

        frames_np = np.stack(frames, axis=0)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.episodes:
            return None

        for _ in range(20):
            episode = random.choice(self.episodes)
            try:
                if episode.get("format", "video") == "lerobot_parquet":
                    total_frames = self._parquet_frame_count(episode["parquet_path"])
                else:
                    total_frames = get_video_frame_count(episode["video_path"])

                condition_idx, video_indices = self._select_indices(total_frames)
                frame_indices = [condition_idx] + video_indices

                if episode.get("format", "video") == "lerobot_parquet":
                    frames = self._load_lerobot_parquet_frames(episode["parquet_path"], frame_indices)
                else:
                    frames = load_video_frames(episode["video_path"], frame_indices, self.video_size)

                language_embedding = self._load_language_embedding(episode.get("lang_path"))

                return {
                    "first_frame": frames[0],
                    "video_frames": frames[1:],
                    "language_embedding": language_embedding,
                    "episode_name": episode["episode_name"],
                    "video_path": episode.get("video_path", episode.get("parquet_path")),
                }
            except Exception as exc:
                logger.warning(
                    "Retry due to video bridge sample error (%s): %s",
                    episode.get("episode_name", "unknown"),
                    exc,
                )

        return None
