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


def _has_lerobot_v3_video_layout(directory: Path) -> bool:
    return (
        (directory / "data").exists()
        and (directory / "meta" / "episodes").exists()
        and (directory / "videos").exists()
    )


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


def _find_lerobot_v3_video_dirs(root: Path) -> List[Path]:
    """Recursively find LeRobot v3 roots with metadata plus external videos."""
    results: List[Path] = []
    try:
        if _has_lerobot_v3_video_layout(root):
            results.append(root)
            return results
        for file in os.listdir(root):
            current = root / file
            if current.is_dir():
                results += _find_lerobot_v3_video_dirs(current)
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


def _process_state_tensors_batch(states: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Stack optional state tensors, returning None when the batch has no states."""
    if not states or all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("Mixed state/no-state samples in one batch")
    return torch.stack([state.float() for state in states if state is not None], dim=0)


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
        "first_state": _process_state_tensors_batch([sample.get("first_state") for sample in batch]),
        "video_states": _process_state_tensors_batch([sample.get("video_states") for sample in batch]),
        "last_state": _process_state_tensors_batch([sample.get("last_state") for sample in batch]),
        "episode_name": [sample.get("episode_name") for sample in batch],
        "video_path": [sample.get("video_path") for sample in batch],
    }


class VideoBridgeDataset(data.Dataset):
    """
    Pure video dataset for first/last-frame bridge training.

    Supported layouts:
      <leaf>/videos/<episode>.mp4
      <leaf>/umt5_wan/<episode>.pt   optional unless require_language_embedding=True

    LeRobot parquet layout:
      <task>/data/chunk-000/episode_000000.parquet with image bytes columns.

    LeRobot v3 video layout:
      <root>/meta/episodes/chunk-000/file-000.parquet
      <root>/videos/<video_key>/chunk-000/file-000.mp4
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
        task_language_embedding_dir: Optional[str] = None,
        task_language_embedding_pattern: str = "task_{task_index:06d}.pt",
        load_state: bool = False,
        state_column: str = "observation.state",
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
        self.task_language_embedding_dir = task_language_embedding_dir
        self.task_language_embedding_pattern = task_language_embedding_pattern
        self.load_state = bool(load_state)
        self.state_column = str(state_column)
        self.cache_scan = bool(cache_scan)
        self.val = val
        self._state_cache: Dict[str, Tuple[torch.Tensor, Optional[Dict[int, int]]]] = {}

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
        if self.load_state:
            safe_state_column = self.state_column.replace("/", "_").replace(".", "_")
            cache_suffix += f".state.{safe_state_column}"

        for root in self.dataset_dir:
            cache_file = root / f"cached_video_bridge_episodes.{cache_suffix}.v2.pkl"
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

        if self.data_format in ("auto", "lerobot_v3_video"):
            cur_episodes.extend(self._scan_lerobot_v3_video_root(root))

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

    def _scan_lerobot_v3_video_root(self, root: Path) -> List[Dict[str, Any]]:
        cur_episodes: List[Dict[str, Any]] = []
        leaf_dirs = _find_lerobot_v3_video_dirs(root)
        if not leaf_dirs:
            return cur_episodes

        import pandas as pd

        pbar = tqdm(leaf_dirs)
        for leaf_dir in pbar:
            pbar.set_description(f"Scanning LeRobot v3 videos {leaf_dir}")
            load_task_language = self.require_language_embedding or self.task_language_embedding_dir is not None
            task_text_by_index = self._load_lerobot_v3_task_texts(leaf_dir, pd) if load_task_language else {}
            episode_task_by_index = self._load_lerobot_v3_episode_tasks(leaf_dir, pd) if load_task_language else {}
            episode_data_paths_by_index = (
                self._load_lerobot_v3_episode_data_paths(leaf_dir, pd)
                if self.load_state
                else {}
            )
            task_embedding_dir = self._resolve_task_language_embedding_dir(leaf_dir) if load_task_language else None
            episode_files = sorted((leaf_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for episode_file in episode_files:
                episodes_df = pd.read_parquet(episode_file)
                for _, row in episodes_df.iterrows():
                    video_paths = []
                    missing_video = False
                    for video_key in self.image_columns:
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col not in episodes_df.columns or file_col not in episodes_df.columns:
                            raise KeyError(
                                f"Missing LeRobot v3 metadata columns {chunk_col!r}/{file_col!r} in {episode_file}"
                            )

                        chunk_index = int(row[chunk_col])
                        file_index = int(row[file_col])
                        video_path = (
                            leaf_dir
                            / "videos"
                            / video_key
                            / f"chunk-{chunk_index:03d}"
                            / f"file-{file_index:03d}.mp4"
                        )
                        if not video_path.exists():
                            logger.warning("Skipping episode with missing video: %s", video_path)
                            missing_video = True
                            break
                        video_paths.append(str(video_path))

                    if missing_video:
                        continue

                    episode_index = int(row["episode_index"])
                    task_index = episode_task_by_index.get(episode_index)
                    task_text = task_text_by_index.get(task_index) if task_index is not None else None
                    lang_path = (
                        self._task_language_embedding_path(task_embedding_dir, task_index)
                        if task_embedding_dir is not None
                        else None
                    )
                    if self.require_language_embedding and lang_path is None:
                        logger.warning(
                            "Skipping %s/episode_%06d because no task language embedding was found",
                            leaf_dir.name,
                            episode_index,
                        )
                        continue
                    data_paths = episode_data_paths_by_index.get(episode_index, [])
                    if self.load_state and not data_paths:
                        logger.warning(
                            "Skipping %s/episode_%06d because no LeRobot data parquet was found for state loading",
                            leaf_dir.name,
                            episode_index,
                        )
                        continue

                    cur_episodes.append(
                        {
                            "format": "lerobot_v3_video",
                            "video_paths": video_paths,
                            "num_frames": int(row["length"]),
                            "root": str(leaf_dir),
                            "episode_name": f"{leaf_dir.name}/episode_{episode_index:06d}",
                            "episode_index": episode_index,
                            "task_index": task_index,
                            "task_text": task_text,
                            "lang_path": str(lang_path) if lang_path is not None else None,
                            "data_paths": data_paths,
                        }
                    )

        return cur_episodes

    def _load_lerobot_v3_task_texts(self, leaf_dir: Path, pd: Any) -> Dict[int, str]:
        tasks_path = leaf_dir / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            if self.require_language_embedding:
                logger.warning("LeRobot v3 tasks parquet is missing: %s", tasks_path)
            return {}

        tasks_df = pd.read_parquet(tasks_path)
        task_text_by_index: Dict[int, str] = {}
        for row_index, row in tasks_df.iterrows():
            if "task_index" in row:
                task_index = int(row["task_index"])
            else:
                try:
                    task_index = int(row_index)
                except Exception:
                    continue

            task_text = None
            for column in ("task", "text", "instruction", "language_instruction"):
                if column in row and row[column] is not None:
                    task_text = str(row[column])
                    break
            if task_text is None:
                task_text = str(row_index)
            task_text_by_index[task_index] = task_text

        return task_text_by_index

    def _load_lerobot_v3_episode_tasks(self, leaf_dir: Path, pd: Any) -> Dict[int, int]:
        episode_task_by_index: Dict[int, int] = {}
        data_files = sorted((leaf_dir / "data").glob("chunk-*/file-*.parquet"))
        for data_file in data_files:
            try:
                df = pd.read_parquet(data_file, columns=["episode_index", "task_index"])
            except Exception as exc:
                if self.require_language_embedding:
                    logger.warning("Failed reading task_index from %s: %s", data_file, exc)
                continue
            if df.empty:
                continue
            reduced = df.drop_duplicates(subset=["episode_index"], keep="first")
            for _, row in reduced.iterrows():
                episode_task_by_index[int(row["episode_index"])] = int(row["task_index"])
        return episode_task_by_index

    def _load_lerobot_v3_episode_data_paths(self, leaf_dir: Path, pd: Any) -> Dict[int, List[str]]:
        episode_data_paths: Dict[int, List[str]] = {}
        data_files = sorted((leaf_dir / "data").glob("chunk-*/file-*.parquet"))
        for data_file in data_files:
            try:
                df = pd.read_parquet(data_file, columns=["episode_index"])
            except Exception as exc:
                logger.warning("Failed reading episode_index from %s: %s", data_file, exc)
                continue
            if df.empty:
                continue
            for episode_index in sorted({int(value) for value in df["episode_index"].tolist()}):
                episode_data_paths.setdefault(episode_index, []).append(str(data_file))
        return episode_data_paths

    def _resolve_task_language_embedding_dir(self, leaf_dir: Path) -> Path:
        if self.task_language_embedding_dir:
            embedding_dir = Path(self.task_language_embedding_dir)
            if not embedding_dir.is_absolute():
                embedding_dir = leaf_dir / embedding_dir
            return embedding_dir
        return leaf_dir / "umt5_wan_tasks"

    def _task_language_embedding_path(self, embedding_dir: Path, task_index: Optional[int]) -> Optional[Path]:
        if task_index is None:
            return None
        try:
            filename = self.task_language_embedding_pattern.format(task_index=int(task_index))
        except Exception:
            filename = f"task_{int(task_index):06d}.pt"
        candidate = embedding_dir / filename
        if candidate.exists():
            return candidate
        if self.require_language_embedding:
            logger.warning("Missing task language embedding: %s", candidate)
        return None

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

    def _episode_frame_count(self, episode: Dict[str, Any]) -> int:
        if episode.get("format", "video") == "lerobot_parquet":
            return self._parquet_frame_count(episode["parquet_path"])
        if episode.get("format") == "lerobot_v3_video":
            return int(episode["num_frames"])
        return get_video_frame_count(episode["video_path"])

    def _load_episode_frames(self, episode: Dict[str, Any], frame_indices: List[int]) -> torch.Tensor:
        if episode.get("format", "video") == "lerobot_parquet":
            return self._load_lerobot_parquet_frames(episode["parquet_path"], frame_indices)
        if episode.get("format") == "lerobot_v3_video":
            return self._load_lerobot_v3_video_frames(episode, frame_indices)
        return load_video_frames(episode["video_path"], frame_indices, self.video_size)

    def _load_episode_states(self, episode: Dict[str, Any], frame_indices: List[int]) -> Optional[torch.Tensor]:
        if not self.load_state:
            return None
        if episode.get("format") == "lerobot_parquet":
            return self._load_lerobot_parquet_states(episode["parquet_path"], frame_indices)
        if episode.get("format") == "lerobot_v3_video":
            return self._load_lerobot_v3_video_states(episode, frame_indices)
        raise ValueError(f"State loading is not supported for episode format={episode.get('format')!r}")

    @staticmethod
    def _state_cell_to_tensor(cell: Any) -> torch.Tensor:
        if isinstance(cell, torch.Tensor):
            tensor = cell.detach().cpu().float()
        elif isinstance(cell, np.ndarray):
            tensor = torch.from_numpy(np.array(cell, copy=True)).float()
        elif isinstance(cell, (list, tuple)):
            tensor = torch.tensor(cell, dtype=torch.float32)
        elif np.isscalar(cell):
            tensor = torch.tensor([cell], dtype=torch.float32)
        else:
            try:
                tensor = torch.tensor(np.array(cell, copy=True), dtype=torch.float32)
            except Exception as exc:
                raise TypeError(f"Unsupported state cell type: {type(cell)}") from exc
        return tensor.flatten()

    def _load_lerobot_parquet_states(self, parquet_path: str, frame_indices: List[int]) -> torch.Tensor:
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=[self.state_column])
        states = [self._state_cell_to_tensor(df[self.state_column].iloc[idx]) for idx in frame_indices]
        return torch.stack(states, dim=0)

    def _read_lerobot_v3_state_table(
        self,
        episode: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[Dict[int, int]]]:
        import pandas as pd
        import pyarrow.parquet as pq

        data_paths = episode.get("data_paths") or []
        if not data_paths:
            raise ValueError(f"No data_paths available for {episode.get('episode_name')}")

        episode_index = int(episode["episode_index"])
        frames = []
        has_frame_index = False
        for data_path in data_paths:
            schema_names = set(pq.ParquetFile(data_path).schema_arrow.names)
            if self.state_column not in schema_names:
                raise KeyError(f"Missing state column {self.state_column!r} in {data_path}")
            columns = ["episode_index", self.state_column]
            if "frame_index" in schema_names:
                columns.append("frame_index")
                has_frame_index = True
            df = pd.read_parquet(data_path, columns=columns)
            if "episode_index" in df:
                df = df[df["episode_index"] == episode_index]
            if not df.empty:
                frames.append(df)

        if not frames:
            raise ValueError(f"No state rows found for {episode.get('episode_name')}")

        state_df = pd.concat(frames, ignore_index=True)
        frame_index_to_pos = None
        if has_frame_index and "frame_index" in state_df:
            state_df = state_df.sort_values("frame_index").reset_index(drop=True)
            frame_index_to_pos = {
                int(frame_idx): pos
                for pos, frame_idx in enumerate(state_df["frame_index"].tolist())
            }
        else:
            state_df = state_df.reset_index(drop=True)

        state_tensor = torch.stack(
            [self._state_cell_to_tensor(cell) for cell in state_df[self.state_column].tolist()],
            dim=0,
        )
        return state_tensor, frame_index_to_pos

    def _load_lerobot_v3_video_states(self, episode: Dict[str, Any], frame_indices: List[int]) -> torch.Tensor:
        cache_key = f"{episode.get('episode_name')}::{self.state_column}"
        if cache_key not in self._state_cache:
            self._state_cache[cache_key] = self._read_lerobot_v3_state_table(episode)

        states, frame_index_to_pos = self._state_cache[cache_key]
        if frame_index_to_pos is not None:
            missing = [idx for idx in frame_indices if idx not in frame_index_to_pos]
            if missing:
                raise ValueError(f"Missing state frame indices {missing} for {episode.get('episode_name')}")
            positions = [frame_index_to_pos[idx] for idx in frame_indices]
            return states[positions]

        max_index = max(frame_indices)
        if max_index >= states.shape[0]:
            raise ValueError(
                f"State table is too short for {episode.get('episode_name')}: "
                f"need frame {max_index}, got {states.shape[0]} rows"
            )
        return states[frame_indices]

    @staticmethod
    def _add_states_to_sample(sample: Dict[str, Any], states: Optional[torch.Tensor]) -> Dict[str, Any]:
        if states is None:
            return sample
        sample["first_state"] = states[0]
        sample["video_states"] = states[1:]
        sample["last_state"] = states[-1]
        return sample

    def get_bridge_window(self, episode_index: int, condition_idx: Optional[int] = None) -> Dict[str, Any]:
        """Load a deterministic first-last bridge window for evaluation."""
        if not self.episodes:
            raise IndexError("VideoBridgeDataset has no episodes")

        episode = self.episodes[episode_index % len(self.episodes)]
        total_frames = self._episode_frame_count(episode)
        max_cond = total_frames - 1 - self.num_video_frames * self.global_downsample_rate
        if max_cond < 0:
            raise ValueError(
                f"Video is too short: total_frames={total_frames}, "
                f"needs at least {1 + self.num_video_frames * self.global_downsample_rate}"
            )

        if condition_idx is None:
            condition_idx = max_cond // 2
        condition_idx = int(max(0, min(condition_idx, max_cond)))
        video_indices = [
            condition_idx + (i + 1) * self.global_downsample_rate
            for i in range(self.num_video_frames)
        ]
        frame_indices = [condition_idx] + video_indices
        frames = self._load_episode_frames(episode, frame_indices)
        states = self._load_episode_states(episode, frame_indices)
        language_embedding = self._load_language_embedding(episode.get("lang_path"))

        sample = {
            "first_frame": frames[0],
            "video_frames": frames[1:],
            "language_embedding": language_embedding,
            "episode_name": episode["episode_name"],
            "video_path": episode.get("video_path", episode.get("parquet_path", episode.get("video_paths"))),
            "task_index": episode.get("task_index"),
            "task_text": episode.get("task_text"),
            "condition_idx": condition_idx,
            "frame_indices": frame_indices,
            "total_frames": total_frames,
        }
        return self._add_states_to_sample(sample, states)

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

    def _load_video_frames_pyav(self, video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
        import av

        wanted = set(frame_indices)
        max_index = max(wanted)
        decoded: Dict[int, np.ndarray] = {}
        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in wanted:
                    decoded[frame_idx] = frame.to_ndarray(format="rgb24")
                if frame_idx >= max_index:
                    break
        finally:
            container.close()

        missing = [idx for idx in frame_indices if idx not in decoded]
        if missing:
            raise ValueError(f"Failed to decode frames {missing} from {video_path}")
        return [decoded[idx] for idx in frame_indices]

    def _load_lerobot_v3_video_frames(self, episode: Dict[str, Any], frame_indices: List[int]) -> torch.Tensor:
        view_batches = [
            self._load_video_frames_pyav(video_path, frame_indices)
            for video_path in episode["video_paths"]
        ]

        frames = []
        for frame_pos in range(len(frame_indices)):
            frame_np = self._compose_lerobot_views([view_frames[frame_pos] for view_frames in view_batches])
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
                total_frames = self._episode_frame_count(episode)
                condition_idx, video_indices = self._select_indices(total_frames)
                frame_indices = [condition_idx] + video_indices

                frames = self._load_episode_frames(episode, frame_indices)
                states = self._load_episode_states(episode, frame_indices)
                language_embedding = self._load_language_embedding(episode.get("lang_path"))

                sample = {
                    "first_frame": frames[0],
                    "video_frames": frames[1:],
                    "language_embedding": language_embedding,
                    "episode_name": episode["episode_name"],
                    "video_path": episode.get("video_path", episode.get("parquet_path", episode.get("video_paths"))),
                    "task_index": episode.get("task_index"),
                    "task_text": episode.get("task_text"),
                }
                return self._add_states_to_sample(sample, states)
            except Exception as exc:
                logger.warning(
                    "Retry due to video bridge sample error (%s): %s",
                    episode.get("episode_name", "unknown"),
                    exc,
                )

        return None
