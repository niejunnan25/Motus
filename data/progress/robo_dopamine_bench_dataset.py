"""Robo-Dopamine-Bench adapter for endpoint-conditioned Progress evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from data.utils.image_utils import resize_with_padding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _normalized_task(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


class RoboDopamineBenchDataset:
    """Expose one Robo-Dopamine subset through the Stage-2 cache dataset API.

    The LIBERO release stores three camera directories, but its left/right wrist
    images are byte-identical.  The VGM was trained on two views, so the default
    mapping is ``cam_high`` followed by ``cam_left_wrist``.
    """

    def __init__(
        self,
        *,
        benchmark_json: str | Path,
        images_root: str | Path,
        language_manifest: str | Path,
        video_size: Tuple[int, int],
        view_video_size: Tuple[int, int] | None = None,
        view_names: Sequence[str] = ("cam_high", "cam_left_wrist"),
        view_layout: str = "vertical",
        require_language_embedding: bool = True,
        max_episodes: int | None = None,
    ) -> None:
        self.benchmark_json = Path(benchmark_json).expanduser().resolve()
        self.images_root = Path(images_root).expanduser().resolve()
        self.language_manifest = Path(language_manifest).expanduser().resolve()
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.view_video_size = (
            (int(view_video_size[0]), int(view_video_size[1]))
            if view_video_size is not None
            else None
        )
        self.view_names = tuple(str(name) for name in view_names)
        self.view_layout = str(view_layout)
        self.require_language_embedding = bool(require_language_embedding)

        if not self.benchmark_json.is_file():
            raise FileNotFoundError(f"Benchmark JSON does not exist: {self.benchmark_json}")
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"Benchmark image root does not exist: {self.images_root}")
        if not self.language_manifest.is_file():
            raise FileNotFoundError(
                f"Task-language manifest does not exist: {self.language_manifest}"
            )
        if not self.view_names:
            raise ValueError("At least one benchmark view is required")
        if self.view_layout not in {"single", "vertical", "horizontal"}:
            raise ValueError(
                f"Unsupported view_layout={self.view_layout!r}; expected single/vertical/horizontal"
            )
        if len(self.view_names) > 1 and self.view_layout == "single":
            raise ValueError("view_layout='single' cannot consume multiple benchmark views")

        task_metadata = self._load_task_metadata()
        benchmark = json.loads(self.benchmark_json.read_text(encoding="utf-8"))
        paths = benchmark.get("sample_path_list")
        tasks = benchmark.get("sample_path_task")
        if not isinstance(paths, list) or not isinstance(tasks, list):
            raise ValueError(
                f"Expected sample_path_list/sample_path_task lists in {self.benchmark_json}"
            )
        if len(paths) != len(tasks):
            raise ValueError(
                f"Benchmark path/task lengths differ: {len(paths)} != {len(tasks)}"
            )
        if max_episodes is not None and int(max_episodes) > 0:
            paths = paths[: int(max_episodes)]
            tasks = tasks[: int(max_episodes)]

        episodes: List[Dict[str, Any]] = []
        for episode_index, (relative_path, task_text) in enumerate(zip(paths, tasks)):
            relative_path = str(relative_path)
            task_text = str(task_text)
            task_key = _normalized_task(task_text)
            if task_key not in task_metadata:
                raise KeyError(
                    f"Benchmark task {task_text!r} has no language embedding in "
                    f"{self.language_manifest}"
                )
            language = task_metadata[task_key]
            episode_root = self.images_root / relative_path
            view_files = [self._list_view_frames(episode_root / name) for name in self.view_names]
            reference_names = [path.name for path in view_files[0]]
            for view_name, files in zip(self.view_names[1:], view_files[1:]):
                names = [path.name for path in files]
                if names != reference_names:
                    raise ValueError(
                        f"Frame names differ between views in {episode_root}: "
                        f"{self.view_names[0]} vs {view_name}"
                    )
            episodes.append(
                {
                    "format": "robo_dopamine_bench",
                    "root": str(self.images_root),
                    "benchmark_json": str(self.benchmark_json),
                    "episode_root": str(episode_root),
                    "episode_name": f"robo_dopamine_bench/{relative_path}",
                    "episode_index": episode_index,
                    "task_index": int(language["task_index"]),
                    "task_text": task_text,
                    "language_caption": language.get("caption_text"),
                    "language_caption_version": language.get("caption_version"),
                    "lang_path": str(language["embedding_path"]),
                    "view_files": view_files,
                    "num_frames": len(reference_names),
                }
            )
        if not episodes:
            raise RuntimeError(f"No episodes found in {self.benchmark_json}")
        self.episodes = episodes

    def _load_task_metadata(self) -> Dict[str, Dict[str, Any]]:
        manifest = json.loads(self.language_manifest.read_text(encoding="utf-8"))
        caption_source = manifest.get("caption_source", {})
        caption_version = (
            caption_source.get("version") if isinstance(caption_source, dict) else None
        )
        metadata: Dict[str, Dict[str, Any]] = {}
        for entry in manifest.get("tasks", []):
            if not isinstance(entry, dict):
                continue
            task_text = entry.get("original_task_text") or entry.get("task_text")
            task_index = entry.get("task_index")
            embedding_path = entry.get("embedding_path")
            if task_text is None or task_index is None:
                continue
            if embedding_path is None:
                embedding_path = self.language_manifest.parent / f"task_{int(task_index):06d}.pt"
            else:
                embedding_path = Path(str(embedding_path))
                if not embedding_path.is_absolute():
                    embedding_path = self.language_manifest.parent / embedding_path
            embedding_path = embedding_path.expanduser().resolve()
            if self.require_language_embedding and not embedding_path.is_file():
                raise FileNotFoundError(
                    f"Missing language embedding for task {task_index}: {embedding_path}"
                )
            key = _normalized_task(str(task_text))
            if key in metadata:
                raise ValueError(f"Duplicate normalized task text in {self.language_manifest}: {task_text}")
            metadata[key] = {
                "task_index": int(task_index),
                "embedding_path": embedding_path,
                "caption_text": entry.get("caption_text"),
                "caption_version": caption_version,
            }
        if not metadata:
            raise RuntimeError(f"No task metadata found in {self.language_manifest}")
        return metadata

    @staticmethod
    def _list_view_frames(view_dir: Path) -> List[Path]:
        if not view_dir.is_dir():
            raise FileNotFoundError(f"Benchmark view directory does not exist: {view_dir}")
        frames = sorted(
            path for path in view_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(frames) < 2:
            raise ValueError(f"Benchmark view needs at least two frames: {view_dir}")
        return frames

    def __len__(self) -> int:
        return len(self.episodes)

    def get_episode_frame_count(self, episode_index: int) -> int:
        return int(self.episodes[int(episode_index)]["num_frames"])

    def get_episode_metadata(self, episode_index: int) -> Dict[str, Any]:
        metadata = dict(self.episodes[int(episode_index)])
        metadata.pop("view_files", None)
        return metadata

    def load_episode_language_embedding(self, episode_index: int) -> torch.Tensor | None:
        episode = self.episodes[int(episode_index)]
        path = Path(str(episode["lang_path"]))
        if not path.is_file():
            if self.require_language_embedding:
                raise FileNotFoundError(f"Missing language embedding: {path}")
            return None
        embedding = _load_torch(path)
        if isinstance(embedding, dict):
            for key in ("embedding", "language_embedding", "text_embedding"):
                if key in embedding:
                    embedding = embedding[key]
                    break
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"Expected tensor language embedding in {path}")
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if embedding.ndim != 2:
            raise ValueError(
                f"Expected language embedding [L,D] in {path}, got {tuple(embedding.shape)}"
            )
        return embedding.float().contiguous()

    def _compose_views(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if len(frames) == 1:
            return frames[0]
        target_h = max(frame.shape[0] for frame in frames)
        target_w = max(frame.shape[1] for frame in frames)
        aligned = [
            frame
            if frame.shape[:2] == (target_h, target_w)
            else resize_with_padding(frame, (target_h, target_w))
            for frame in frames
        ]
        axis = 0 if self.view_layout == "vertical" else 1
        return np.concatenate(aligned, axis=axis)

    def load_episode_frames(
        self, episode_index: int, frame_indices: Sequence[int]
    ) -> torch.Tensor:
        episode = self.episodes[int(episode_index)]
        total_frames = int(episode["num_frames"])
        indices = [int(index) for index in frame_indices]
        if any(index < 0 or index >= total_frames for index in indices):
            raise IndexError(
                f"Frame indices out of bounds for {episode['episode_name']}: "
                f"indices={indices}, total_frames={total_frames}"
            )
        output: List[np.ndarray] = []
        view_files: Sequence[Sequence[Path]] = episode["view_files"]
        for frame_index in indices:
            views = [
                np.array(Image.open(files[frame_index]).convert("RGB"), copy=True)
                for files in view_files
            ]
            frame = self._compose_views(views)
            if frame.shape[:2] != self.video_size:
                frame = resize_with_padding(frame, self.video_size)
            output.append(frame)
        array = np.stack(output, axis=0)
        return torch.from_numpy(array).permute(0, 3, 1, 2).float().div(255.0)

    def load_episode_view_frames(
        self, episode_index: int, frame_indices: Sequence[int]
    ) -> torch.Tensor:
        """Load benchmark images as [F,V,C,H,W] without RGB mosaic fusion."""
        episode = self.episodes[int(episode_index)]
        total_frames = int(episode["num_frames"])
        indices = [int(index) for index in frame_indices]
        if any(index < 0 or index >= total_frames for index in indices):
            raise IndexError(
                f"Frame indices out of bounds for {episode['episode_name']}: "
                f"indices={indices}, total_frames={total_frames}"
            )
        output: List[np.ndarray] = []
        view_files: Sequence[Sequence[Path]] = episode["view_files"]
        for frame_index in indices:
            views = []
            for files in view_files:
                frame = np.array(Image.open(files[frame_index]).convert("RGB"), copy=True)
                if self.view_video_size is not None and frame.shape[:2] != self.view_video_size:
                    frame = resize_with_padding(frame, self.view_video_size)
                views.append(frame)
            output.append(np.stack(views, axis=0))
        array = np.stack(output, axis=0)
        return torch.from_numpy(array).permute(0, 1, 4, 2, 3).float().div(255.0)
