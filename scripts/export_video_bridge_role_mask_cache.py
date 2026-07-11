#!/usr/bin/env python3
"""Pre-export immutable LeRobot role-mask Parquet data into shared episode caches."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Any role-mask VGM config using the target dataset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_dataset(config: Any) -> VideoBridgeDataset:
    return VideoBridgeDataset(
        dataset_dir=[str(path) for path in config.dataset.dataset_dir],
        global_downsample_rate=config.common.global_downsample_rate,
        num_video_frames=config.common.num_video_frames,
        video_size=(
            config.dataset.get("video_height", config.common.video_height),
            config.dataset.get("video_width", config.common.video_width),
        ),
        max_episodes=config.dataset.get("max_episodes", None),
        require_language_embedding=config.dataset.get("require_language_embedding", False),
        video_extensions=list(config.dataset.get("video_extensions", [".mp4"])),
        data_format=config.dataset.get("data_format", "auto"),
        image_column=config.dataset.get("image_column", "image"),
        image_columns=config.dataset.get("image_columns", None),
        view_layout=config.dataset.get("view_layout", "single"),
        task_language_embedding_dir=config.dataset.get("task_language_embedding_dir", None),
        task_language_embedding_pattern=config.dataset.get(
            "task_language_embedding_pattern", "task_{task_index:06d}.pt"
        ),
        task_language_caption_version=config.dataset.get("task_language_caption_version", None),
        load_state=False,
        bridge_sampling_mode=config.dataset.get("bridge_sampling_mode", "sliding_window"),
        bridge_sampling_jitter=config.dataset.get("bridge_sampling_jitter", False),
        load_role_mask=True,
        role_mask_columns=config.dataset.get("role_mask_columns", None),
        role_mask_render_mode=config.dataset.get("role_mask_render_mode", "binary"),
        role_mask_foreground_ids=config.dataset.get("role_mask_foreground_ids", [1, 2, 4]),
        role_mask_palette=config.dataset.get("role_mask_palette", None),
        role_mask_cache_dir=config.dataset.get("role_mask_cache_dir", None),
        role_mask_memory_cache_size=0,
        strict_role_mask=True,
        cache_scan=config.dataset.get("cache_scan", False),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = OmegaConf.load(args.config)
    if args.max_episodes is not None:
        config.dataset.max_episodes = args.max_episodes
    dataset = build_dataset(config)
    if dataset.role_mask_cache_dir is None:
        raise ValueError("dataset.role_mask_cache_dir must be configured")

    def export_episode(episode: dict[str, Any]) -> str:
        cache_path = dataset._role_mask_cache_path(episode)
        if cache_path is None:
            raise RuntimeError("Role-mask cache path unexpectedly resolved to None")
        if cache_path.exists() and not args.overwrite:
            source_signature = dataset._role_mask_source_signature(episode)
            if dataset._read_role_mask_cache(cache_path, source_signature) is not None:
                return "cached"
        if args.overwrite and cache_path.exists():
            cache_path.unlink()
        dataset._read_lerobot_v3_role_mask_table(episode)
        return "written"

    counts = {"cached": 0, "written": 0}
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(export_episode, episode): episode for episode in dataset.episodes}
        for completed, future in enumerate(as_completed(futures), start=1):
            episode = futures[future]
            try:
                counts[future.result()] += 1
            except Exception:
                logger.exception("Failed exporting %s", episode.get("episode_name"))
                raise
            if completed % 100 == 0 or completed == len(futures):
                logger.info("Exported %s/%s episodes: %s", completed, len(futures), counts)

    logger.info("Role-mask cache ready at %s: %s", dataset.role_mask_cache_dir, counts)


if __name__ == "__main__":
    main()
