#!/usr/bin/env python3
"""Convert the official PushT replay zarr into traceable VGM video splits.

The conversion intentionally keeps the original 96x96 render. Motus resizes
frames at load time, while the accompanying per-episode NPZ files retain the
original low-dimensional state needed for path-diversity diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import zarr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_zarr", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_start_clusters", type=int, default=8)
    parser.add_argument("--holdout_cluster", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def start_descriptors(states: np.ndarray, episode_starts: np.ndarray) -> np.ndarray:
    first = states[episode_starts]
    return np.column_stack(
        [
            first[:, :4],
            np.sin(first[:, 4]),
            np.cos(first[:, 4]),
        ]
    )


def write_video(path: Path, frames: np.ndarray, fps: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
    writer = imageio.get_writer(
        tmp_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "16", "-movflags", "+faststart"],
        macro_block_size=16,
    )
    try:
        for frame in frames:
            writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.close()
    os.replace(tmp_path, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    source = args.source_zarr.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if args.fps < 1:
        raise ValueError("fps must be positive")
    if not 0 <= args.holdout_cluster < args.num_start_clusters:
        raise ValueError("holdout_cluster must be in [0, num_start_clusters)")

    root = zarr.open(str(source), mode="r")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate([np.zeros(1, dtype=np.int64), episode_ends[:-1]])
    states = np.asarray(root["data/state"][:], dtype=np.float32)
    descriptors = start_descriptors(states, episode_starts)
    scaled = StandardScaler().fit_transform(descriptors)
    clusterer = KMeans(
        n_clusters=args.num_start_clusters,
        random_state=args.seed,
        n_init=20,
    )
    cluster_ids = clusterer.fit_predict(scaled)

    output.mkdir(parents=True, exist_ok=True)
    metadata_dir = output / "metadata" / "episodes"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}

    arrays = {
        "state": root["data/state"],
        "action": root["data/action"],
        "keypoint": root["data/keypoint"],
        "n_contacts": root["data/n_contacts"],
    }
    images = root["data/img"]
    for episode_index, (begin, end) in enumerate(zip(episode_starts, episode_ends)):
        begin_i = int(begin)
        end_i = int(end)
        split = "test" if int(cluster_ids[episode_index]) == args.holdout_cluster else "train"
        episode_name = f"episode_{episode_index:06d}"
        video_path = output / split / "videos" / f"{episode_name}.mp4"
        metadata_path = metadata_dir / f"{episode_name}.npz"

        write_video(video_path, images[begin_i:end_i], args.fps, args.overwrite)
        if args.overwrite or not metadata_path.exists():
            np.savez_compressed(
                metadata_path,
                episode_index=np.int64(episode_index),
                original_frame_indices=np.arange(begin_i, end_i, dtype=np.int64),
                start_descriptor=descriptors[episode_index].astype(np.float32),
                start_cluster=np.int64(cluster_ids[episode_index]),
                **{
                    name: np.asarray(array[begin_i:end_i], dtype=np.float32)
                    for name, array in arrays.items()
                },
            )

        row = {
            "episode_index": episode_index,
            "episode_name": episode_name,
            "split": split,
            "start_cluster": int(cluster_ids[episode_index]),
            "num_frames": end_i - begin_i,
            "video_path": str(video_path.relative_to(output)),
            "metadata_path": str(metadata_path.relative_to(output)),
            "first_state": states[begin_i].tolist(),
            "last_state": states[end_i - 1].tolist(),
        }
        rows_by_split[split].append(row)

    for split, rows in rows_by_split.items():
        write_jsonl(output / split / "manifest.jsonl", rows)

    video_files = sorted(output.glob("*/videos/*.mp4"))
    summary = {
        "source_zarr": str(source),
        "source_zip_sha256": None,
        "num_episodes": int(len(episode_ends)),
        "num_frames": int(episode_ends[-1]),
        "fps": args.fps,
        "num_start_clusters": args.num_start_clusters,
        "holdout_cluster": args.holdout_cluster,
        "seed": args.seed,
        "split_counts": {key: len(value) for key, value in rows_by_split.items()},
        "cluster_counts": {
            str(index): int(np.sum(cluster_ids == index))
            for index in range(args.num_start_clusters)
        },
        "video_count": len(video_files),
        "manifest_sha256": {
            split: sha256(output / split / "manifest.jsonl") for split in rows_by_split
        },
    }
    source_zip = source.parent.parent / "pusht.zip"
    if source_zip.is_file():
        summary["source_zip_sha256"] = sha256(source_zip)
    with (output / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
