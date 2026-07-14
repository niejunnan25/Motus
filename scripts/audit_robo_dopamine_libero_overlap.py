#!/usr/bin/env python3
"""Estimate likely trajectory overlap between LeRobot and Robo-Dopamine LIBERO."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import av
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_VIEWS = ("observation.images.image", "observation.images.image2")
BENCH_VIEWS = ("cam_high", "cam_left_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--benchmark_json", type=Path, required=True)
    parser.add_argument("--benchmark_images", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--match_mse_threshold",
        type=float,
        default=100.0,
        help="Maximum sampled-frame RGB MSE for a likely transcoded match.",
    )
    return parser.parse_args()


def normalized_task(text: str) -> str:
    return " ".join(str(text).split()).casefold()


@lru_cache(maxsize=4096)
def decode_video_frame(path_text: str, frame_index: int) -> np.ndarray:
    path = Path(path_text)
    container = av.open(str(path))
    selected = None
    try:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                selected = frame.to_ndarray(format="rgb24")
                break
    finally:
        container.close()
    if selected is None:
        raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
    return selected


def rgb_mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Image shapes differ: {left.shape} != {right.shape}")
    delta = left.astype(np.float32) - right.astype(np.float32)
    return float(np.mean(np.square(delta)))


def video_path(source_root: Path, row: pd.Series, view: str) -> Path:
    chunk = int(row[f"videos/{view}/chunk_index"])
    file_index = int(row[f"videos/{view}/file_index"])
    return source_root / "videos" / view / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def load_source_tables(source_root: Path) -> tuple[pd.DataFrame, Dict[str, int]]:
    episode_files = sorted((source_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    data_files = sorted((source_root / "data").glob("chunk-*/file-*.parquet"))
    if not episode_files or not data_files:
        raise FileNotFoundError(f"Incomplete LeRobot v3 dataset under {source_root}")
    episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    episode_tasks = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in data_files],
        ignore_index=True,
    ).drop_duplicates("episode_index")
    episodes = episodes.merge(episode_tasks, on="episode_index", how="left", validate="one_to_one")
    tasks = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    task_to_index = {
        normalized_task(str(task_text)): int(row["task_index"])
        for task_text, row in tasks.iterrows()
    }
    return episodes, task_to_index


def benchmark_frame(
    images_root: Path, relative_path: str, view: str, frame_index: int
) -> np.ndarray:
    path = images_root / relative_path / view / f"frame_{frame_index:06d}.jpg"
    if not path.is_file():
        matches = sorted((images_root / relative_path / view).glob(f"frame_{frame_index:06d}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing benchmark frame {frame_index}: {path.parent}")
        path = matches[0]
    return np.array(Image.open(path).convert("RGB"), copy=True)


def endpoint_mse(
    source_root: Path,
    source_row: pd.Series,
    benchmark_images: Path,
    benchmark_path: str,
    frame_index: int,
) -> float:
    values = []
    for source_view, bench_view in zip(SOURCE_VIEWS, BENCH_VIEWS):
        source = decode_video_frame(str(video_path(source_root, source_row, source_view)), frame_index)
        benchmark = benchmark_frame(benchmark_images, benchmark_path, bench_view, frame_index)
        values.append(rgb_mse(source, benchmark))
    return float(np.mean(values))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    benchmark_json = args.benchmark_json.expanduser().resolve()
    benchmark_images = args.benchmark_images.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_episodes, task_to_index = load_source_tables(source_root)
    benchmark = json.loads(benchmark_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for benchmark_index, (relative_path, task_text) in enumerate(
        zip(benchmark["sample_path_list"], benchmark["sample_path_task"])
    ):
        relative_path = str(relative_path)
        task_text = str(task_text)
        task_index = task_to_index.get(normalized_task(task_text))
        if task_index is None:
            raise KeyError(f"Unknown benchmark task: {task_text!r}")
        frame_count = sum(
            path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for path in (benchmark_images / relative_path / BENCH_VIEWS[0]).iterdir()
        )
        candidates = source_episodes[
            (source_episodes["length"] == frame_count)
            & (source_episodes["task_index"] == task_index)
        ]
        scored = []
        for _, candidate in candidates.iterrows():
            first_mse = endpoint_mse(
                source_root,
                candidate,
                benchmark_images,
                relative_path,
                0,
            )
            scored.append(
                {
                    "candidate": candidate,
                    "first_mse": first_mse,
                    "middle_mse": None,
                    "last_mse": None,
                    "trajectory_mse": None,
                }
            )
        scored.sort(key=lambda item: float(item["first_mse"]))
        second_first = float(scored[1]["first_mse"]) if len(scored) > 1 else None
        for item in scored:
            if float(item["first_mse"]) > args.match_mse_threshold:
                continue
            item["middle_mse"] = endpoint_mse(
                source_root,
                item["candidate"],
                benchmark_images,
                relative_path,
                frame_count // 2,
            )
            item["last_mse"] = endpoint_mse(
                source_root,
                item["candidate"],
                benchmark_images,
                relative_path,
                frame_count - 1,
            )
            item["trajectory_mse"] = float(
                np.mean([item["first_mse"], item["middle_mse"], item["last_mse"]])
            )
        trajectory_candidates = [
            item for item in scored if item["trajectory_mse"] is not None
        ]
        if trajectory_candidates:
            best = min(
                trajectory_candidates,
                key=lambda item: float(item["trajectory_mse"]),
            )
        else:
            best = scored[0] if scored else None
        best_row = best["candidate"] if best is not None else None
        best_first = float(best["first_mse"]) if best is not None else float("inf")
        best_middle = best["middle_mse"] if best is not None else None
        best_last = best["last_mse"] if best is not None else None
        best_trajectory = best["trajectory_mse"] if best is not None else None
        matched = False
        if best is not None and best_trajectory is not None:
            matched = all(
                float(best[key]) <= args.match_mse_threshold
                for key in ("first_mse", "middle_mse", "last_mse")
            )
            best_last = best["last_mse"]
        rows.append(
            {
                "benchmark_index": benchmark_index,
                "benchmark_path": relative_path,
                "task_index": task_index,
                "task_text": task_text,
                "frames": frame_count,
                "candidate_count": len(candidates),
                "best_source_episode": (
                    int(best_row["episode_index"]) if best_row is not None else None
                ),
                "first_frame_mse": best_first if np.isfinite(best_first) else None,
                "middle_frame_mse": best_middle,
                "last_frame_mse": best_last,
                "three_point_trajectory_mse": best_trajectory,
                "second_best_first_frame_mse": second_first,
                "likely_same_initial_state": best_first <= args.match_mse_threshold,
                "likely_same_trajectory": matched,
            }
        )
        print(
            f"{benchmark_index:03d} candidates={len(candidates):2d} "
            f"first_mse={best_first:9.3f} middle_mse={best_middle} "
            f"last_mse={best_last} match={matched}",
            flush=True,
        )

    matched_rows = [row for row in rows if row["likely_same_trajectory"]]
    initial_state_matches = [
        row for row in rows if row["likely_same_initial_state"]
    ]
    summary = {
        "source_root": str(source_root),
        "benchmark_json": str(benchmark_json),
        "benchmark_images": str(benchmark_images),
        "match_mse_threshold": args.match_mse_threshold,
        "episodes": len(rows),
        "likely_same_initial_state_matches": len(initial_state_matches),
        "likely_same_initial_state_rate": len(initial_state_matches) / len(rows),
        "likely_same_trajectory_matches": len(matched_rows),
        "likely_same_trajectory_rate": len(matched_rows) / len(rows),
        "episodes_without_task_length_candidate": sum(
            row["candidate_count"] == 0 for row in rows
        ),
    }
    write_csv(args.output_dir / "episode_overlap.csv", rows)
    (args.output_dir / "episode_overlap.json").write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf-8"
    )
    first_mse = [float(row["first_frame_mse"]) for row in rows if row["first_frame_mse"] is not None]
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.hist(first_mse, bins=40, color="#2563eb", alpha=0.8)
    axis.axvline(args.match_mse_threshold, color="#dc2626", linestyle="--", linewidth=2)
    axis.set_xlabel("Best candidate first-frame RGB MSE (two views)")
    axis.set_ylabel("Benchmark episodes")
    axis.set_title("Robo-Dopamine LIBERO likely-overlap audit")
    axis.set_yscale("log")
    figure.savefig(args.output_dir / "overlap_mse_histogram.png", dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
