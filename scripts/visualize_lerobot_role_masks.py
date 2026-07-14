#!/usr/bin/env python3
"""Visualize RGB, binary role masks, and semantic role masks for LeRobot episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from PIL import Image, ImageDraw, ImageFont


MASK_COLUMNS = (
    "observation.segmentation.image.role_mask",
    "observation.segmentation.image2.role_mask",
)
ROLE_NAMES = {
    0: "background",
    1: "active",
    2: "target",
    3: "distractor",
    4: "robot",
    5: "fixture",
    6: "other",
}
ROLE_COLORS = np.asarray(
    [
        [0, 0, 0],
        [255, 48, 48],
        [45, 220, 90],
        [40, 135, 255],
        [255, 220, 35],
        [220, 70, 230],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--task_indices", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def read_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def decode_video(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    container = av.open(str(path))
    try:
        frames.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
    finally:
        container.close()
    return frames


def nested_mask_array(table: Any, column_name: str) -> np.ndarray:
    column = table[column_name].combine_chunks()
    height = len(column[0])
    width = len(column[0].values)
    values = column.values.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(column), height, width).astype(np.uint8, copy=False)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    foreground = np.isin(mask, np.asarray([1, 2, 4], dtype=np.uint8))
    return np.repeat((foreground.astype(np.uint8) * 255)[..., None], 3, axis=-1)


def semantic_mask(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask, 0, len(ROLE_COLORS) - 1)
    return ROLE_COLORS[clipped]


def draw_cell_label(canvas: Image.Image, x: int, y: int, text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    font = load_font(16, bold=True)
    draw.rectangle((x, y, x + 255, y + 25), fill=(0, 0, 0))
    draw.text((x + 7, y + 4), text, font=font, fill=(255, 255, 255))


def make_panel(
    main_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    main_role: np.ndarray,
    wrist_role: np.ndarray,
    title: str,
    frame_idx: int,
) -> np.ndarray:
    header_height = 70
    canvas = Image.new("RGB", (768, header_height + 512), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), f"{title} | frame={frame_idx}", font=load_font(20, bold=True), fill=(20, 24, 31))
    legend = "red=active  green=target  blue=distractor  yellow=robot  magenta=fixture  white=other"
    draw.text((10, 38), legend, font=load_font(14), fill=(20, 24, 31))

    cells = [
        (main_rgb, 0, header_height, "main RGB"),
        (binary_mask(main_role), 256, header_height, "main binary [1,2,4]"),
        (semantic_mask(main_role), 512, header_height, "main semantic roles"),
        (wrist_rgb, 0, header_height + 256, "wrist RGB"),
        (binary_mask(wrist_role), 256, header_height + 256, "wrist binary [1,2,4]"),
        (semantic_mask(wrist_role), 512, header_height + 256, "wrist semantic roles"),
    ]
    for image, x, y, label in cells:
        canvas.paste(Image.fromarray(image), (x, y))
        draw_cell_label(canvas, x, y, label)
    return np.asarray(canvas)


def video_path(root: Path, row: pd.Series, key: str) -> Path:
    chunk_index = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    return root / "videos" / key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_parquet(args.dataset_root / "meta" / "selection_manifest.parquet")
    instances = pd.read_parquet(args.dataset_root / "meta" / "segmentation_instances.parquet")
    episodes = read_episode_metadata(args.dataset_root)
    data = pads.dataset(args.dataset_root / "data", format="parquet")
    summary: dict[str, Any] = {}

    for task_index in args.task_indices:
        task_rows = selection[selection["task_index"] == task_index].sort_values("selected_rank_within_task")
        if task_rows.empty:
            raise KeyError(f"No selection manifest rows for task_index={task_index}")
        selected = task_rows.iloc[0]
        episode_index = int(selected["episode_index"])
        episode_rows = episodes[episodes["episode_index"] == episode_index]
        if len(episode_rows) != 1:
            raise RuntimeError(f"Expected one metadata row for episode_index={episode_index}, found {len(episode_rows)}")
        episode = episode_rows.iloc[0]
        title = f"task {task_index}: {selected['task_description']} | episode {episode_index}"

        table = data.to_table(
            columns=["episode_index", "frame_index", *MASK_COLUMNS],
            filter=pads.field("episode_index") == episode_index,
        ).sort_by("frame_index")
        main_role = nested_mask_array(table, MASK_COLUMNS[0])
        wrist_role = nested_mask_array(table, MASK_COLUMNS[1])
        main_rgb = decode_video(video_path(args.dataset_root, episode, "observation.images.image"))
        wrist_rgb = decode_video(video_path(args.dataset_root, episode, "observation.images.image2"))
        frame_count = min(len(main_rgb), len(wrist_rgb), len(main_role), len(wrist_role))
        if frame_count != int(episode["length"]):
            raise RuntimeError(
                f"Length mismatch for episode {episode_index}: metadata={int(episode['length'])}, decoded={frame_count}"
            )

        panels = [
            make_panel(main_rgb[idx], wrist_rgb[idx], main_role[idx], wrist_role[idx], title, idx)
            for idx in range(frame_count)
        ]
        slug = f"task_{task_index:02d}_episode_{episode_index:04d}"
        video_output = args.output_dir / f"{slug}_role_masks.mp4"
        imageio.mimsave(video_output, panels, fps=args.fps, macro_block_size=1)

        selected_frames = [0, frame_count // 2, frame_count - 1]
        poster = Image.new("RGB", (768 * 3, panels[0].shape[0]), (255, 255, 255))
        for panel_idx, frame_idx in enumerate(selected_frames):
            poster.paste(Image.fromarray(panels[frame_idx]), (panel_idx * 768, 0))
        poster_output = args.output_dir / f"{slug}_first_mid_last.png"
        poster.save(poster_output)

        role_rows = instances[instances["task_index"] == task_index]
        roles = role_rows[
            ["instance_id", "instance_name", "class_name", "role"]
        ].to_dict(orient="records")
        summary[str(task_index)] = {
            "task": str(selected["task_description"]),
            "episode_index": episode_index,
            "frame_count": frame_count,
            "video": str(video_output),
            "poster": str(poster_output),
            "roles": roles,
        }

    with (args.output_dir / "task_role_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
