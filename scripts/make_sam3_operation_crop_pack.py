#!/usr/bin/env python3
"""Create operation-focused crops from SAM3 debug sheets for human QA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_sam3_mask_review_gallery import row_map_from_stats, sample_name


LABEL_W = 380
HEADER_H = 128
CELL_W = 224
CELL_H = 448
ROW_GAP = 10
VIEW_H = CELL_H // 2


ROWS = ["RGB", "role: object", "role: target", "role: robot", "final train mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_dir", default="artifacts/v3_sam3_mask_debug_manual_v4_draft_qa16")
    parser.add_argument(
        "--issue_frames_csv",
        default="artifacts/v3_sam3_mask_review_manual_v4_draft/summary/issue_frames.csv",
    )
    parser.add_argument("--output_dir", default="artifacts/v3_sam3_mask_review_manual_v4_draft/operation_crops")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--pad", type=int, default=18)
    parser.add_argument("--min_crop", type=int, default=96)
    parser.add_argument("--crop_size", type=int, default=280)
    parser.add_argument("--diff_threshold", type=float, default=18.0)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_issue_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def stats_path(qa_dir: Path, sample_id: int) -> Path:
    name = sample_name(sample_id)
    return qa_dir / name / f"{name}_sam3_mask_debug_stats.json"


def sheet_path(qa_dir: Path, sample_id: int) -> Path:
    name = sample_name(sample_id)
    return qa_dir / name / f"{name}_sam3_mask_debug_17f.png"


def load_stats(qa_dir: Path, sample_id: int) -> Dict:
    with open(stats_path(qa_dir, sample_id), "r", encoding="utf-8") as file:
        return json.load(file)


def cell_crop(sheet: Image.Image, row_idx: int, frame_idx: int, view_name: str) -> Image.Image:
    if view_name not in ("top", "bottom"):
        raise ValueError(view_name)
    x0 = LABEL_W + frame_idx * CELL_W
    y0 = HEADER_H + row_idx * (CELL_H + ROW_GAP)
    if view_name == "bottom":
        y0 += VIEW_H
    return sheet.crop((x0, y0, x0 + CELL_W, y0 + VIEW_H)).convert("RGB")


def bbox_from_overlay(rgb: Image.Image, overlay: Image.Image, threshold: float, pad: int, min_crop: int) -> Tuple[int, int, int, int]:
    rgb_arr = np.asarray(rgb).astype(np.int16)
    overlay_arr = np.asarray(overlay).astype(np.int16)
    diff = np.abs(overlay_arr - rgb_arr).mean(axis=2)
    mask = diff > threshold
    if not mask.any():
        return (0, 0, CELL_W, VIEW_H)
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(CELL_W, int(xs.max()) + 1 + pad)
    y1 = min(VIEW_H, int(ys.max()) + 1 + pad)

    width = x1 - x0
    height = y1 - y0
    if width < min_crop:
        extra = min_crop - width
        x0 = max(0, x0 - extra // 2)
        x1 = min(CELL_W, x1 + extra - extra // 2)
    if height < min_crop:
        extra = min_crop - height
        y0 = max(0, y0 - extra // 2)
        y1 = min(VIEW_H, y1 + extra - extra // 2)
    return (x0, y0, x1, y1)


def resized_crop(image: Image.Image, bbox: Tuple[int, int, int, int], scale: int) -> Image.Image:
    crop = image.crop(bbox)
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


def fit_to_cell(image: Image.Image, cell_size: int) -> Image.Image:
    scale = min(cell_size / image.width, cell_size / image.height)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.NEAREST)
    out = Image.new("RGB", (cell_size, cell_size), (248, 250, 252))
    out.paste(resized, ((cell_size - new_w) // 2, (cell_size - new_h) // 2))
    return out


def draw_crop_border(image: Image.Image, color: Tuple[int, int, int], width: int = 3) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for idx in range(width):
        draw.rectangle((idx, idx, out.width - 1 - idx, out.height - 1 - idx), outline=color)
    return out


def crop_rows_for_frame(
    sheet: Image.Image,
    row_map: Dict[str, int],
    frame_idx: int,
    view_name: str,
    args: argparse.Namespace,
) -> List[Tuple[str, Image.Image]]:
    rgb = cell_crop(sheet, row_map["RGB"], frame_idx, view_name)
    final = cell_crop(sheet, row_map["final train mask"], frame_idx, view_name)
    bbox = bbox_from_overlay(rgb, final, args.diff_threshold, args.pad, args.min_crop)

    crops: List[Tuple[str, Image.Image]] = []
    for row_name in ROWS:
        if row_name not in row_map:
            continue
        crop = cell_crop(sheet, row_map[row_name], frame_idx, view_name)
        fitted = fit_to_cell(resized_crop(crop, bbox, args.scale), args.crop_size)
        crops.append((row_name, fitted))
    return crops


def make_sample_pack(sample_id: int, frame_ids: Sequence[int], issue_row: Dict[str, str], args: argparse.Namespace) -> Path:
    qa_dir = Path(args.qa_dir)
    stats = load_stats(qa_dir, sample_id)
    row_map = row_map_from_stats(stats)
    sheet = Image.open(sheet_path(qa_dir, sample_id)).convert("RGB")
    name = sample_name(sample_id)
    task = stats.get("window", {}).get("task_text", "")

    panels: List[Tuple[str, Dict[str, Image.Image]]] = []
    for frame_idx in frame_ids:
        for view_name in ("top", "bottom"):
            crops = crop_rows_for_frame(sheet, row_map, frame_idx, view_name, args)
            if not crops:
                continue
            panels.append((f"F{frame_idx:02d}\n{view_name}", {row_name: crop for row_name, crop in crops}))

    if not panels:
        raise RuntimeError(f"No operation crops created for {name}")

    title_font = font(28)
    body_font = font(17)
    label_font = font(18)
    header_h = 150
    row_label_w = 220
    col_header_h = 54
    gap = 8
    cell = args.crop_size
    out_w = row_label_w + len(panels) * (cell + gap) + gap
    out_h = header_h + col_header_h + len(ROWS) * (cell + gap) + gap
    out = Image.new("RGB", (out_w, out_h), (245, 247, 250))
    draw = ImageDraw.Draw(out)
    draw.text((24, 20), f"{name} operation-focused crop review", fill=(15, 20, 25), font=title_font)
    draw.text((24, 62), task, fill=(55, 63, 75), font=body_font)
    note = f"status={issue_row.get('current_status','')} | frames={list(frame_ids)} | crop bbox inferred from final_train_mask overlay"
    draw.text((24, 96), note, fill=(105, 84, 30), font=body_font)

    colors = {
        "RGB": (80, 90, 105),
        "role: object": (255, 80, 80),
        "role: target": (45, 180, 255),
        "role: robot": (180, 105, 255),
        "final train mask": (220, 35, 35),
    }
    # Column headers.
    for panel_idx, (panel_title, _crops) in enumerate(panels):
        x = row_label_w + gap + panel_idx * (cell + gap)
        y = header_h
        draw.rectangle((x, y, x + cell, y + col_header_h), fill=(232, 237, 244), outline=(211, 218, 228))
        lines = panel_title.splitlines()
        draw.text((x + 10, y + 7), lines[0], fill=(22, 28, 36), font=label_font)
        draw.text((x + 10, y + 29), lines[1], fill=(92, 102, 116), font=label_font)

    # Row labels and crop cells.
    for row_idx, row_name in enumerate(ROWS):
        y = header_h + col_header_h + gap + row_idx * (cell + gap)
        draw.rectangle((0, y, row_label_w, y + cell), fill=(234, 238, 244))
        draw.text((18, y + 18), row_name, fill=(35, 42, 52), font=label_font)
        for panel_idx, (_panel_title, crop_map) in enumerate(panels):
            x = row_label_w + gap + panel_idx * (cell + gap)
            draw.rectangle((x, y, x + cell, y + cell), fill=(255, 255, 255), outline=(211, 218, 228))
            crop = crop_map.get(row_name)
            if crop is None:
                continue
            crop = draw_crop_border(crop, colors.get(row_name, (80, 80, 80)))
            out.paste(crop, (x, y))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_operation_crop_review.png"
    out.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    issue_rows = load_issue_rows(Path(args.issue_frames_csv))
    if args.sample_ids is not None:
        allowed = {sample_name(sample_id) for sample_id in args.sample_ids}
        issue_rows = [row for row in issue_rows if row.get("sample") in allowed]
    output_paths = []
    for row in issue_rows:
        sample = row.get("sample", "")
        if not sample.startswith("sample_"):
            continue
        sample_id = int(sample.rsplit("_", 1)[1])
        frame_ids = [int(token) for token in row.get("issue_frames", "").split() if token.strip()]
        if not frame_ids:
            continue
        output_paths.append(make_sample_pack(sample_id, frame_ids, row, args))
    index_path = Path(args.output_dir) / "operation_crop_index.txt"
    with open(index_path, "w", encoding="utf-8") as file:
        for path in output_paths:
            file.write(str(path.resolve()) + "\n")
    print(f"Wrote {len(output_paths)} operation crop reviews")
    print(index_path.resolve())


if __name__ == "__main__":
    main()
