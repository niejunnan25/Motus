#!/usr/bin/env python3
"""Preview manual SAM3 box override coordinates on fixed 17F debug sheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa_dir",
        default="artifacts/v3_sam3_mask_debug_manual_v2_qa16",
        help="Directory containing sample_XXX/sample_XXX_sam3_mask_debug_17f.png sheets.",
    )
    parser.add_argument(
        "--manual_overrides",
        default="configs/sam3_manual_overrides_debug16.v2.json",
        help="Manual override JSON to preview.",
    )
    parser.add_argument(
        "--issue_frames_csv",
        default=None,
        help="Optional issue_frames.csv. If set, each sample uses its own issue frames.",
    )
    parser.add_argument("--output_dir", default="artifacts/v3_sam3_manual_override_previews")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--frame_ids", nargs="+", type=int, default=list(range(17)))
    parser.add_argument("--label_width", type=int, default=380)
    parser.add_argument("--header_height", type=int, default=128)
    parser.add_argument("--cell_width", type=int, default=224)
    parser.add_argument("--cell_height", type=int, default=448)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--grid_step", type=int, default=20)
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


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("samples", data)


def load_issue_frame_map(path: str | None) -> Dict[int, List[int]]:
    if not path:
        return {}
    issue_map: Dict[int, List[int]] = {}
    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sample = row.get("sample", "")
            if not sample.startswith("sample_"):
                continue
            sample_id = int(sample.split("_", 1)[1])
            frames = [int(token) for token in row.get("issue_frames", "").split() if token.strip()]
            if frames:
                issue_map[sample_id] = frames
    return issue_map


def cxcywh_norm_to_xyxy_abs(box: Sequence[float], image_w: int, image_h: int) -> List[float]:
    cx, cy, w, h = [float(v) for v in box]
    return [
        (cx - 0.5 * w) * image_w,
        (cy - 0.5 * h) * image_h,
        (cx + 0.5 * w) * image_w,
        (cy + 0.5 * h) * image_h,
    ]


def box_entry_to_xyxy_abs(entry: Dict, image_w: int, image_h: int) -> List[float]:
    if "xyxy_abs" in entry:
        return [float(v) for v in entry["xyxy_abs"]]
    if "xyxy_norm" in entry:
        x0, y0, x1, y1 = [float(v) for v in entry["xyxy_norm"]]
        return [x0 * image_w, y0 * image_h, x1 * image_w, y1 * image_h]
    if "cxcywh_norm" in entry:
        return cxcywh_norm_to_xyxy_abs(entry["cxcywh_norm"], image_w, image_h)
    raise KeyError(f"Manual override entry must contain a box field: {entry}")


def interpolate_box(frame_idx: int, keyed_boxes: Dict[int, Sequence[float]]) -> List[float] | None:
    if not keyed_boxes:
        return None
    if frame_idx in keyed_boxes:
        return [float(v) for v in keyed_boxes[frame_idx]]
    keys = sorted(keyed_boxes)
    if frame_idx < keys[0] or frame_idx > keys[-1]:
        return None
    left_keys = [key for key in keys if key < frame_idx]
    right_keys = [key for key in keys if key > frame_idx]
    if not left_keys or not right_keys:
        return None
    left = left_keys[-1]
    right = right_keys[0]
    alpha = (frame_idx - left) / float(right - left)
    left_box = keyed_boxes[left]
    right_box = keyed_boxes[right]
    return [
        (1.0 - alpha) * float(left_box[idx]) + alpha * float(right_box[idx])
        for idx in range(4)
    ]


def entries_for_sample_frame(
    manual_overrides: Dict,
    sample_id: int,
    frame_idx: int,
    image_w: int,
    image_h: int,
) -> List[Dict]:
    sample_cfg = manual_overrides.get(str(sample_id), manual_overrides.get(sample_id, {}))
    entries = sample_cfg.get("objects", sample_cfg.get("entries", [])) if isinstance(sample_cfg, dict) else []
    out = []
    for entry_idx, entry in enumerate(entries):
        box = None
        if "frames" in entry:
            keyed_boxes = {
                int(key): box_entry_to_xyxy_abs(value, image_w, image_h)
                for key, value in entry["frames"].items()
            }
            if entry.get("interpolate", True):
                box = interpolate_box(frame_idx, keyed_boxes)
            else:
                box = keyed_boxes.get(frame_idx)
        elif "frame" in entry:
            if int(entry["frame"]) == frame_idx:
                box = box_entry_to_xyxy_abs(entry, image_w, image_h)
        else:
            box = box_entry_to_xyxy_abs(entry, image_w, image_h)
        if box is None:
            continue
        out.append(
            {
                "name": entry.get("name", f"manual_{entry_idx}"),
                "role": entry.get("role", "object"),
                "box": box,
            }
        )
    return out


def draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, scale: int, grid_step: int) -> None:
    grid_color = (255, 235, 60, 110)
    major_color = (255, 70, 70, 165)
    small_font = font(13 * scale)
    for x in range(0, width + 1, grid_step):
        sx = x * scale
        color = major_color if x % (grid_step * 2) == 0 else grid_color
        draw.line((sx, 0, sx, height * scale), fill=color, width=1)
        if x < width:
            draw.text((sx + 3, 2), str(x), fill=(255, 255, 255), font=small_font, stroke_width=2, stroke_fill=(0, 0, 0))
    for y in range(0, height + 1, grid_step):
        sy = y * scale
        color = major_color if y % (grid_step * 2) == 0 else grid_color
        draw.line((0, sy, width * scale, sy), fill=color, width=1)
        if y < height:
            draw.text((3, sy + 2), str(y), fill=(255, 255, 255), font=small_font, stroke_width=2, stroke_fill=(0, 0, 0))


def crop_rgb_frame(sheet: Image.Image, frame_idx: int, args: argparse.Namespace) -> Image.Image:
    x0 = args.label_width + frame_idx * args.cell_width
    y0 = args.header_height
    crop = sheet.crop((x0, y0, x0 + args.cell_width, y0 + args.cell_height))
    if args.scale != 1:
        crop = crop.resize((args.cell_width * args.scale, args.cell_height * args.scale), Image.Resampling.NEAREST)
    return crop


def draw_boxes(
    image: Image.Image,
    frame_idx: int,
    entries: Sequence[Dict],
    args: argparse.Namespace,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    draw_grid(draw, args.cell_width, args.cell_height, args.scale, args.grid_step)
    frame_font = font(22 * args.scale)
    label_font = font(12 * args.scale)
    draw.rectangle((0, 0, 92 * args.scale, 30 * args.scale), fill=(0, 0, 0, 170))
    draw.text((8 * args.scale, 4 * args.scale), f"F{frame_idx:02d}", fill=(255, 255, 255), font=frame_font)

    colors = {
        "object": (255, 70, 70, 255),
        "target": (50, 180, 255, 255),
        "robot": (180, 105, 255, 255),
        "context": (110, 150, 160, 255),
    }
    for entry in entries:
        color = colors.get(entry["role"], (255, 210, 60, 255))
        x0, y0, x1, y1 = [float(v) * args.scale for v in entry["box"]]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(3, args.scale + 1))
        label = f"{entry['role']}:{entry['name']}"
        tx = max(0, min(x0, args.cell_width * args.scale - 220))
        ty = max(0, y0 - 22 * args.scale)
        draw.rectangle((tx, ty, tx + max(110, len(label) * 7) * args.scale, ty + 18 * args.scale), fill=(0, 0, 0, 150))
        draw.text((tx + 3 * args.scale, ty + 1 * args.scale), label, fill=color, font=label_font)
    return out


def sample_name(sample_id: int) -> str:
    return f"sample_{sample_id:03d}"


def make_preview(
    sample_id: int,
    frame_ids: Sequence[int],
    manual_overrides: Dict,
    args: argparse.Namespace,
) -> Path:
    name = sample_name(sample_id)
    sheet_path = Path(args.qa_dir) / name / f"{name}_sam3_mask_debug_17f.png"
    if not sheet_path.exists():
        raise FileNotFoundError(sheet_path)
    sheet = Image.open(sheet_path).convert("RGB")
    cells = []
    for frame_idx in frame_ids:
        entries = entries_for_sample_frame(
            manual_overrides,
            sample_id,
            frame_idx,
            args.cell_width,
            args.cell_height,
        )
        crop = crop_rgb_frame(sheet, frame_idx, args)
        cells.append(draw_boxes(crop, frame_idx, entries, args))

    cell_w, cell_h = cells[0].size
    header_h = 58 * args.scale
    out = Image.new("RGB", (cell_w * len(cells), header_h + cell_h), (245, 246, 248))
    draw = ImageDraw.Draw(out)
    title_font = font(19 * args.scale)
    body_font = font(12 * args.scale)
    draw.text((12 * args.scale, 8 * args.scale), f"{name} manual override preview | frames {list(frame_ids)}", fill=(15, 20, 25), font=title_font)
    draw.text((12 * args.scale, 34 * args.scale), f"source: {args.manual_overrides}", fill=(80, 85, 92), font=body_font)
    for idx, cell in enumerate(cells):
        out.paste(cell, (idx * cell_w, header_h))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_manual_override_preview.png"
    out.save(out_path)
    return out_path


def default_sample_ids(manual_overrides: Dict, issue_map: Dict[int, List[int]]) -> List[int]:
    if issue_map:
        return sorted(issue_map)
    return sorted(int(key) for key in manual_overrides.keys())


def main() -> None:
    args = parse_args()
    manual_overrides = load_json(args.manual_overrides)
    issue_map = load_issue_frame_map(args.issue_frames_csv)
    sample_ids = args.sample_ids if args.sample_ids is not None else default_sample_ids(manual_overrides, issue_map)
    output_paths = []
    for sample_id in sample_ids:
        frame_ids = issue_map.get(sample_id, args.frame_ids)
        output_paths.append(make_preview(sample_id, frame_ids, manual_overrides, args))

    index_path = Path(args.output_dir) / "manual_override_preview_index.txt"
    with open(index_path, "w", encoding="utf-8") as file:
        for path in output_paths:
            file.write(str(path.resolve()) + "\n")
    print(f"Wrote {len(output_paths)} previews to {Path(args.output_dir).resolve()}")
    print(index_path.resolve())


if __name__ == "__main__":
    main()
