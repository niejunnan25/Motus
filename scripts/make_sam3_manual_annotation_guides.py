#!/usr/bin/env python3
"""Create coordinate-grid RGB guides for manual SAM3 box overrides."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa_dir",
        default="artifacts/v3_sam3_mask_debug_v02_qa16",
        help="Directory containing sample_XXX/sample_XXX_sam3_mask_debug_17f.png sheets.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/v3_sam3_manual_annotation_guides",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=None,
        help="Sample ids to export. Defaults to issue-frame samples when --issue_frames_csv is set.",
    )
    parser.add_argument("--frame_ids", nargs="+", type=int, default=list(range(17)))
    parser.add_argument(
        "--issue_frames_csv",
        default=None,
        help="Optional CSV from make_sam3_mask_review_gallery.py; uses per-sample issue_frames.",
    )
    parser.add_argument("--label_width", type=int, default=380)
    parser.add_argument("--header_height", type=int, default=128)
    parser.add_argument("--cell_width", type=int, default=224)
    parser.add_argument("--cell_height", type=int, default=448)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--grid_step", type=int, default=20)
    parser.add_argument("--row_index", type=int, default=0, help="0 is RGB row.")
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


def draw_coordinate_grid(
    image: Image.Image,
    original_w: int,
    original_h: int,
    scale: int,
    grid_step: int,
    frame_idx: int,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    grid_color = (255, 235, 60, 105)
    major_color = (255, 80, 80, 150)
    text_bg = (0, 0, 0, 145)
    text_color = (255, 255, 255, 255)
    small_font = font(14 * scale)
    frame_font = font(22 * scale)

    for x in range(0, original_w + 1, grid_step):
        sx = x * scale
        color = major_color if x % (grid_step * 2) == 0 else grid_color
        draw.line((sx, 0, sx, original_h * scale), fill=color, width=1)
        if x < original_w:
            draw.rectangle((sx + 2, 2, sx + 44 * scale, 18 * scale), fill=text_bg)
            draw.text((sx + 4, 2), str(x), fill=text_color, font=small_font)

    for y in range(0, original_h + 1, grid_step):
        sy = y * scale
        color = major_color if y % (grid_step * 2) == 0 else grid_color
        draw.line((0, sy, original_w * scale, sy), fill=color, width=1)
        if y < original_h:
            draw.rectangle((2, sy + 2, 44 * scale, sy + 18 * scale), fill=text_bg)
            draw.text((4, sy + 2), str(y), fill=text_color, font=small_font)

    draw.rectangle((0, 0, 86 * scale, 30 * scale), fill=(0, 0, 0, 170))
    draw.text((8 * scale, 4 * scale), f"F{frame_idx:02d}", fill=(255, 255, 255, 255), font=frame_font)
    return out


def crop_rgb_frame(
    sheet: Image.Image,
    frame_idx: int,
    args: argparse.Namespace,
) -> Image.Image:
    row_y = args.header_height + args.row_index * args.cell_height
    x0 = args.label_width + frame_idx * args.cell_width
    crop = sheet.crop((x0, row_y, x0 + args.cell_width, row_y + args.cell_height))
    if args.scale != 1:
        crop = crop.resize((args.cell_width * args.scale, args.cell_height * args.scale), Image.Resampling.NEAREST)
    return draw_coordinate_grid(crop, args.cell_width, args.cell_height, args.scale, args.grid_step, frame_idx)


def make_sample_guide(sample_id: int, args: argparse.Namespace) -> Path:
    return make_sample_guide_for_frames(sample_id, args.frame_ids, args, "17f")


def make_sample_guide_for_frames(
    sample_id: int,
    frame_ids: Sequence[int],
    args: argparse.Namespace,
    suffix: str,
) -> Path:
    qa_dir = Path(args.qa_dir)
    sample_name = f"sample_{sample_id:03d}"
    sheet_path = qa_dir / sample_name / f"{sample_name}_sam3_mask_debug_17f.png"
    if not sheet_path.exists():
        raise FileNotFoundError(sheet_path)
    if not frame_ids:
        raise ValueError(f"No frame ids provided for {sample_name}")

    sheet = Image.open(sheet_path).convert("RGB")
    frames = [crop_rgb_frame(sheet, frame_idx, args) for frame_idx in frame_ids]
    frame_w, frame_h = frames[0].size
    header_h = 58 * args.scale
    out_w = frame_w * len(frames)
    out_h = header_h + frame_h
    out = Image.new("RGB", (out_w, out_h), (245, 246, 248))
    draw = ImageDraw.Draw(out)
    title_font = font(19 * args.scale)
    body_font = font(12 * args.scale)
    draw.text((12 * args.scale, 8 * args.scale), f"{sample_name} manual box guide | frames {list(frame_ids)}", fill=(15, 20, 25), font=title_font)
    draw.text(
        (12 * args.scale, 34 * args.scale),
        "Use original composed-frame coordinates: x in [0,224], y in [0,448]. Box format: xyxy_abs=[x0,y0,x1,y1].",
        fill=(80, 85, 92),
        font=body_font,
    )
    for idx, frame in enumerate(frames):
        out.paste(frame, (idx * frame_w, header_h))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_name}_manual_box_guide_{suffix}.png"
    out.save(out_path)
    return out_path


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
            frame_text = row.get("issue_frames", "")
            frames = [int(token) for token in frame_text.split() if token.strip()]
            if frames:
                issue_map[sample_id] = frames
    return issue_map


def default_sample_ids(issue_frame_map: Dict[int, List[int]]) -> List[int]:
    if issue_frame_map:
        return sorted(issue_frame_map)
    return [7, 8, 9, 12, 13, 14]


def main() -> None:
    args = parse_args()
    issue_frame_map = load_issue_frame_map(args.issue_frames_csv)
    sample_ids = args.sample_ids if args.sample_ids is not None else default_sample_ids(issue_frame_map)
    output_paths: List[Path] = []
    for sample_id in sample_ids:
        if issue_frame_map:
            frame_ids = issue_frame_map.get(sample_id)
            if not frame_ids:
                continue
            suffix = "issue_frames"
            output_paths.append(make_sample_guide_for_frames(sample_id, frame_ids, args, suffix))
        else:
            output_paths.append(make_sample_guide(sample_id, args))
    index_path = Path(args.output_dir) / "manual_box_guide_index.txt"
    with open(index_path, "w", encoding="utf-8") as file:
        for path in output_paths:
            file.write(str(path.resolve()) + "\n")
    print(f"Wrote {len(output_paths)} guides to {Path(args.output_dir).resolve()}")
    print(index_path.resolve())


if __name__ == "__main__":
    main()
