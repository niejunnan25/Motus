#!/usr/bin/env python3
"""Create candidate identity sheets for ambiguous SAM3 object cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


LABEL_W = 380
HEADER_H = 128
CELL_W = 224
CELL_H = 448
ROW_GAP = 10
VIEW_H = CELL_H // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_dir", default="artifacts/v3_sam3_mask_debug_manual_v4_draft_qa16")
    parser.add_argument(
        "--candidates_json",
        default="configs/sam3_identity_candidates_debug16.sample009.json",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/v3_sam3_mask_review_manual_v4_draft/identity_decision_aids",
    )
    parser.add_argument("--crop_scale", type=int, default=5)
    parser.add_argument("--pad", type=int, default=12)
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


def sample_id_from_name(sample: str) -> int:
    return int(sample.rsplit("_", 1)[1])


def sheet_path(qa_dir: Path, sample: str) -> Path:
    return qa_dir / sample / f"{sample}_sam3_mask_debug_17f.png"


def rgb_cell(sheet: Image.Image, frame_idx: int, view: str) -> Image.Image:
    x0 = LABEL_W + frame_idx * CELL_W
    y0 = HEADER_H
    if view == "bottom":
        y0 += VIEW_H
    elif view != "top":
        raise ValueError(f"view must be top or bottom, got {view}")
    return sheet.crop((x0, y0, x0 + CELL_W, y0 + VIEW_H)).convert("RGB")


def box_for_view(candidate: Dict) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in candidate["xyxy_abs"]]
    if candidate["view"] == "bottom":
        y0 -= VIEW_H
        y1 -= VIEW_H
    return x0, y0, x1, y1


def clamp_box(box: Tuple[int, int, int, int], pad: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(CELL_W, x1 + pad),
        min(VIEW_H, y1 + pad),
    )


def candidate_crop(cell: Image.Image, candidate: Dict, pad: int, scale: int) -> Image.Image:
    box = clamp_box(box_for_view(candidate), pad)
    crop = cell.crop(box)
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


def draw_candidate_on_cell(cell: Image.Image, candidate: Dict) -> Image.Image:
    out = cell.copy()
    draw = ImageDraw.Draw(out)
    box = box_for_view(candidate)
    color = (255, 80, 80) if candidate.get("status") == "current_manual" else (45, 180, 255)
    for idx in range(3):
        draw.rectangle((box[0] - idx, box[1] - idx, box[2] + idx, box[3] + idx), outline=color)
    draw.text((max(2, box[0]), max(2, box[1] - 18)), candidate["id"].split("_", 1)[0], fill=color, font=font(14))
    return out


def make_sheet(config: Dict, args: argparse.Namespace) -> Path:
    sample = config["sample"]
    qa_dir = Path(args.qa_dir)
    sheet = Image.open(sheet_path(qa_dir, sample)).convert("RGB")
    candidates: List[Dict] = config["candidates"]
    title_font = font(28)
    body_font = font(17)
    label_font = font(16)
    small_font = font(14)

    card_w = 360
    card_h = 430
    cols = 3
    rows = (len(candidates) + cols - 1) // cols
    header_h = 150
    gap = 18
    out_w = cols * card_w + (cols + 1) * gap
    out_h = header_h + rows * card_h + (rows + 1) * gap
    out = Image.new("RGB", (out_w, out_h), (245, 247, 250))
    draw = ImageDraw.Draw(out)
    draw.text((24, 18), f"{sample} identity decision aid", fill=(15, 20, 25), font=title_font)
    draw.text((24, 60), config.get("task_text", ""), fill=(55, 63, 75), font=body_font)
    draw.text((24, 96), config.get("note", ""), fill=(105, 84, 30), font=body_font)

    for idx, candidate in enumerate(candidates):
        col = idx % cols
        row = idx // cols
        x = gap + col * (card_w + gap)
        y = header_h + gap + row * (card_h + gap)
        draw.rectangle((x, y, x + card_w, y + card_h), fill=(255, 255, 255), outline=(211, 218, 228))
        cell = rgb_cell(sheet, int(candidate["frame"]), candidate["view"])
        full = draw_candidate_on_cell(cell, candidate).resize((CELL_W * 1, VIEW_H * 1), Image.Resampling.NEAREST)
        crop = candidate_crop(cell, candidate, args.pad, args.crop_scale)
        crop_max_w = card_w - 24
        crop_max_h = 180
        crop_scale = min(crop_max_w / crop.width, crop_max_h / crop.height, 1.0)
        if crop_scale < 1.0:
            crop = crop.resize((int(crop.width * crop_scale), int(crop.height * crop_scale)), Image.Resampling.NEAREST)

        draw.text((x + 12, y + 10), candidate["id"], fill=(22, 28, 36), font=label_font)
        draw.text((x + 12, y + 34), f"F{candidate['frame']:02d} {candidate['view']} | {candidate['status']}", fill=(86, 96, 110), font=small_font)
        for line_idx, line in enumerate(wrap_text(candidate["label"], 39)[:2]):
            draw.text((x + 12, y + 58 + line_idx * 18), line, fill=(55, 63, 75), font=small_font)
        out.paste(full, (x + 12, y + 100))
        out.paste(crop, (x + 12, y + 100 + VIEW_H + 12))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample}_identity_decision_aid.png"
    out.save(out_path)
    return out_path


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def main() -> None:
    args = parse_args()
    with open(args.candidates_json, "r", encoding="utf-8") as file:
        config = json.load(file)
    path = make_sheet(config, args)
    index_path = Path(args.output_dir) / "identity_decision_aid_index.txt"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write(str(path.resolve()) + "\n")
    print(path.resolve())
    print(index_path.resolve())


if __name__ == "__main__":
    main()
