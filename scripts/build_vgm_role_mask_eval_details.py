#!/usr/bin/env python3
"""Build readable RGB, role-mask, and full-frame details for the 8-way VGM eval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None
FRAME_COUNT = 53
HEADER_HEIGHT = 230
TIMELINE_HEIGHT = 32
ROW_LABEL_WIDTH = 130
WINDOW_KEYS = ("episode_name", "condition_idx", "frame_indices", "total_frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frames_per_block", type=int, default=18)
    parser.add_argument("--full_frames_per_block", type=int, default=9)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def find_eval_dir(root: Path) -> Path:
    if (root / "metrics.json").is_file() and (root / "windows.json").is_file():
        return root
    matches = sorted(path.parent for path in root.rglob("metrics.json") if (path.parent / "windows.json").is_file())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completed eval below {root}, found {matches}")
    return matches[0]


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def windows_match(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    return len(reference) == len(candidate) and all(
        all(left.get(key) == right.get(key) for key in WINDOW_KEYS)
        for left, right in zip(reference, candidate)
    )


def metric(metrics: dict[str, Any], group: str, key: str) -> float:
    value = metrics.get(group, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def read_sheet_rows(path: Path) -> tuple[Image.Image, Image.Image, int]:
    sheet = Image.open(path).convert("RGB")
    if sheet.width % FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} frames in {path}, got width={sheet.width}")
    frame_width = sheet.width // FRAME_COUNT
    label_height = sheet.height - 2 * frame_width
    if label_height < 0:
        raise RuntimeError(f"Unexpected sheet dimensions for square four-grid frames: {path}: {sheet.size}")
    gt = sheet.crop((0, label_height, sheet.width, label_height + frame_width))
    pred = sheet.crop((0, label_height + frame_width, sheet.width, sheet.height))
    return gt, pred, frame_width


def extract_panel(row: Image.Image, frame_width: int, panel: str) -> tuple[Image.Image, int]:
    if panel == "full":
        return row, frame_width
    half_width = frame_width // 2
    if frame_width % 2:
        raise RuntimeError(f"Four-grid frame width must be even, got {frame_width}")
    source_x = 0 if panel == "rgb" else half_width
    output = Image.new("RGB", (FRAME_COUNT * half_width, row.height), (255, 255, 255))
    for frame_idx in range(FRAME_COUNT):
        frame_x = frame_idx * frame_width
        crop = row.crop((frame_x + source_x, 0, frame_x + source_x + half_width, row.height))
        output.paste(crop, (frame_idx * half_width, 0))
    return output, half_width


def draw_header(
    canvas: Image.Image,
    model: dict[str, Any],
    window: dict[str, Any],
    metrics: dict[str, Any],
    sample_name: str,
    panel: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    regular_font = load_font(18)
    bold_font = load_font(18, bold=True)
    caption = window.get("language_caption") or window.get("task_text") or "<missing>"
    rgb_mse = metric(metrics, "samples", "rgb_generated_mse")
    rgb_psnr = metric(metrics, "samples", "rgb_generated_psnr")
    interaction_mse = metric(metrics, "samples", "rgb_generated_interaction_mse")
    foreground_iou = metric(metrics, "samples", "role_mask_foreground_iou")
    macro_iou = metric(metrics, "samples", "role_mask_macro_iou")
    total_loss = metric(metrics, "loss", "total_loss")
    rgb_loss = metric(metrics, "loss", "rgb_loss")
    role_loss = metric(metrics, "loss", "role_mask_loss")

    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill=(247, 248, 250))
    draw.text((20, 12), f"{sample_name} | {model['label']} | panel={panel}", font=title_font, fill=(20, 24, 31))
    draw.text(
        (20, 50),
        f"episode: {window.get('episode_name')} | condition_idx={window.get('condition_idx')}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text((20, 77), model.get("condition", ""), font=regular_font, fill=(20, 24, 31))
    draw.text(
        (20, 104),
        f"RGB MSE {rgb_mse:.5f} | RGB PSNR {rgb_psnr:.2f} | interaction MSE {interaction_mse:.5f} | "
        f"foreground IoU {foreground_iou:.4f} | macro IoU {macro_iou:.4f}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text(
        (20, 131),
        f"eval loss total {total_loss:.5f} | rgb {rgb_loss:.5f} | role mask {role_loss:.5f}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text((20, 158), "language_caption:", font=bold_font, fill=(20, 24, 31))
    y = 182
    for line in textwrap.wrap(caption, width=155, break_long_words=False, break_on_hyphens=False)[:2]:
        draw.text((20, y), line, font=regular_font, fill=(20, 24, 31))
        y += 22


def build_detail(
    gt_row: Image.Image,
    pred_row: Image.Image,
    frame_width: int,
    model: dict[str, Any],
    window: dict[str, Any],
    metrics: dict[str, Any],
    sample_name: str,
    panel: str,
    frames_per_block: int,
) -> Image.Image:
    frame_height = gt_row.height
    block_count = math.ceil(FRAME_COUNT / frames_per_block)
    block_height = TIMELINE_HEIGHT + frame_height * 2
    canvas = Image.new(
        "RGB",
        (ROW_LABEL_WIDTH + frames_per_block * frame_width, HEADER_HEIGHT + block_count * block_height),
        (238, 240, 243),
    )
    draw_header(canvas, model, window, metrics, sample_name, panel)
    draw = ImageDraw.Draw(canvas)
    timeline_font = load_font(17, bold=True)
    row_font = load_font(21, bold=True)

    for block_idx in range(block_count):
        start = block_idx * frames_per_block
        end = min(FRAME_COUNT, start + frames_per_block)
        block_y = HEADER_HEIGHT + block_idx * block_height
        gt_y = block_y + TIMELINE_HEIGHT
        pred_y = gt_y + frame_height
        draw.rectangle((0, block_y, canvas.width - 1, block_y + block_height - 1), fill="white", outline=(152, 158, 167), width=2)
        draw.rectangle((0, block_y, canvas.width, gt_y), fill=(230, 233, 238))
        draw.rectangle((0, gt_y, ROW_LABEL_WIDTH, pred_y + frame_height), fill=(244, 246, 249))
        draw.text((38, gt_y + frame_height // 2 - 14), "GT", font=row_font, fill=(20, 24, 31))
        draw.text((8, pred_y + frame_height // 2 - 14), "Prediction", font=row_font, fill=(20, 24, 31))

        for local_idx, frame_idx in enumerate(range(start, end)):
            x = ROW_LABEL_WIDTH + local_idx * frame_width
            source_x = frame_idx * frame_width
            draw.text((x + 7, block_y + 6), f"t={frame_idx:02d}", font=timeline_font, fill=(20, 24, 31))
            draw.line((x, gt_y, x, pred_y + frame_height), fill=(190, 194, 201), width=1)
            canvas.paste(gt_row.crop((source_x, 0, source_x + frame_width, frame_height)), (x, gt_y))
            canvas.paste(pred_row.crop((source_x, 0, source_x + frame_width, frame_height)), (x, pred_y))
        draw.line((0, pred_y, canvas.width, pred_y), fill=(255, 196, 0), width=4)
    return canvas


def flatten_metrics(model: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"label": model["label"], "slug": model["slug"], "render_mode": model["render_mode"]}
    for group in ("loss", "samples"):
        for key, value in model["metrics"].get(group, {}).items():
            if isinstance(value, (int, float)):
                row[f"{group}.{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    models = manifest["models"]
    for model in models:
        model["eval_dir"] = find_eval_dir(Path(model["eval_dir"]))
        model["windows"] = load_json(model["eval_dir"] / "windows.json")
        model["metrics"] = load_json(model["eval_dir"] / "metrics.json")

    reference_windows = models[0]["windows"]
    if len(reference_windows) != manifest["num_samples"]:
        raise RuntimeError(f"Expected {manifest['num_samples']} windows, found {len(reference_windows)}")
    mismatched = [model["label"] for model in models[1:] if not windows_match(reference_windows, model["windows"])]
    if mismatched:
        raise RuntimeError(f"Fixed-window mismatch: {mismatched}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [flatten_metrics(model) for model in models]
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "metrics_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(rows, indent=2))
    (args.output_dir / "windows.json").write_text(json.dumps(reference_windows, indent=2))
    (args.output_dir / "windows_match.txt").write_text("same_fixed_windows=true\n")

    outputs: list[str] = []
    for sample_idx, window in enumerate(reference_windows):
        sample_name = f"sample_{sample_idx:03d}"
        sample_dir = args.output_dir / "by_sample" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        reference_rgb: bytes | None = None
        reference_masks: dict[str, bytes] = {}

        for model_idx, model in enumerate(models, start=1):
            sheet = model["eval_dir"] / "samples" / f"{sample_name}_sheet.png"
            gt_full, pred_full, full_width = read_sheet_rows(sheet)
            gt_rgb, rgb_width = extract_panel(gt_full, full_width, "rgb")
            pred_rgb, _ = extract_panel(pred_full, full_width, "rgb")
            gt_mask, mask_width = extract_panel(gt_full, full_width, "mask")
            pred_mask, _ = extract_panel(pred_full, full_width, "mask")

            rgb_bytes = gt_rgb.tobytes()
            if reference_rgb is None:
                reference_rgb = rgb_bytes
            elif rgb_bytes != reference_rgb:
                raise RuntimeError(f"RGB GT differs for {sample_name}: {model['label']}")
            render_mode = model["render_mode"]
            mask_bytes = gt_mask.tobytes()
            if render_mode in reference_masks and mask_bytes != reference_masks[render_mode]:
                raise RuntimeError(f"Mask GT differs within {render_mode} for {sample_name}: {model['label']}")
            reference_masks.setdefault(render_mode, mask_bytes)

            panels = [
                ("rgb", gt_rgb, pred_rgb, rgb_width, args.frames_per_block),
                ("mask", gt_mask, pred_mask, mask_width, args.frames_per_block),
                ("full", gt_full, pred_full, full_width, args.full_frames_per_block),
            ]
            for panel, gt_row, pred_row, frame_width, block_size in panels:
                detail = build_detail(
                    gt_row, pred_row, frame_width, model, window, model["metrics"], sample_name, panel, block_size
                )
                output = sample_dir / f"{model_idx:02d}_{model['slug']}_{panel}_gt_over_pred.png"
                detail.save(output)
                outputs.append(str(output))

    metadata = {
        "description": "Per-model horizontal GT-over-pred details for RGB, role mask, and full four-grid frames.",
        "num_samples": len(reference_windows),
        "num_models": len(models),
        "frame_count": FRAME_COUNT,
        "expected_pngs": len(reference_windows) * len(models) * 3,
        "actual_pngs": len(outputs),
        "outputs": outputs,
    }
    (args.output_dir / "detail_outputs.json").write_text(json.dumps(metadata, indent=2))
    if metadata["actual_pngs"] != metadata["expected_pngs"]:
        raise RuntimeError(f"Output count mismatch: {metadata}")


if __name__ == "__main__":
    main()
