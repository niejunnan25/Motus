#!/usr/bin/env python3
"""Render VGM details with aligned GT rows above prediction rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from build_vgm_bridge_readable_comparison import (
    find_eval_dir,
    load_font,
    load_json,
    metric_value,
    read_sheet_rows,
    windows_match,
    wrap_text,
)


FRAME_COUNT = 53
HEADER_HEIGHT = 230
TIMELINE_HEIGHT = 32
ROW_LABEL_WIDTH = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frames_per_block", type=int, default=18)
    return parser.parse_args()


def draw_header(
    canvas: Image.Image,
    model: dict[str, Any],
    window: dict[str, Any],
    metrics: dict[str, Any],
    sample_name: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    regular_font = load_font(19)
    bold_font = load_font(19, bold=True)
    mse = metric_value(metrics, "samples", "generated_mse")
    psnr = metric_value(metrics, "samples", "generated_psnr")
    interaction_mse = metric_value(metrics, "samples", "generated_interaction_mse")
    loss = metric_value(metrics, "loss", "total_loss")
    caption = window.get("language_caption") or window.get("task_text") or "<missing>"

    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill=(247, 248, 250))
    draw.text((20, 14), f"{sample_name} | {model['label']}", font=title_font, fill=(20, 24, 31))
    draw.text(
        (20, 52),
        f"episode: {window.get('episode_name')} | condition_idx={window.get('condition_idx')}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text((20, 80), model.get("condition", ""), font=regular_font, fill=(20, 24, 31))
    draw.text(
        (20, 108),
        f"MSE {mse:.5f} | PSNR {psnr:.2f} | interaction MSE {interaction_mse:.5f} | loss {loss:.5f}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text((20, 140), "language_caption:", font=bold_font, fill=(20, 24, 31))
    y = 168
    for line in wrap_text(caption, 145)[:3]:
        draw.text((20, y), line, font=regular_font, fill=(20, 24, 31))
        y += 24


def paste_frame_range(
    canvas: Image.Image,
    row: Image.Image,
    start_frame: int,
    end_frame: int,
    frame_width: int,
    output_y: int,
) -> None:
    for local_idx, frame_idx in enumerate(range(start_frame, end_frame)):
        frame = row.crop((frame_idx * frame_width, 0, (frame_idx + 1) * frame_width, row.height))
        canvas.paste(frame, (ROW_LABEL_WIDTH + local_idx * frame_width, output_y))


def build_stacked_rows(
    gt_row: Image.Image,
    pred_row: Image.Image,
    frame_width: int,
    model: dict[str, Any],
    window: dict[str, Any],
    metrics: dict[str, Any],
    sample_name: str,
    frames_per_block: int,
) -> Image.Image:
    frame_height = gt_row.height
    block_height = TIMELINE_HEIGHT + frame_height * 2
    block_count = math.ceil(FRAME_COUNT / frames_per_block)
    canvas = Image.new(
        "RGB",
        (ROW_LABEL_WIDTH + frames_per_block * frame_width, HEADER_HEIGHT + block_count * block_height),
        (238, 240, 243),
    )
    draw_header(canvas, model, window, metrics, sample_name)
    draw = ImageDraw.Draw(canvas)
    timeline_font = load_font(17, bold=True)
    row_font = load_font(22, bold=True)

    for block_idx in range(block_count):
        start_frame = block_idx * frames_per_block
        end_frame = min(FRAME_COUNT, start_frame + frames_per_block)
        block_y = HEADER_HEIGHT + block_idx * block_height
        gt_y = block_y + TIMELINE_HEIGHT
        pred_y = gt_y + frame_height

        draw.rectangle(
            (0, block_y, canvas.width - 1, block_y + block_height - 1),
            fill=(255, 255, 255),
            outline=(152, 158, 167),
            width=2,
        )
        draw.rectangle((0, block_y, canvas.width, gt_y), fill=(230, 233, 238))
        for local_idx, frame_idx in enumerate(range(start_frame, end_frame)):
            x = ROW_LABEL_WIDTH + local_idx * frame_width
            draw.text((x + 7, block_y + 6), f"t={frame_idx:02d}", font=timeline_font, fill=(20, 24, 31))
            draw.line((x, gt_y, x, pred_y + frame_height), fill=(190, 194, 201), width=1)

        draw.rectangle((0, gt_y, ROW_LABEL_WIDTH, pred_y), fill=(244, 246, 249))
        draw.rectangle((0, pred_y, ROW_LABEL_WIDTH, pred_y + frame_height), fill=(244, 246, 249))
        draw.text((38, gt_y + frame_height // 2 - 14), "GT", font=row_font, fill=(20, 24, 31))
        draw.text((8, pred_y + frame_height // 2 - 14), "Prediction", font=row_font, fill=(20, 24, 31))

        paste_frame_range(canvas, gt_row, start_frame, end_frame, frame_width, gt_y)
        paste_frame_range(canvas, pred_row, start_frame, end_frame, frame_width, pred_y)
        draw.line((0, pred_y, canvas.width, pred_y), fill=(255, 196, 0), width=4)

    return canvas


def main() -> None:
    args = parse_args()
    if args.frames_per_block < 1:
        raise ValueError("--frames_per_block must be positive")

    manifest = load_json(args.manifest)
    models = manifest["models"]
    for model in models:
        model["eval_dir"] = find_eval_dir(Path(model["eval_dir"]))
        model["windows"] = load_json(model["eval_dir"] / "windows.json")
        model["metrics"] = load_json(model["eval_dir"] / "metrics.json")

    reference_windows = models[0]["windows"]
    for model in models[1:]:
        if not windows_match(reference_windows, model["windows"]):
            raise RuntimeError(f"Fixed-window mismatch for {model['label']}")

    outputs: list[str] = []
    for sample_idx, window in enumerate(reference_windows):
        sample_name = f"sample_{sample_idx:03d}"
        sample_dir = args.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        reference_gt: Image.Image | None = None

        for model_idx, model in enumerate(models, start=1):
            sheet = model["eval_dir"] / "samples" / f"{sample_name}_sheet.png"
            gt_row, pred_row, frame_width, _ = read_sheet_rows(sheet)
            if reference_gt is None:
                reference_gt = gt_row
            elif gt_row.tobytes() != reference_gt.tobytes():
                raise RuntimeError(f"GT pixels differ for {sample_name}: {model['label']}")

            detail = build_stacked_rows(
                gt_row=gt_row,
                pred_row=pred_row,
                frame_width=frame_width,
                model=model,
                window=window,
                metrics=model["metrics"],
                sample_name=sample_name,
                frames_per_block=args.frames_per_block,
            )
            output_path = sample_dir / f"{model_idx:02d}_{model['slug']}_gt_over_pred.png"
            detail.save(output_path)
            outputs.append(str(output_path))

    metadata = {
        "description": "Each temporal block has a GT row above an aligned prediction row.",
        "num_samples": len(reference_windows),
        "num_models": len(models),
        "frame_count": FRAME_COUNT,
        "frames_per_block": args.frames_per_block,
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "stacked_rows_outputs.json").open("w") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
