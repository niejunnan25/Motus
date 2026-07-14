#!/usr/bin/env python3
"""Render one inspectable 53-frame GT/prediction grid per VGM model and case."""

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
TILE_LABEL_HEIGHT = 34
HEADER_HEIGHT = 230


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=11)
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


def build_tiled_detail(
    gt_row: Image.Image,
    pred_row: Image.Image,
    frame_width: int,
    model: dict[str, Any],
    window: dict[str, Any],
    metrics: dict[str, Any],
    sample_name: str,
    columns: int,
) -> Image.Image:
    frame_height = gt_row.height
    tile_width = frame_width * 2
    tile_height = TILE_LABEL_HEIGHT + frame_height
    rows = math.ceil(FRAME_COUNT / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, HEADER_HEIGHT + rows * tile_height),
        (238, 240, 243),
    )
    draw_header(canvas, model, window, metrics, sample_name)
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(18, bold=True)

    for frame_idx in range(FRAME_COUNT):
        row_idx, column_idx = divmod(frame_idx, columns)
        x = column_idx * tile_width
        y = HEADER_HEIGHT + row_idx * tile_height
        gt = gt_row.crop((frame_idx * frame_width, 0, (frame_idx + 1) * frame_width, frame_height))
        pred = pred_row.crop((frame_idx * frame_width, 0, (frame_idx + 1) * frame_width, frame_height))

        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), fill=(255, 255, 255), outline=(160, 165, 173), width=2)
        draw.text((x + 8, y + 7), f"t={frame_idx:02d}", font=label_font, fill=(20, 24, 31))
        draw.text((x + frame_width - 42, y + 7), "GT", font=label_font, fill=(20, 24, 31))
        draw.text((x + frame_width + 8, y + 7), "Prediction", font=label_font, fill=(20, 24, 31))
        canvas.paste(gt, (x, y + TILE_LABEL_HEIGHT))
        canvas.paste(pred, (x + frame_width, y + TILE_LABEL_HEIGHT))
        draw.line(
            (x + frame_width, y + TILE_LABEL_HEIGHT, x + frame_width, y + tile_height),
            fill=(255, 215, 0),
            width=3,
        )
    return canvas


def main() -> None:
    args = parse_args()
    if args.columns < 1:
        raise ValueError("--columns must be positive")

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

            detail = build_tiled_detail(
                gt_row=gt_row,
                pred_row=pred_row,
                frame_width=frame_width,
                model=model,
                window=window,
                metrics=model["metrics"],
                sample_name=sample_name,
                columns=args.columns,
            )
            output_path = sample_dir / f"{model_idx:02d}_{model['slug']}_tiled.png"
            detail.save(output_path)
            outputs.append(str(output_path))

    metadata = {
        "description": "One model per image; each tile is GT (left) versus prediction (right).",
        "num_samples": len(reference_windows),
        "num_models": len(models),
        "frame_count": FRAME_COUNT,
        "columns": args.columns,
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tiled_outputs.json").open("w") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
