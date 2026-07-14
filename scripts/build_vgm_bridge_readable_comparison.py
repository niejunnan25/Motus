#!/usr/bin/env python3
"""Build reproducible full-trajectory VGM comparison sheets from eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WINDOW_KEYS = ("episode_name", "condition_idx", "frame_indices", "total_frames")
HEADER_HEIGHT = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label_width", type=int, default=1200)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--copy_videos", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def find_eval_dir(path: Path) -> Path:
    if (path / "metrics.json").is_file() and (path / "windows.json").is_file():
        return path
    candidates = sorted(path.glob("*/step_*"))
    candidates = [
        candidate
        for candidate in candidates
        if (candidate / "metrics.json").is_file() and (candidate / "windows.json").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one completed eval under {path}, found {candidates}")
    return candidates[0]


def windows_match(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> bool:
    if len(reference) != len(current):
        return False
    return all(
        all(left.get(key) == right.get(key) for key in WINDOW_KEYS)
        for left, right in zip(reference, current)
    )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    text = " ".join((text or "<missing>").split())
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or ["<missing>"]


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> int:
    x, y = xy
    line_height = int(font.getbbox("Ag")[3] * 1.15)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + spacing
    return y


def metric_value(metrics: dict[str, Any], group: str, key: str) -> float:
    value = metrics.get(group, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def model_label_lines(model: dict[str, Any], caption: str, metrics: dict[str, Any]) -> list[tuple[str, bool]]:
    mse = metric_value(metrics, "samples", "generated_mse")
    psnr = metric_value(metrics, "samples", "generated_psnr")
    imse = metric_value(metrics, "samples", "generated_interaction_mse")
    loss = metric_value(metrics, "loss", "total_loss")
    return [
        (model["label"], True),
        (model.get("condition", ""), False),
        (f"MSE {mse:.5f} | PSNR {psnr:.2f} | interaction MSE {imse:.5f} | loss {loss:.5f}", False),
        ("language_caption:", True),
        (caption, False),
    ]


def draw_label_panel(
    canvas: Image.Image,
    y: int,
    height: int,
    width: int,
    blocks: list[tuple[str, bool]],
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, y, width, y + height), fill=(247, 248, 250), outline=(205, 209, 216), width=2)
    regular = load_font(23)
    bold = load_font(27, bold=True)
    cursor = y + 18
    for text, is_bold in blocks:
        if not text:
            continue
        font = bold if is_bold else regular
        chars = 68 if is_bold else 84
        lines = wrap_text(text, chars)
        cursor = draw_multiline(draw, (22, cursor), lines, font, (20, 24, 31), 2)
        cursor += 5
        if cursor > y + height - 20:
            break


def read_sheet_rows(path: Path) -> tuple[Image.Image, Image.Image, int, int]:
    sheet = Image.open(path).convert("RGB")
    if sheet.height <= HEADER_HEIGHT or (sheet.height - HEADER_HEIGHT) % 2:
        raise RuntimeError(f"Unexpected contact-sheet dimensions: {path}: {sheet.size}")
    row_height = (sheet.height - HEADER_HEIGHT) // 2
    gt = sheet.crop((0, HEADER_HEIGHT, sheet.width, HEADER_HEIGHT + row_height))
    pred = sheet.crop((0, HEADER_HEIGHT + row_height, sheet.width, sheet.height))
    if sheet.width % 53:
        raise RuntimeError(f"Expected 53 equal-width frames in {path}, got width={sheet.width}")
    return gt, pred, sheet.width // 53, row_height


def save_frame_row(row: Image.Image, frame_width: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(53):
        frame = row.crop((frame_idx * frame_width, 0, (frame_idx + 1) * frame_width, row.height))
        frame.save(output_dir / f"frame_{frame_idx:03d}.png")


def flatten_metrics(model: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": model["label"],
        "slug": model["slug"],
        "condition": model.get("condition", ""),
        "eval_dir": str(model["eval_dir"]),
    }
    for group in ("loss", "samples"):
        for key, value in metrics.get(group, {}).items():
            if isinstance(value, (int, float)):
                row[f"{group}.{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    models = manifest["models"]
    if not models:
        raise RuntimeError("Manifest contains no models")

    for model in models:
        model["eval_dir"] = find_eval_dir(Path(model["eval_dir"]))
        model["windows"] = load_json(model["eval_dir"] / "windows.json")
        model["metrics"] = load_json(model["eval_dir"] / "metrics.json")

    reference_windows = models[0]["windows"]
    mismatched = [model["label"] for model in models[1:] if not windows_match(reference_windows, model["windows"])]
    if mismatched:
        raise RuntimeError(f"Fixed-window mismatch for: {mismatched}")
    if len(reference_windows) != 16:
        raise RuntimeError(f"Expected 16 fixed windows, found {len(reference_windows)}")

    output_dir = args.output_dir
    by_sample_dir = output_dir / "by_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_sample_dir.mkdir(parents=True, exist_ok=True)

    rows = [flatten_metrics(model, model["metrics"]) for model in models]
    metric_fields = sorted({key for row in rows for key in row})
    with (output_dir / "metrics_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics_summary.json").open("w") as file:
        json.dump(rows, file, indent=2)
    with (output_dir / "windows.json").open("w") as file:
        json.dump(reference_windows, file, indent=2)
    (output_dir / "windows_match.txt").write_text("same_fixed_windows=true\n")

    global_outputs: list[str] = []
    detail_outputs: list[str] = []
    for sample_idx, window in enumerate(reference_windows):
        sample_name = f"sample_{sample_idx:03d}"
        caption = window.get("language_caption") or window.get("task_text") or "<missing>"
        gt_rows: list[Image.Image] = []
        pred_rows: list[Image.Image] = []
        frame_width = 0
        row_height = 0
        for model in models:
            sheet = model["eval_dir"] / "samples" / f"{sample_name}_sheet.png"
            gt, pred, frame_width, row_height = read_sheet_rows(sheet)
            gt_rows.append(gt)
            pred_rows.append(pred)
        if any(gt.tobytes() != gt_rows[0].tobytes() for gt in gt_rows[1:]):
            raise RuntimeError(f"GT pixels differ across models for {sample_name}")

        image_width = gt_rows[0].width
        canvas = Image.new(
            "RGB",
            (args.label_width + image_width, HEADER_HEIGHT + (len(models) + 1) * row_height),
            (255, 255, 255),
        )
        title = (
            f"{sample_name} | {window.get('episode_name')} | condition={window.get('condition_idx')} | "
            "53 frames | seed=50 | inference_steps=50"
        )
        ImageDraw.Draw(canvas).text((12, 4), title, font=load_font(17, bold=True), fill=(20, 24, 31))
        draw_label_panel(
            canvas,
            HEADER_HEIGHT,
            row_height,
            args.label_width,
            [("GT", True), (f"episode: {window.get('episode_name')}", False), ("language_caption:", True), (caption, False)],
        )
        canvas.paste(gt_rows[0], (args.label_width, HEADER_HEIGHT))

        sample_detail_dir = by_sample_dir / sample_name
        sample_detail_dir.mkdir(parents=True, exist_ok=True)
        for model_idx, model in enumerate(models):
            y = HEADER_HEIGHT + (model_idx + 1) * row_height
            blocks = model_label_lines(model, caption, model["metrics"])
            draw_label_panel(canvas, y, row_height, args.label_width, blocks)
            canvas.paste(pred_rows[model_idx], (args.label_width, y))

            detail = Image.new("RGB", (args.label_width + image_width, HEADER_HEIGHT + 2 * row_height), (255, 255, 255))
            ImageDraw.Draw(detail).text((12, 4), title, font=load_font(17, bold=True), fill=(20, 24, 31))
            draw_label_panel(
                detail,
                HEADER_HEIGHT,
                row_height,
                args.label_width,
                [("GT", True), (f"episode: {window.get('episode_name')}", False), ("language_caption:", True), (caption, False)],
            )
            detail.paste(gt_rows[0], (args.label_width, HEADER_HEIGHT))
            draw_label_panel(detail, HEADER_HEIGHT + row_height, row_height, args.label_width, blocks)
            detail.paste(pred_rows[model_idx], (args.label_width, HEADER_HEIGHT + row_height))
            detail_path = sample_detail_dir / f"{model['slug']}_readable.png"
            detail.save(detail_path)
            detail_outputs.append(str(detail_path))

            if args.save_frames:
                save_frame_row(
                    pred_rows[model_idx],
                    frame_width,
                    output_dir / "frames" / model["slug"] / sample_name,
                )
            if args.copy_videos:
                video_src = model["eval_dir"] / "samples" / f"{sample_name}_gt_pred.mp4"
                video_dst = output_dir / "videos" / model["slug"] / video_src.name
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_src, video_dst)

        if args.save_frames:
            save_frame_row(gt_rows[0], frame_width, output_dir / "frames" / "gt" / sample_name)

        global_path = output_dir / f"comparison_{sample_name}_full_readable.png"
        canvas.save(global_path)
        global_outputs.append(str(global_path))

    metadata = {
        "manifest": str(args.manifest),
        "same_fixed_windows": True,
        "num_samples": len(reference_windows),
        "frame_count": 53,
        "seed": 50,
        "num_inference_steps": 50,
        "global_outputs": global_outputs,
        "detail_outputs": detail_outputs,
        "models": [
            {
                "label": model["label"],
                "slug": model["slug"],
                "condition": model.get("condition", ""),
                "eval_dir": str(model["eval_dir"]),
            }
            for model in models
        ],
    }
    with (output_dir / "readable_outputs.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    (output_dir / "README.txt").write_text(
        "Fixed 16 windows; identical frame indices; seed=50; 50 inference steps; all 53 frames shown.\n"
        "GT is the first row. The remaining rows follow manifest order. Actual language_caption is shown in labels.\n"
    )


if __name__ == "__main__":
    main()
