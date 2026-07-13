#!/usr/bin/env python3
"""Build readable RGB, role-mask, and full-frame details for the 8-way VGM eval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import textwrap
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None
HEADER_HEIGHT = 230
TIMELINE_HEIGHT = 32
ROW_LABEL_WIDTH = 130
WINDOW_KEYS = (
    "episode_name",
    "condition_idx",
    "frame_indices",
    "total_frames",
    "task_index",
    "task_text",
    "language_caption",
    "language_caption_version",
    "language_embedding_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frames_per_block", type=int, default=18)
    parser.add_argument("--full_frames_per_block", type=int, default=9)
    parser.add_argument("--render_workers", type=int, default=4)
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


def read_sheet_rows(path: Path, frame_count: int) -> tuple[Image.Image, Image.Image, int]:
    sheet = Image.open(path).convert("RGB")
    if sheet.width % frame_count:
        raise RuntimeError(f"Expected {frame_count} frames in {path}, got width={sheet.width}")
    frame_width = sheet.width // frame_count
    label_height = sheet.height - 2 * frame_width
    if label_height < 0:
        raise RuntimeError(f"Unexpected sheet dimensions for square four-grid frames: {path}: {sheet.size}")
    gt = sheet.crop((0, label_height, sheet.width, label_height + frame_width))
    pred = sheet.crop((0, label_height + frame_width, sheet.width, sheet.height))
    return gt, pred, frame_width


def extract_panel(row: Image.Image, frame_width: int, frame_count: int, panel: str) -> tuple[Image.Image, int]:
    if panel == "full":
        return row, frame_width
    half_width = frame_width // 2
    if frame_width % 2:
        raise RuntimeError(f"Four-grid frame width must be even, got {frame_width}")
    source_x = 0 if panel == "rgb" else half_width
    output = Image.new("RGB", (frame_count * half_width, row.height), (255, 255, 255))
    for frame_idx in range(frame_count):
        frame_x = frame_idx * frame_width
        crop = row.crop((frame_x + source_x, 0, frame_x + source_x + half_width, row.height))
        output.paste(crop, (frame_idx * half_width, 0))
    return output, half_width


def draw_header(
    canvas: Image.Image,
    model: dict[str, Any],
    window: dict[str, Any],
    sample_metrics: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    manifest: dict[str, Any],
    sample_name: str,
    panel: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    regular_font = load_font(18)
    bold_font = load_font(18, bold=True)
    caption = window.get("language_caption") or window.get("task_text") or "<missing>"
    rgb_mse = float(sample_metrics.get("rgb_generated_mse", float("nan")))
    rgb_psnr = float(sample_metrics.get("rgb_generated_psnr", float("nan")))
    interaction_mse = float(sample_metrics.get("rgb_generated_interaction_mse", float("nan")))
    foreground_iou = float(sample_metrics.get("role_mask_foreground_iou", float("nan")))
    macro_iou = float(sample_metrics.get("role_mask_macro_iou", float("nan")))
    total_loss = metric(aggregate_metrics, "loss", "total_loss")
    rgb_loss = metric(aggregate_metrics, "loss", "rgb_loss")
    role_loss = metric(aggregate_metrics, "loss", "role_mask_loss")
    mask_metric_text = f"foreground IoU {foreground_iou:.4f}"
    if math.isfinite(macro_iou):
        mask_metric_text += f" | macro IoU {macro_iou:.4f}"

    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill=(247, 248, 250))
    draw.text((20, 12), f"{sample_name} | {model['label']} | panel={panel}", font=title_font, fill=(20, 24, 31))
    draw.text(
        (20, 50),
        f"episode: {window.get('episode_name')} | condition_idx={window.get('condition_idx')} | "
        f"frames={manifest['frame_count']} | seed={manifest['seed']} | steps={manifest['num_inference_steps']}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text((20, 77), model.get("condition", ""), font=regular_font, fill=(20, 24, 31))
    draw.text(
        (20, 104),
        f"sample RGB MSE {rgb_mse:.5f} | RGB PSNR {rgb_psnr:.2f} | interaction MSE {interaction_mse:.5f} | "
        f"{mask_metric_text}",
        font=regular_font,
        fill=(20, 24, 31),
    )
    draw.text(
        (20, 131),
        f"eval loss (all windows) total {total_loss:.5f} | rgb {rgb_loss:.5f} | role mask {role_loss:.5f}",
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
    sample_metrics: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    manifest: dict[str, Any],
    sample_name: str,
    panel: str,
    frames_per_block: int,
    frame_count: int,
) -> Image.Image:
    frame_height = gt_row.height
    block_count = math.ceil(frame_count / frames_per_block)
    block_height = TIMELINE_HEIGHT + frame_height * 2
    canvas = Image.new(
        "RGB",
        (ROW_LABEL_WIDTH + frames_per_block * frame_width, HEADER_HEIGHT + block_count * block_height),
        (238, 240, 243),
    )
    draw_header(canvas, model, window, sample_metrics, aggregate_metrics, manifest, sample_name, panel)
    draw = ImageDraw.Draw(canvas)
    timeline_font = load_font(17, bold=True)
    row_font = load_font(21, bold=True)

    for block_idx in range(block_count):
        start = block_idx * frames_per_block
        end = min(frame_count, start + frames_per_block)
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
    frame_count = int(manifest["frame_count"])
    if frame_count <= 2:
        raise RuntimeError(f"frame_count must include two endpoints and generated frames, got {frame_count}")
    if not models:
        raise RuntimeError("Manifest contains no models")
    slugs = [model["slug"] for model in models]
    if len(slugs) != len(set(slugs)):
        raise RuntimeError(f"Manifest model slugs must be unique: {slugs}")
    for model in models:
        model["eval_dir"] = find_eval_dir(Path(model["eval_dir"]))
        model["windows"] = load_json(model["eval_dir"] / "windows.json")
        model["metrics"] = load_json(model["eval_dir"] / "metrics.json")
        sample_metrics_path = model["eval_dir"] / "sample_metrics.json"
        if not sample_metrics_path.is_file():
            raise RuntimeError(
                f"Missing per-sample metrics for {model['label']}: {sample_metrics_path}. "
                "Re-run evaluation with the current evaluator before rendering details."
            )
        model["sample_metrics"] = load_json(sample_metrics_path)

    reference_windows = models[0]["windows"]
    if len(reference_windows) != manifest["num_samples"]:
        raise RuntimeError(f"Expected {manifest['num_samples']} windows, found {len(reference_windows)}")
    mismatched = [model["label"] for model in models[1:] if not windows_match(reference_windows, model["windows"])]
    if mismatched:
        raise RuntimeError(f"Fixed-window mismatch: {mismatched}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
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

    sample_rows: list[dict[str, Any]] = []
    for sample_idx, window in enumerate(reference_windows):
        sample_name = f"sample_{sample_idx:03d}"
        for model in models:
            values = model["sample_metrics"].get(sample_name)
            if values is None:
                raise RuntimeError(f"Missing metrics for {sample_name}: {model['label']}")
            sample_rows.append(
                {
                    "sample": sample_name,
                    "episode_name": window.get("episode_name"),
                    "task_index": window.get("task_index"),
                    "label": model["label"],
                    "slug": model["slug"],
                    **values,
                }
            )
    sample_fields = sorted({key for row in sample_rows for key in row})
    with (args.output_dir / "sample_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)
    (args.output_dir / "sample_metrics.json").write_text(json.dumps(sample_rows, indent=2))

    outputs: list[str] = []
    render_workers = max(1, int(args.render_workers))
    executor = ThreadPoolExecutor(max_workers=render_workers)
    pending: set[Future[None]] = set()

    def submit_detail(image: Image.Image, output: Path) -> None:
        nonlocal pending
        pending.add(executor.submit(image.save, output))
        if len(pending) >= render_workers * 2:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()

    for sample_idx, window in enumerate(reference_windows):
        sample_name = f"sample_{sample_idx:03d}"
        sample_dir = args.output_dir / "by_sample" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        reference_rgb: bytes | None = None
        reference_masks: dict[str, bytes] = {}

        for model_idx, model in enumerate(models, start=1):
            sheet = model["eval_dir"] / "samples" / f"{sample_name}_sheet.png"
            gt_full, pred_full, full_width = read_sheet_rows(sheet, frame_count)
            gt_rgb, rgb_width = extract_panel(gt_full, full_width, frame_count, "rgb")
            pred_rgb, _ = extract_panel(pred_full, full_width, frame_count, "rgb")
            gt_mask, mask_width = extract_panel(gt_full, full_width, frame_count, "mask")
            pred_mask, _ = extract_panel(pred_full, full_width, frame_count, "mask")

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
            if sample_name not in model["sample_metrics"]:
                raise RuntimeError(f"Missing metrics for {sample_name}: {model['label']}")

            panels = [
                ("rgb", gt_rgb, pred_rgb, rgb_width, args.frames_per_block),
                ("mask", gt_mask, pred_mask, mask_width, args.frames_per_block),
                ("full", gt_full, pred_full, full_width, args.full_frames_per_block),
            ]
            for panel, gt_row, pred_row, frame_width, block_size in panels:
                detail = build_detail(
                    gt_row,
                    pred_row,
                    frame_width,
                    model,
                    window,
                    model["sample_metrics"][sample_name],
                    model["metrics"],
                    manifest,
                    sample_name,
                    panel,
                    block_size,
                    frame_count,
                )
                output = sample_dir / f"{model_idx:02d}_{model['slug']}_{panel}_gt_over_pred.png"
                submit_detail(detail, output)
                outputs.append(str(output))

    for future in pending:
        future.result()
    executor.shutdown()

    metadata = {
        "description": "Per-model horizontal GT-over-pred details for RGB, role mask, and full four-grid frames.",
        "num_samples": len(reference_windows),
        "num_models": len(models),
        "frame_count": frame_count,
        "expected_pngs": len(reference_windows) * len(models) * 3,
        "actual_pngs": len(outputs),
        "outputs": outputs,
    }
    (args.output_dir / "detail_outputs.json").write_text(json.dumps(metadata, indent=2))
    if metadata["actual_pngs"] != metadata["expected_pngs"]:
        raise RuntimeError(f"Output count mismatch: {metadata}")


if __name__ == "__main__":
    main()
