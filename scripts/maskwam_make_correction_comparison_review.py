#!/usr/bin/env python3
"""Create first-pass vs corrected MaskWAM mask comparison sheets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from maskwam_make_annotation_review import ROLE_COLORS, bordered, font, load_mask, load_rgb, overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--before_annotation_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass_qa16")
    parser.add_argument("--after_annotation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--review_frames", nargs="+", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def rel(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path.resolve(), base.resolve().parent))


def select_samples(samples: Sequence[Dict[str, Any]], sample_ids: Sequence[int] | None) -> List[Dict[str, Any]]:
    if sample_ids is None:
        return list(samples)
    wanted = {int(value) for value in sample_ids}
    return [sample for sample in samples if int(sample["sample_id"]) in wanted]


def role_overlay(sample_dir: Path, frame_idx: int) -> Image.Image:
    rgb = load_rgb(sample_dir, frame_idx)
    shape = rgb.shape[:2]
    out = rgb
    out = overlay(out, load_mask(sample_dir, "object", frame_idx, shape), ROLE_COLORS["object"], 0.42)
    out = overlay(out, load_mask(sample_dir, "target", frame_idx, shape), ROLE_COLORS["target"], 0.42)
    out = overlay(out, load_mask(sample_dir, "robot", frame_idx, shape), ROLE_COLORS["robot"], 0.38)
    return Image.fromarray(out)


def final_overlay(sample_dir: Path, frame_idx: int) -> Image.Image:
    rgb = load_rgb(sample_dir, frame_idx)
    shape = rgb.shape[:2]
    out = overlay(rgb, load_mask(sample_dir, "final_train_mask", frame_idx, shape), ROLE_COLORS["final_train_mask"], 0.50)
    return Image.fromarray(out)


def mask_iou(before_dir: Path, after_dir: Path, role: str, frame_idx: int, shape: Tuple[int, int]) -> float | None:
    before = load_mask(before_dir, role, frame_idx, shape)
    after = load_mask(after_dir, role, frame_idx, shape)
    union = np.logical_or(before, after).sum()
    if union == 0:
        return None
    return float(np.logical_and(before, after).sum() / union)


def changed_overlay(before_dir: Path, after_dir: Path, frame_idx: int) -> Image.Image:
    rgb = load_rgb(after_dir, frame_idx)
    shape = rgb.shape[:2]
    before = load_mask(before_dir, "final_train_mask", frame_idx, shape)
    after = load_mask(after_dir, "final_train_mask", frame_idx, shape)
    out = rgb.astype(np.float32).copy()
    removed = np.logical_and(before, np.logical_not(after))
    added = np.logical_and(after, np.logical_not(before))
    same = np.logical_and(before, after)
    out[same] = 0.70 * out[same] + 0.30 * np.asarray([235, 40, 40], dtype=np.float32)
    out[removed] = 0.45 * out[removed] + 0.55 * np.asarray([255, 180, 0], dtype=np.float32)
    out[added] = 0.45 * out[added] + 0.55 * np.asarray([0, 170, 90], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def summarize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_zero": meta.get("final_train_mask", {}).get("zero_frame_count", ""),
        "final_mean": meta.get("final_train_mask", {}).get("mean_area_ratio", ""),
        "object_zero": meta.get("role_summaries", {}).get("object", {}).get("zero_frame_count", ""),
        "object_mean": meta.get("role_summaries", {}).get("object", {}).get("mean_area_ratio", ""),
        "target_zero": meta.get("role_summaries", {}).get("target", {}).get("zero_frame_count", ""),
        "target_mean": meta.get("role_summaries", {}).get("target", {}).get("mean_area_ratio", ""),
        "robot_zero": meta.get("role_summaries", {}).get("robot", {}).get("zero_frame_count", ""),
        "robot_mean": meta.get("role_summaries", {}).get("robot", {}).get("mean_area_ratio", ""),
    }


def make_comparison_sheet(
    sample: Dict[str, Any],
    before_dir: Path,
    after_dir: Path,
    output_path: Path,
    review_frames: Sequence[int],
) -> Dict[str, Any]:
    sample_name = sample["sample_name"]
    before_sample = before_dir / sample_name
    after_sample = after_dir / sample_name
    before_meta = load_json(before_sample / "annotation_metadata.json")
    after_meta = load_json(after_sample / "annotation_metadata.json")

    label_font = font(14)
    body_font = font(15)
    title_font = font(20)
    frame_columns = []
    frame_ious: List[float] = []

    for frame_idx in review_frames:
        rgb = Image.fromarray(load_rgb(after_sample, frame_idx))
        shape = (rgb.height, rgb.width)
        iou = mask_iou(before_sample, after_sample, "final_train_mask", frame_idx, shape)
        if iou is not None:
            frame_ious.append(iou)
        cells = [
            bordered(rgb, f"F{frame_idx:02d} RGB", (80, 86, 96), label_font),
            bordered(role_overlay(before_sample, frame_idx), "before roles", (35, 92, 150), label_font),
            bordered(role_overlay(after_sample, frame_idx), "after roles", (20, 124, 86), label_font),
            bordered(final_overlay(before_sample, frame_idx), "before final", (180, 45, 45), label_font),
            bordered(final_overlay(after_sample, frame_idx), "after final", (38, 128, 90), label_font),
            bordered(changed_overlay(before_sample, after_sample, frame_idx), "change +green -yellow", (96, 86, 45), label_font),
        ]
        col_w = max(cell.width for cell in cells)
        col_h = sum(cell.height for cell in cells)
        col = Image.new("RGB", (col_w, col_h), (255, 255, 255))
        y = 0
        for cell in cells:
            col.paste(cell, (0, y))
            y += cell.height
        frame_columns.append(col)

    gutter = 10
    header_h = 150
    legend_h = 36
    width = sum(frame.width for frame in frame_columns) + gutter * max(0, len(frame_columns) - 1)
    height = header_h + max(frame.height for frame in frame_columns) + legend_h
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    before_summary = summarize_metadata(before_meta)
    after_summary = summarize_metadata(after_meta)
    mean_iou = float(np.mean(frame_ious)) if frame_ious else None

    draw.text((8, 8), f"{sample_name} correction comparison", fill=(18, 24, 32), font=title_font)
    draw.text((8, 38), sample.get("task_text", ""), fill=(42, 48, 56), font=body_font)
    draw.text(
        (8, 66),
        f"before final zero/mean: {before_summary['final_zero']} / {before_summary['final_mean']}",
        fill=(88, 96, 105),
        font=body_font,
    )
    draw.text(
        (8, 92),
        f"after final zero/mean: {after_summary['final_zero']} / {after_summary['final_mean']} | final IoU on shown frames: {mean_iou if mean_iou is not None else 'n/a'}",
        fill=(88, 96, 105),
        font=body_font,
    )
    draw.text(
        (8, 120),
        "green = added final mask, yellow = removed final mask, red = unchanged final mask",
        fill=(88, 96, 105),
        font=body_font,
    )

    x = 0
    for col in frame_columns:
        sheet.paste(col, (x, header_h))
        x += col.width + gutter
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)

    row = {
        "sample_id": int(sample["sample_id"]),
        "sample_name": sample_name,
        "task_text": sample.get("task_text", ""),
        "shown_final_iou_mean": mean_iou,
        **{f"before_{key}": value for key, value in before_summary.items()},
        **{f"after_{key}": value for key, value in after_summary.items()},
        "sheet": str(output_path),
    }
    return row


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_index(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        sheet = Path(row["sheet"])
        href = rel(sheet, path)
        cards.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>{html.escape(row['sample_name'])}</h2>",
                    f"<p>{html.escape(row.get('task_text', ''))}</p>",
                    f"<p>before final zero/mean: <code>{row['before_final_zero']}</code> / <code>{row['before_final_mean']}</code></p>",
                    f"<p>after final zero/mean: <code>{row['after_final_zero']}</code> / <code>{row['after_final_mean']}</code></p>",
                    f"<a href='{href}'><img src='{href}' alt='{html.escape(row['sample_name'])}'></a>",
                    "</section>",
                ]
            )
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MaskWAM Correction Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f6f7f9; }}
    section {{ background: #fff; border: 1px solid #dde1e7; padding: 14px; border-radius: 8px; margin-bottom: 18px; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 20px; margin: 0 0 4px; }}
    p {{ color: #52606d; margin: 4px 0; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #d7dce2; }}
    code {{ background: #eef1f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>MaskWAM Correction Comparison</h1>
  <p>Rows compare first-pass masks and corrected masks. In the change row, green means added final mask, yellow means removed final mask, red means unchanged final mask.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    samples = select_samples(manifest["samples"], args.sample_ids)
    before_dir = Path(args.before_annotation_dir)
    after_dir = Path(args.after_annotation_dir)
    output_dir = Path(args.output_dir)
    rows = []
    skipped = []
    for sample in samples:
        sample_name = sample["sample_name"]
        before_meta = before_dir / sample_name / "annotation_metadata.json"
        after_meta = after_dir / sample_name / "annotation_metadata.json"
        if not before_meta.exists() or not after_meta.exists():
            skipped.append(sample_name)
            continue
        review_frames = args.review_frames or sample.get("review_frame_indices", [0, 4, 8, 12, 16])
        rows.append(
            make_comparison_sheet(
                sample,
                before_dir,
                after_dir,
                output_dir / "sheets" / f"{sample_name}_comparison.png",
                review_frames,
            )
        )

    write_summary_csv(output_dir / "comparison_summary.csv", rows)
    write_index(output_dir / "index.html", rows)
    payload = {
        "schema_version": "maskwam_correction_comparison_review_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "before_annotation_dir": str(before_dir.resolve()),
        "after_annotation_dir": str(after_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "sample_count": len(rows),
        "skipped_count": len(skipped),
        "skipped_samples": skipped,
        "rows": rows,
    }
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(output_dir / "index.html")
    print(f"samples={len(rows)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
