#!/usr/bin/env python3
"""Build a readable per-task correction workbench for MaskWAM review."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont

from maskwam_view_schema import FINAL_ROLE, VIEW_TRAIN_ROLES


DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")
TRAIN_ROLES = set(VIEW_TRAIN_ROLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--include_aggregate", action="store_true")
    parser.add_argument("--thumb_scale", type=float, default=1.15)
    parser.add_argument("--max_text_width", type=int, default=82)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def safe_slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "task"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = 1 if current else 0
        if current and current_len + extra + len(word) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra + len(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def is_train_task(task: Dict[str, Any]) -> bool:
    return task.get("role") in TRAIN_ROLES and bool(task.get("name"))


def parse_click_frames(task: Dict[str, Any], review_dir: Path) -> List[Path]:
    frames = []
    for raw_path in str(task.get("click_frames", "")).split():
        path = Path(raw_path)
        if path.is_absolute() or path.exists():
            frames.append(path)
            continue
        review_path = review_dir / path
        if review_path.exists():
            frames.append(review_path)
            continue
        if raw_path.startswith("artifacts/"):
            artifact_root = next((parent for parent in review_dir.resolve().parents if parent.name == "artifacts"), None)
            if artifact_root is not None:
                frames.append(artifact_root.parent / path)
                continue
        frames.append(review_path)
    return frames


def frame_index_from_path(path: Path) -> int | None:
    match = re.search(r"_frame_(\d+)\.png$", path.name)
    if not match:
        return None
    return int(match.group(1))


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 6,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def make_task_image(
    task: Dict[str, Any],
    task_idx: int,
    review_dir: Path,
    output_path: Path,
    thumb_scale: float,
    max_text_width: int,
) -> Dict[str, Any]:
    frames = parse_click_frames(task, review_dir)
    missing = [str(path) for path in frames if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing click frame(s): {missing}")

    loaded = [Image.open(path).convert("RGB") for path in frames]
    thumb_w = int(loaded[0].width * thumb_scale)
    thumb_h = int(loaded[0].height * thumb_scale)
    thumbs = [image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS) for image in loaded]

    gap = 16
    margin = 28
    header_h = 280
    footer_h = 34
    width = max(1220, margin * 2 + len(thumbs) * thumb_w + (len(thumbs) - 1) * gap)
    height = header_h + thumb_h + footer_h
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(30, bold=True)
    meta_font = load_font(21, bold=True)
    text_font = load_font(19)
    small_font = load_font(16)

    priority = task.get("priority", "")
    sample = task.get("sample_name", f"sample_{int(task.get('sample_id', 0)):03d}")
    role = task.get("role", "")
    name = task.get("name", "")
    recommended = int(task.get("recommended_frame_index", 0))

    draw.rounded_rectangle((0, 0, width, 72), radius=0, fill=(31, 45, 61))
    title = f"#{task_idx:02d} {priority.upper()}  {sample}  {role}: {name}"
    draw.text((margin, 20), title, font=title_font, fill=(255, 255, 255))

    y = 92
    y = draw_text_block(
        draw,
        (margin, y),
        wrap_text(f"Task: {task.get('task_text', '')}", max_text_width),
        text_font,
        (26, 32, 40),
        line_gap=5,
    )
    y += 2
    y = draw_text_block(
        draw,
        (margin, y),
        wrap_text(f"Action: {task.get('suggested_action', '')}", max_text_width),
        text_font,
        (52, 64, 78),
        line_gap=5,
    )
    reason = f"Recommended: F{recommended} ({task.get('recommended_frame_reason', '')})"
    draw_text_block(draw, (margin, y + 2), wrap_text(reason, max_text_width), small_font, (92, 73, 23), line_gap=4)

    start_x = margin
    frame_y = header_h
    for idx, (path, thumb) in enumerate(zip(frames, thumbs)):
        frame_idx = frame_index_from_path(path)
        x = start_x + idx * (thumb_w + gap)
        label = f"F{frame_idx}" if frame_idx is not None else f"frame {idx}"
        is_recommended = frame_idx == recommended
        border = (214, 63, 50) if is_recommended else (160, 169, 181)
        border_w = 6 if is_recommended else 2
        canvas.paste(thumb, (x, frame_y))
        for offset in range(border_w):
            draw.rectangle(
                (x - offset, frame_y - offset, x + thumb_w + offset - 1, frame_y + thumb_h + offset - 1),
                outline=border,
            )
        tag_fill = (214, 63, 50) if is_recommended else (31, 45, 61)
        draw.rounded_rectangle((x, frame_y - 30, x + 98, frame_y - 5), radius=5, fill=tag_fill)
        draw.text((x + 9, frame_y - 28), label + ("  use" if is_recommended else ""), font=small_font, fill=(255, 255, 255))

    draw.text(
        (margin, height - 28),
        "Use the review HTML to add point/box corrections; this image is only a human-readable guide.",
        font=small_font,
        fill=(84, 94, 108),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "task_idx": task_idx,
        "priority": priority,
        "sample_id": task.get("sample_id"),
        "sample_name": sample,
        "role": role,
        "name": name,
        "recommended_frame_index": recommended,
        "image": str(output_path),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_idx",
        "priority",
        "sample_id",
        "sample_name",
        "role",
        "name",
        "recommended_frame_index",
        "image",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_index(output_dir: Path, rows: Sequence[Dict[str, Any]], review_dir: Path) -> None:
    cards = []
    for row in rows:
        rel = Path(row["image"]).resolve().relative_to(output_dir.resolve())
        title = f"#{row['task_idx']:02d} {row['priority']} {row['sample_name']} {row['role']}:{row['name']} @F{row['recommended_frame_index']}"
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(title)}</h2>"
            f"<a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}' alt='{html.escape(title)}'></a>"
            "</section>"
        )
    review_rel = Path("../index.html")
    payload = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MaskWAM Correction Workbench</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ position: sticky; top: 0; background: #1f2d3d; color: white; padding: 18px 24px; z-index: 2; }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; }}
    header a {{ color: #b9dcff; }}
    main {{ padding: 20px 24px 40px; }}
    .card {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; margin: 0 0 24px; padding: 14px; }}
    .card h2 {{ font-size: 18px; margin: 0 0 12px; }}
    img {{ width: 100%; max-width: 1380px; border: 1px solid #d5dbe5; display: block; }}
  </style>
</head>
<body>
  <header>
    <h1>MaskWAM Correction Workbench</h1>
    <div>{len(rows)} actionable tasks. Open <a href="{html.escape(str(review_rel))}">review HTML</a> to enter point/box corrections.</div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    tasks_path = Path(args.tasks) if args.tasks else review_dir / "correction_tasks.json"
    output_dir = Path(args.output_dir) if args.output_dir else review_dir / "correction_workbench"
    task_dir = output_dir / "tasks"
    tasks_payload = load_json(tasks_path)

    selected = []
    for task in tasks_payload.get("tasks", []):
        if is_train_task(task) or (args.include_aggregate and task.get("role") == FINAL_ROLE):
            selected.append(task)

    rows = []
    for idx, task in enumerate(selected, start=1):
        filename = "{idx:02d}_{sample}_{priority}_{role}_{name}.png".format(
            idx=idx,
            sample=safe_slug(str(task.get("sample_name", ""))),
            priority=safe_slug(str(task.get("priority", ""))),
            role=safe_slug(str(task.get("role", ""))),
            name=safe_slug(str(task.get("name", ""))),
        )
        rows.append(
            make_task_image(
                task=task,
                task_idx=idx,
                review_dir=review_dir,
                output_path=task_dir / filename,
                thumb_scale=args.thumb_scale,
                max_text_width=args.max_text_width,
            )
        )

    write_json(output_dir / "workbench_manifest.json", {"tasks": rows})
    write_csv(output_dir / "workbench_manifest.csv", rows)
    write_index(output_dir, rows, review_dir)
    print(f"tasks={len(rows)}")
    print(f"output_dir={output_dir}")
    print(f"index={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
