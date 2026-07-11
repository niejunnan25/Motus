#!/usr/bin/env python3
"""Build human-review SAM3 mask galleries from existing 17F debug sheets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


LABEL_W = 380
HEADER_H = 128
CELL_W = 224
CELL_H = 448
ROW_GAP = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa_dirs",
        nargs="+",
        default=[
            "artifacts/v3_sam3_mask_debug_v02_qa16",
            "artifacts/v3_sam3_mask_debug_manual_v2_qa16",
        ],
        help="One or more SAM3 QA output directories.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["auto", "manual-v2"],
        help="Display labels matching --qa_dirs.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/v3_sam3_mask_review_manual_v2",
    )
    parser.add_argument("--sample_ids", nargs="+", type=int, default=list(range(16)))
    parser.add_argument("--frame_count", type=int, default=17)
    parser.add_argument("--detail_run", default=None, help="Label or index to use for focused detail outputs.")
    parser.add_argument("--include_motion", action="store_true", help="Include motion rows in focused detail sheets.")
    parser.add_argument(
        "--make_view_splits",
        action="store_true",
        help="Also export enlarged top/bottom view focused sheets for fine-detail inspection.",
    )
    parser.add_argument("--view_scale", type=int, default=2, help="Scale factor for split-view cells.")
    parser.add_argument(
        "--make_issue_crops",
        action="store_true",
        help="Export compact enlarged crops for zero-mask or high-priority manual-review cases.",
    )
    parser.add_argument("--issue_scale", type=int, default=4, help="Scale factor for issue crop cells.")
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


def text_lines(text: str, max_chars: int) -> List[str]:
    words = str(text).split()
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
    return lines or [str(text)]


def sample_name(sample_id: int) -> str:
    return f"sample_{sample_id:03d}"


def stats_path(qa_dir: Path, sample_id: int) -> Path:
    name = sample_name(sample_id)
    return qa_dir / name / f"{name}_sam3_mask_debug_stats.json"


def sheet_path(qa_dir: Path, sample_id: int) -> Path:
    name = sample_name(sample_id)
    return qa_dir / name / f"{name}_sam3_mask_debug_17f.png"


def load_stats(qa_dir: Path, sample_id: int) -> Dict:
    with open(stats_path(qa_dir, sample_id), "r", encoding="utf-8") as file:
        return json.load(file)


def is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return True


def row_names_from_stats(stats: Dict) -> List[str]:
    names = ["RGB"]
    if is_present(stats.get("motion")):
        names.append("motion cue")
    if is_present(stats.get("gated_motion")):
        names.append("motion near SAM3")
    names.extend(list(stats.get("summary", {}).keys()))
    if is_present(stats.get("sam3_union")):
        names.append("SAM3 union")
    if is_present(stats.get("interaction_draft")):
        names.append("interaction draft")
    if is_present(stats.get("selected_union")):
        names.append("selected SAM3")
    if is_present(stats.get("selected_focus")):
        names.append("selected focus")
    if is_present(stats.get("selected_interaction")):
        names.append("selected interaction")
    for role in ("robot", "object", "target", "context"):
        if is_present(stats.get("role_unions", {}).get(role)):
            names.append(f"role: {role}")
    if is_present(stats.get("actor")):
        names.append("actor mask")
    if is_present(stats.get("target_local")):
        names.append("local target")
    if is_present(stats.get("motion_near_final")):
        names.append("motion near final")
    if is_present(stats.get("final_train_mask")):
        names.append("final train mask")
    return names


def row_map_from_stats(stats: Dict) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(row_names_from_stats(stats))}


def stat_line(stats: Dict, row_name: str) -> str:
    if row_name.startswith("role: "):
        key = row_name.split(": ", 1)[1]
        row_stats = stats.get("role_unions", {}).get(key, {})
    elif row_name == "actor mask":
        row_stats = stats.get("actor", {})
    elif row_name == "local target":
        row_stats = stats.get("target_local", {})
    elif row_name == "motion near final":
        row_stats = stats.get("motion_near_final", {})
    elif row_name == "final train mask":
        row_stats = stats.get("final_train_mask", {})
    elif row_name in stats.get("summary", {}):
        row_stats = stats.get("summary", {}).get(row_name, {})
    else:
        row_stats = {}

    if not row_stats:
        return ""
    zero = row_stats.get("zero_frame_count", "")
    mean = row_stats.get("mean_area_ratio", "")
    max_area = row_stats.get("max_area_ratio", "")
    top = row_stats.get("mean_top_view_fraction", "")
    bottom = row_stats.get("mean_bottom_view_fraction", "")
    parts = []
    if zero != "":
        parts.append(f"zero={zero}")
    if mean != "":
        parts.append(f"mean={float(mean):.3f}")
    if max_area != "":
        parts.append(f"max={float(max_area):.3f}")
    if top != "" and bottom != "":
        parts.append(f"top/bot={float(top):.2f}/{float(bottom):.2f}")
    return " | ".join(parts)


def crop_frame_strip(sheet: Image.Image, row_idx: int, frame_count: int) -> Image.Image:
    y0 = HEADER_H + row_idx * (CELL_H + ROW_GAP)
    x0 = LABEL_W
    return sheet.crop((x0, y0, x0 + frame_count * CELL_W, y0 + CELL_H))


def crop_view_strip(
    sheet: Image.Image,
    row_idx: int,
    frame_count: int,
    view_name: str,
    scale: int,
) -> Image.Image:
    if view_name not in ("top", "bottom"):
        raise ValueError(f"Unsupported view_name={view_name!r}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    view_h = CELL_H // 2
    y_base = HEADER_H + row_idx * (CELL_H + ROW_GAP)
    y0 = y_base if view_name == "top" else y_base + view_h
    strip = Image.new("RGB", (frame_count * CELL_W * scale, view_h * scale), (245, 246, 248))
    for frame_idx in range(frame_count):
        x0 = LABEL_W + frame_idx * CELL_W
        crop = sheet.crop((x0, y0, x0 + CELL_W, y0 + view_h))
        if scale != 1:
            crop = crop.resize((CELL_W * scale, view_h * scale), Image.Resampling.NEAREST)
        strip.paste(crop, (frame_idx * CELL_W * scale, 0))
    return strip


def crop_view_cell(
    sheet: Image.Image,
    row_idx: int,
    frame_idx: int,
    view_name: str,
    scale: int,
) -> Image.Image:
    if view_name not in ("top", "bottom"):
        raise ValueError(f"Unsupported view_name={view_name!r}")
    view_h = CELL_H // 2
    y_base = HEADER_H + row_idx * (CELL_H + ROW_GAP)
    y0 = y_base if view_name == "top" else y_base + view_h
    x0 = LABEL_W + frame_idx * CELL_W
    crop = sheet.crop((x0, y0, x0 + CELL_W, y0 + view_h))
    if scale != 1:
        crop = crop.resize((CELL_W * scale, view_h * scale), Image.Resampling.NEAREST)
    return crop


def draw_header(
    out: Image.Image,
    title: str,
    subtitle: str,
    note: str,
) -> None:
    draw = ImageDraw.Draw(out)
    title_font = font(31)
    body_font = font(18)
    small_font = font(16)
    draw.rectangle((0, 0, out.width, 154), fill=(246, 247, 249))
    draw.text((24, 18), title, fill=(15, 20, 25), font=title_font)
    draw.text((24, 64), subtitle, fill=(70, 76, 84), font=body_font)
    draw.text((24, 100), note, fill=(105, 84, 30), font=small_font)


def draw_row_label(draw: ImageDraw.ImageDraw, y: int, row_label_w: int, label: str, stats_text: str) -> None:
    title_font = font(25)
    small_font = font(16)
    draw.rectangle((0, y, row_label_w - 1, y + CELL_H), fill=(232, 235, 239))
    yy = y + 24
    for line in text_lines(label, 24)[:5]:
        draw.text((22, yy), line, fill=(20, 24, 30), font=title_font)
        yy += 30
    if stats_text:
        yy += 14
        for line in text_lines(stats_text, 30)[:4]:
            draw.text((22, yy), line, fill=(82, 88, 96), font=small_font)
            yy += 23


def paste_review_row(
    out: Image.Image,
    sheet: Image.Image,
    row_idx: int,
    y: int,
    row_label_w: int,
    label: str,
    stats_text: str,
    frame_count: int,
) -> None:
    draw = ImageDraw.Draw(out)
    draw_row_label(draw, y, row_label_w, label, stats_text)
    strip = crop_frame_strip(sheet, row_idx, frame_count)
    out.paste(strip, (row_label_w, y))


def focused_rows(stats: Dict, include_motion: bool) -> List[str]:
    rows = ["RGB"]
    if include_motion:
        rows.extend(["motion cue", "motion near SAM3"])
    rows.extend([name for name in stats.get("summary", {}) if name.startswith("manual:")])
    for name in [
        "role: robot",
        "role: object",
        "role: target",
        "actor mask",
        "local target",
        "motion near final",
        "final train mask",
    ]:
        rows.append(name)
    return rows


def make_focused_detail(
    qa_dir: Path,
    run_label: str,
    sample_id: int,
    output_dir: Path,
    frame_count: int,
    include_motion: bool,
) -> Path:
    stats = load_stats(qa_dir, sample_id)
    row_map = row_map_from_stats(stats)
    rows = [row for row in focused_rows(stats, include_motion) if row in row_map]
    if not rows:
        raise RuntimeError(f"No review rows found for {qa_dir} {sample_name(sample_id)}")
    sheet = Image.open(sheet_path(qa_dir, sample_id)).convert("RGB")

    row_label_w = 520
    header_h = 154
    out_w = row_label_w + frame_count * CELL_W
    out_h = header_h + len(rows) * CELL_H + (len(rows) - 1) * ROW_GAP
    out = Image.new("RGB", (out_w, out_h), (245, 246, 248))
    task = stats.get("window", {}).get("task_text", "")
    frame_indices = stats.get("window", {}).get("frame_indices", [])
    subtitle = f"{run_label} | {task}"
    note = f"17F fixed window | original frames {frame_indices[0] if frame_indices else '?'}..{frame_indices[-1] if frame_indices else '?'}"
    draw_header(out, f"{sample_name(sample_id)} focused SAM3 mask review", subtitle, note)

    y = header_h
    for row in rows:
        paste_review_row(
            out,
            sheet,
            row_map[row],
            y,
            row_label_w,
            row,
            stat_line(stats, row),
            frame_count,
        )
        y += CELL_H + ROW_GAP

    out_dir = output_dir / "focused" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_name(sample_id)}_{run_label}_focused_review.png"
    out.save(out_path)
    return out_path


def make_view_split_detail(
    qa_dir: Path,
    run_label: str,
    sample_id: int,
    output_dir: Path,
    frame_count: int,
    include_motion: bool,
    view_name: str,
    scale: int,
) -> Path:
    stats = load_stats(qa_dir, sample_id)
    row_map = row_map_from_stats(stats)
    rows = [row for row in focused_rows(stats, include_motion) if row in row_map]
    if not rows:
        raise RuntimeError(f"No view-split rows found for {qa_dir} {sample_name(sample_id)}")
    sheet = Image.open(sheet_path(qa_dir, sample_id)).convert("RGB")

    cell_h = (CELL_H // 2) * scale
    row_label_w = 560
    header_h = 154
    out_w = row_label_w + frame_count * CELL_W * scale
    out_h = header_h + len(rows) * cell_h + (len(rows) - 1) * ROW_GAP
    out = Image.new("RGB", (out_w, out_h), (245, 246, 248))
    task = stats.get("window", {}).get("task_text", "")
    frame_indices = stats.get("window", {}).get("frame_indices", [])
    subtitle = f"{run_label} | {view_name} view | {task}"
    note = (
        f"17F fixed window | view split from 448x224 vertical composite | "
        f"frames {frame_indices[0] if frame_indices else '?'}..{frame_indices[-1] if frame_indices else '?'}"
    )
    draw_header(out, f"{sample_name(sample_id)} enlarged {view_name} view review", subtitle, note)

    draw = ImageDraw.Draw(out)
    title_font = font(25)
    small_font = font(16)
    y = header_h
    for row in rows:
        draw.rectangle((0, y, row_label_w - 1, y + cell_h), fill=(232, 235, 239))
        yy = y + 20
        for line in text_lines(row, 26)[:4]:
            draw.text((22, yy), line, fill=(20, 24, 30), font=title_font)
            yy += 30
        stats_text = stat_line(stats, row)
        if stats_text:
            yy += 10
            for line in text_lines(stats_text, 32)[:4]:
                draw.text((22, yy), line, fill=(82, 88, 96), font=small_font)
                yy += 23

        strip = crop_view_strip(sheet, row_map[row], frame_count, view_name, scale)
        out.paste(strip, (row_label_w, y))
        y += cell_h + ROW_GAP

    out_dir = output_dir / "focused_view_splits" / run_label / view_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_name(sample_id)}_{run_label}_{view_name}_view_review.png"
    out.save(out_path)
    return out_path


def zero_frames_for_role(stats: Dict, role: str) -> List[int]:
    role_stats = stats.get("role_unions", {}).get(role, {})
    return [
        idx
        for idx, area in enumerate(role_stats.get("mask_area_ratio", []) or [])
        if float(area) == 0.0
    ]


def issue_frames_for_sample(stats: Dict, checklist: Dict[str, str], frame_count: int) -> Tuple[List[int], List[str]]:
    reasons: List[str] = []
    frame_ids = set()
    for role in ("object", "target"):
        zeros = zero_frames_for_role(stats, role)
        if zeros:
            frame_ids.update(zeros)
            reasons.append(f"{role}_zero={zeros}")

    priority = checklist.get("priority", "")
    status = checklist.get("current_status", "")
    if not frame_ids and priority in ("critical", "high"):
        frame_ids.update(idx for idx in (0, 4, 8, 12, 16) if idx < frame_count)
        reasons.append(status or priority)

    return sorted(frame_ids), reasons


def issue_rows_for_stats(stats: Dict) -> List[str]:
    row_map = row_map_from_stats(stats)
    rows = ["RGB", "role: object", "role: target", "role: robot", "final train mask"]
    return [row for row in rows if row in row_map]


def make_issue_crop_sheet(
    qa_dir: Path,
    run_label: str,
    sample_id: int,
    output_dir: Path,
    frame_count: int,
    checklist: Dict[str, str],
    scale: int,
) -> Path | None:
    stats = load_stats(qa_dir, sample_id)
    issue_frames, reasons = issue_frames_for_sample(stats, checklist, frame_count)
    if not issue_frames:
        return None

    row_map = row_map_from_stats(stats)
    rows = issue_rows_for_stats(stats)
    sheet = Image.open(sheet_path(qa_dir, sample_id)).convert("RGB")

    cell_w = CELL_W * scale
    cell_h = (CELL_H // 2) * scale
    label_w = 540
    header_h = 158
    frame_header_h = 44
    out_w = label_w + len(issue_frames) * cell_w
    out_h = header_h + frame_header_h + len(rows) * 2 * cell_h + (len(rows) * 2 - 1) * ROW_GAP
    out = Image.new("RGB", (out_w, out_h), (245, 246, 248))
    draw = ImageDraw.Draw(out)
    task = stats.get("window", {}).get("task_text", "")
    reason_text = "; ".join(reasons)
    draw_header(
        out,
        f"{sample_name(sample_id)} issue crop review",
        f"{run_label} | {task}",
        f"Frames: {issue_frames} | {reason_text}",
    )

    small_font = font(18)
    title_font = font(24)
    y = header_h
    draw.rectangle((0, y, label_w - 1, y + frame_header_h), fill=(224, 229, 236))
    draw.text((18, y + 11), "row / view", fill=(40, 48, 58), font=small_font)
    for col_idx, frame_idx in enumerate(issue_frames):
        x = label_w + col_idx * cell_w
        draw.rectangle((x, y, x + cell_w - 1, y + frame_header_h), fill=(224, 229, 236))
        draw.text((x + 14, y + 11), f"frame {frame_idx:02d}", fill=(40, 48, 58), font=small_font)
    y += frame_header_h

    for row in rows:
        for view_name in ("top", "bottom"):
            draw.rectangle((0, y, label_w - 1, y + cell_h), fill=(232, 235, 239))
            label = f"{row} | {view_name}"
            yy = y + 22
            for line in text_lines(label, 28)[:3]:
                draw.text((20, yy), line, fill=(20, 24, 30), font=title_font)
                yy += 31
            if row != "RGB":
                stats_text = stat_line(stats, row)
                if stats_text:
                    yy += 8
                    for line in text_lines(stats_text, 34)[:3]:
                        draw.text((20, yy), line, fill=(82, 88, 96), font=small_font)
                        yy += 24
            for col_idx, frame_idx in enumerate(issue_frames):
                crop = crop_view_cell(sheet, row_map[row], frame_idx, view_name, scale)
                out.paste(crop, (label_w + col_idx * cell_w, y))
            y += cell_h + ROW_GAP

    out_dir = output_dir / "issue_crops" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_name(sample_id)}_{run_label}_issue_crop_review.png"
    out.save(out_path)
    return out_path


def compare_rows() -> List[str]:
    return ["role: object", "role: target", "role: robot", "final train mask"]


def make_compare_detail(
    qa_dirs: Sequence[Path],
    labels: Sequence[str],
    sample_id: int,
    output_dir: Path,
    frame_count: int,
) -> Path:
    stats_by_label = {label: load_stats(qa_dir, sample_id) for label, qa_dir in zip(labels, qa_dirs)}
    sheets_by_label = {label: Image.open(sheet_path(qa_dir, sample_id)).convert("RGB") for label, qa_dir in zip(labels, qa_dirs)}
    maps_by_label = {label: row_map_from_stats(stats) for label, stats in stats_by_label.items()}

    first_stats = next(iter(stats_by_label.values()))
    row_label_w = 560
    header_h = 154
    rows: List[Tuple[str, str]] = [("RGB", labels[-1])]
    for row in compare_rows():
        for label in labels:
            if row in maps_by_label[label]:
                rows.append((row, label))

    out_w = row_label_w + frame_count * CELL_W
    out_h = header_h + len(rows) * CELL_H + (len(rows) - 1) * ROW_GAP
    out = Image.new("RGB", (out_w, out_h), (245, 246, 248))
    task = first_stats.get("window", {}).get("task_text", "")
    frame_indices = first_stats.get("window", {}).get("frame_indices", [])
    subtitle = f"{task}"
    label_text = " / ".join(labels)
    note = f"Rows compare {label_text} on the same 17F window | frames {frame_indices[0] if frame_indices else '?'}..{frame_indices[-1] if frame_indices else '?'}"
    draw_header(out, f"{sample_name(sample_id)} baseline/manual mask comparison", subtitle, note)

    y = header_h
    for row, label in rows:
        stats = stats_by_label[label]
        sheet = sheets_by_label[label]
        row_map = maps_by_label[label]
        display_label = "RGB" if row == "RGB" else f"{label} | {row}"
        paste_review_row(
            out,
            sheet,
            row_map[row],
            y,
            row_label_w,
            display_label,
            "" if row == "RGB" else stat_line(stats, row),
            frame_count,
        )
        y += CELL_H + ROW_GAP

    out_dir = output_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_labels = "_vs_".join(label.replace("/", "-").replace(" ", "-") for label in labels)
    out_path = out_dir / f"{sample_name(sample_id)}_{safe_labels}_review.png"
    out.save(out_path)
    return out_path


def metric(stats: Dict, row_name: str, field: str) -> str:
    if row_name.startswith("role: "):
        row_stats = stats.get("role_unions", {}).get(row_name.split(": ", 1)[1], {})
    elif row_name == "final train mask":
        row_stats = stats.get("final_train_mask", {})
    else:
        row_stats = {}
    value = row_stats.get(field, "")
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_review_summary(
    qa_dirs: Sequence[Path],
    labels: Sequence[str],
    sample_ids: Iterable[int],
    output_dir: Path,
) -> Path:
    rows = []
    for sample_id in sample_ids:
        for label, qa_dir in zip(labels, qa_dirs):
            stats = load_stats(qa_dir, sample_id)
            rows.append(
                {
                    "sample": sample_name(sample_id),
                    "run": label,
                    "task_text": stats.get("window", {}).get("task_text", ""),
                    "object_zero": metric(stats, "role: object", "zero_frame_count"),
                    "object_mean": metric(stats, "role: object", "mean_area_ratio"),
                    "target_zero": metric(stats, "role: target", "zero_frame_count"),
                    "target_mean": metric(stats, "role: target", "mean_area_ratio"),
                    "final_mean": metric(stats, "final train mask", "mean_area_ratio"),
                    "final_max": metric(stats, "final train mask", "max_area_ratio"),
                }
            )
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "review_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_issue_summary(
    qa_dir: Path,
    sample_ids: Sequence[int],
    checklist_by_sample: Dict[str, Dict[str, str]],
    output_dir: Path,
    frame_count: int,
) -> Path:
    rows = []
    for sample_id in sample_ids:
        name = sample_name(sample_id)
        stats = load_stats(qa_dir, sample_id)
        checklist = checklist_by_sample.get(name, {})
        issue_frames, reasons = issue_frames_for_sample(stats, checklist, frame_count)
        if not issue_frames:
            continue
        rows.append(
            {
                "sample": name,
                "priority": checklist.get("priority", ""),
                "current_status": checklist.get("current_status", ""),
                "issue_frames": " ".join(str(idx) for idx in issue_frames),
                "reasons": "; ".join(reasons),
                "task_text": stats.get("window", {}).get("task_text", ""),
            }
        )

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "issue_frames.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["sample", "priority", "current_status", "issue_frames", "reasons", "task_text"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def relative_href(path: Path, html_path: Path) -> str:
    return os.path.relpath(path.resolve(), start=html_path.parent.resolve())


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def first_by_key(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {row.get(key, ""): row for row in rows}


def metric_badge(label: str, value: str) -> str:
    if value == "":
        return f"<span class='metric muted'>{html.escape(label)}: n/a</span>"
    return f"<span class='metric'>{html.escape(label)}: {html.escape(value)}</span>"


def sample_from_path(path: Path) -> str:
    return path.name.split("_", 2)[0] + "_" + path.name.split("_", 2)[1]


def path_map(paths: Sequence[Path]) -> Dict[str, Path]:
    return {sample_from_path(path): path for path in paths}


def manual_preview_sets(output_dir: Path) -> List[Tuple[str, Dict[str, Path]]]:
    preview_sets: List[Tuple[str, Dict[str, Path]]] = []
    for preview_dir in sorted(output_dir.glob("manual_override_previews*")):
        if not preview_dir.is_dir():
            continue
        paths = sorted(preview_dir.glob("*_manual_override_preview.png"))
        if not paths:
            continue
        label = preview_dir.name.replace("manual_override_previews_", "override preview ")
        label = label.replace("manual_override_previews", "override preview")
        preview_sets.append((label, path_map(paths)))
    return preview_sets


def write_html_dashboard(
    output_dir: Path,
    sample_ids: Sequence[int],
    focused_paths: Sequence[Path],
    compare_paths: Sequence[Path],
    view_split_paths: Sequence[Path],
    issue_paths: Sequence[Path],
    summary_path: Path,
    detail_label: str,
) -> Path:
    summary_dir = output_dir / "summary"
    html_path = summary_dir / "sam3_mask_review_dashboard.html"
    checklist_path = summary_dir / "manual_review_checklist.csv"
    notes_path = summary_dir / "review_notes.md"
    decisions_path = summary_dir / "human_annotation_decisions.md"
    decision_pack_path = summary_dir / "human_decision_pack.html"
    run_delta_path = summary_dir / "run_delta_summary.md"
    run_delta_csv_path = summary_dir / "run_delta_summary.csv"
    verification_path = summary_dir / "review_bundle_verification.md"
    verification_json_path = summary_dir / "review_bundle_verification.json"

    metrics_rows = read_csv_rows(summary_path)
    checklist_rows = read_csv_rows(checklist_path)
    checklist_by_sample = first_by_key(checklist_rows, "sample")
    manual_metrics = {
        row["sample"]: row
        for row in metrics_rows
        if row.get("run") == detail_label
    }
    if not manual_metrics:
        manual_metrics = {
            row["sample"]: row
            for row in metrics_rows
            if row.get("run") in ("manual-v2", "manual")
        }
    focused_by_sample = path_map(focused_paths)
    compare_by_sample = path_map(compare_paths)
    top_by_sample = {sample_from_path(path): path for path in view_split_paths if "_top_view_" in path.name}
    bottom_by_sample = {sample_from_path(path): path for path in view_split_paths if "_bottom_view_" in path.name}
    issue_by_sample = path_map(issue_paths)
    annotation_by_sample = path_map(
        sorted((output_dir / "issue_annotation_guides").glob("*_manual_box_guide_issue_frames.png"))
    )
    operation_crop_by_sample = path_map(
        sorted((output_dir / "operation_crops").glob("*_operation_crop_review.png"))
    )
    identity_aid_by_sample = path_map(
        sorted((output_dir / "identity_decision_aids").glob("*_identity_decision_aid.png"))
    )
    preview_sets = manual_preview_sets(output_dir)

    priority_rank = {"critical": 0, "high": 1, "normal": 2}
    sample_names = [sample_name(sample_id) for sample_id in sample_ids]
    sample_names.sort(
        key=lambda name: (
            priority_rank.get(checklist_by_sample.get(name, {}).get("priority", "normal"), 9),
            name,
        )
    )

    cards: List[str] = []
    for name in sample_names:
        checklist = checklist_by_sample.get(name, {})
        metrics = manual_metrics.get(name, {})
        priority = checklist.get("priority", "normal")
        status = checklist.get("current_status", "")
        task = checklist.get("task_text") or metrics.get("task_text", "")
        note = checklist.get("notes", "")
        decision = checklist.get("decision", "")
        links = []
        for label, path in (
            ("focused", focused_by_sample.get(name)),
            ("compare", compare_by_sample.get(name)),
            ("issue crop", issue_by_sample.get(name)),
            ("operation crop", operation_crop_by_sample.get(name)),
            ("identity aid", identity_aid_by_sample.get(name)),
            ("box guide", annotation_by_sample.get(name)),
            ("top view", top_by_sample.get(name)),
            ("bottom view", bottom_by_sample.get(name)),
        ):
            if path and path.exists():
                links.append(
                    f"<a href='{html.escape(relative_href(path, html_path))}'>{html.escape(label)}</a>"
                )
        for label, preview_by_sample in preview_sets:
            path = preview_by_sample.get(name)
            if path and path.exists():
                links.append(
                    f"<a href='{html.escape(relative_href(path, html_path))}'>{html.escape(label)}</a>"
                )
        hero = compare_by_sample.get(name) or focused_by_sample.get(name)
        hero_html = ""
        if hero and hero.exists():
            hero_html = (
                f"<a href='{html.escape(relative_href(hero, html_path))}'>"
                f"<img src='{html.escape(relative_href(hero, html_path))}' alt='{html.escape(name)} review'>"
                "</a>"
            )
        cards.append(
            "\n".join(
                [
                    f"<section class='card priority-{html.escape(priority)}'>",
                    "<div class='card-head'>",
                    f"<h2>{html.escape(name)}</h2>",
                    f"<span class='pill'>{html.escape(priority)}</span>",
                    f"<span class='pill status'>{html.escape(status)}</span>",
                    "</div>",
                    f"<p class='task'>{html.escape(task)}</p>",
                    "<div class='metrics'>",
                    metric_badge("object_zero", metrics.get("object_zero", "")),
                    metric_badge("target_zero", metrics.get("target_zero", "")),
                    metric_badge("final_mean", metrics.get("final_mean", "")),
                    metric_badge("final_max", metrics.get("final_max", "")),
                    "</div>",
                    f"<p class='note'>{html.escape(note)}</p>" if note else "",
                    f"<p class='decision'>decision: {html.escape(decision or 'pending')}</p>",
                    f"<p class='links'>{' | '.join(links)}</p>",
                    hero_html,
                    "</section>",
                ]
            )
        )

    extra_links = []
    for label, path in (
        ("metrics csv", summary_path),
        ("issue frames", summary_dir / "issue_frames.csv"),
        ("manual checklist", checklist_path),
        ("review notes", notes_path),
        ("human decisions", decisions_path),
        ("decision pack", decision_pack_path),
        ("run delta", run_delta_path),
        ("run delta csv", run_delta_csv_path),
        ("verification", verification_path),
        ("verification json", verification_json_path),
    ):
        if path.exists():
            extra_links.append(f"<a href='{html.escape(relative_href(path, html_path))}'>{html.escape(label)}</a>")
    for path in sorted(summary_dir.glob("manual_override_validation*.json")):
        label = path.stem.replace("manual_override_", "").replace("_", " ")
        extra_links.append(f"<a href='{html.escape(relative_href(path, html_path))}'>{html.escape(label)}</a>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAM3 Mask Review Dashboard</title>
  <style>
    body {{
      margin: 0;
      background: #f4f6f8;
      color: #15191f;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid #d9dee7;
      padding: 18px 24px 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 720;
    }}
    .sub {{
      margin: 0;
      color: #596270;
      font-size: 14px;
    }}
    .toolbar {{
      margin-top: 10px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 14px;
    }}
    a {{
      color: #1b65d8;
      text-decoration: none;
      font-weight: 620;
    }}
    main {{
      padding: 20px 24px 60px;
      display: grid;
      gap: 22px;
    }}
    .card {{
      background: white;
      border: 1px solid #dce1e8;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(20, 30, 45, 0.04);
    }}
    .priority-critical {{
      border-left: 8px solid #d93636;
    }}
    .priority-high {{
      border-left: 8px solid #e5a000;
    }}
    .priority-normal {{
      border-left: 8px solid #7c8796;
    }}
    .card-head {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .pill {{
      background: #edf1f6;
      border: 1px solid #d7dde7;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .status {{
      text-transform: none;
      font-weight: 620;
    }}
    .task {{
      margin: 4px 0 10px;
      color: #3c4654;
      font-size: 15px;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0;
    }}
    .metric {{
      background: #f6f8fb;
      border: 1px solid #dde4ee;
      border-radius: 6px;
      padding: 5px 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .muted {{
      color: #8a94a3;
    }}
    .note, .decision, .links {{
      margin: 8px 0;
      font-size: 14px;
    }}
    .note {{
      color: #6b4a00;
    }}
    .decision {{
      color: #5a6472;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 900px;
      object-fit: contain;
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #111;
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>SAM3 Mask Review Dashboard</h1>
    <p class="sub">16 fixed LIBERO VGM windows. Focus run: {html.escape(detail_label)}. Use this page to approve mask definitions before any full-dataset mask generation.</p>
    <nav class="toolbar">{' | '.join(extra_links)}</nav>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def detail_run_index(labels: Sequence[str], detail_run: str | None) -> int:
    if detail_run is None:
        return len(labels) - 1
    if detail_run in labels:
        return list(labels).index(detail_run)
    idx = int(detail_run)
    if idx < 0 or idx >= len(labels):
        raise IndexError(f"detail_run index {idx} out of range for {len(labels)} labels")
    return idx


def main() -> None:
    args = parse_args()
    qa_dirs = [Path(path) for path in args.qa_dirs]
    labels = list(args.labels)
    if len(labels) != len(qa_dirs):
        raise ValueError(f"--labels count ({len(labels)}) must match --qa_dirs count ({len(qa_dirs)})")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focused_idx = detail_run_index(labels, args.detail_run)
    checklist_path = output_dir / "summary" / "manual_review_checklist.csv"
    checklist_by_sample = first_by_key(read_csv_rows(checklist_path), "sample")
    focused_paths = []
    compare_paths = []
    view_split_paths = []
    issue_paths = []
    for sample_id in args.sample_ids:
        focused_paths.append(
            make_focused_detail(
                qa_dirs[focused_idx],
                labels[focused_idx],
                sample_id,
                output_dir,
                args.frame_count,
                args.include_motion,
            )
        )
        if args.make_view_splits:
            for view_name in ("top", "bottom"):
                view_split_paths.append(
                    make_view_split_detail(
                        qa_dirs[focused_idx],
                        labels[focused_idx],
                        sample_id,
                        output_dir,
                        args.frame_count,
                        args.include_motion,
                        view_name,
                        args.view_scale,
                    )
                )
        if len(qa_dirs) >= 2:
            compare_paths.append(make_compare_detail(qa_dirs, labels, sample_id, output_dir, args.frame_count))
        if args.make_issue_crops:
            issue_path = make_issue_crop_sheet(
                qa_dirs[focused_idx],
                labels[focused_idx],
                sample_id,
                output_dir,
                args.frame_count,
                checklist_by_sample.get(sample_name(sample_id), {}),
                args.issue_scale,
            )
            if issue_path is not None:
                issue_paths.append(issue_path)

    summary_path = write_review_summary(qa_dirs, labels, args.sample_ids, output_dir)
    issue_summary_path = write_issue_summary(
        qa_dirs[focused_idx],
        args.sample_ids,
        checklist_by_sample,
        output_dir,
        args.frame_count,
    )
    html_path = write_html_dashboard(
        output_dir,
        args.sample_ids,
        focused_paths,
        compare_paths,
        view_split_paths,
        issue_paths,
        summary_path,
        labels[focused_idx],
    )
    index_path = output_dir / "summary" / "review_outputs.txt"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write("[dashboard]\n")
        file.write(str(html_path.resolve()) + "\n")
        file.write("[focused]\n")
        for path in focused_paths:
            file.write(str(path.resolve()) + "\n")
        file.write("\n[compare]\n")
        for path in compare_paths:
            file.write(str(path.resolve()) + "\n")
        file.write("\n[focused_view_splits]\n")
        for path in view_split_paths:
            file.write(str(path.resolve()) + "\n")
        file.write("\n[issue_crops]\n")
        for path in issue_paths:
            file.write(str(path.resolve()) + "\n")
        operation_crop_paths = sorted((output_dir / "operation_crops").glob("*_operation_crop_review.png"))
        if operation_crop_paths:
            file.write("\n[operation_crops]\n")
            for path in operation_crop_paths:
                file.write(str(path.resolve()) + "\n")
        identity_aid_paths = sorted((output_dir / "identity_decision_aids").glob("*_identity_decision_aid.png"))
        if identity_aid_paths:
            file.write("\n[identity_decision_aids]\n")
            for path in identity_aid_paths:
                file.write(str(path.resolve()) + "\n")
        issue_annotation_paths = sorted((output_dir / "issue_annotation_guides").glob("*_manual_box_guide_issue_frames.png"))
        if issue_annotation_paths:
            file.write("\n[issue_annotation_guides]\n")
            for path in issue_annotation_paths:
                file.write(str(path.resolve()) + "\n")
        for preview_dir in sorted(output_dir.glob("manual_override_previews*")):
            if not preview_dir.is_dir():
                continue
            manual_preview_paths = sorted(preview_dir.glob("*_manual_override_preview.png"))
            if manual_preview_paths:
                file.write(f"\n[{preview_dir.name}]\n")
                for path in manual_preview_paths:
                    file.write(str(path.resolve()) + "\n")
        file.write(f"\n[metrics]\n{summary_path.resolve()}\n")
        file.write(str(issue_summary_path.resolve()) + "\n")
        for validation_path in sorted((output_dir / "summary").glob("manual_override_validation*.json")):
            file.write(str(validation_path.resolve()) + "\n")
        checklist_path = output_dir / "summary" / "manual_review_checklist.csv"
        notes_path = output_dir / "summary" / "review_notes.md"
        decisions_path = output_dir / "summary" / "human_annotation_decisions.md"
        decision_pack_path = output_dir / "summary" / "human_decision_pack.html"
        decision_template_path = output_dir / "summary" / "human_decision_template.csv"
        run_delta_path = output_dir / "summary" / "run_delta_summary.md"
        run_delta_csv_path = output_dir / "summary" / "run_delta_summary.csv"
        verification_path = output_dir / "summary" / "review_bundle_verification.md"
        verification_json_path = output_dir / "summary" / "review_bundle_verification.json"
        optional_paths = [
            path
            for path in (
                notes_path,
                decisions_path,
                decision_pack_path,
                decision_template_path,
                run_delta_path,
                run_delta_csv_path,
                verification_path,
                verification_json_path,
                checklist_path,
            )
            if path.exists()
        ]
        if optional_paths:
            file.write("\n[notes]\n")
            for path in optional_paths:
                file.write(str(path.resolve()) + "\n")

    print(f"Wrote {len(focused_paths)} focused review images")
    print(f"Wrote {len(compare_paths)} comparison review images")
    print(f"Wrote {len(view_split_paths)} focused view-split images")
    print(f"Wrote {len(issue_paths)} issue crop images")
    print(f"Wrote dashboard {html_path.resolve()}")
    print(index_path.resolve())


if __name__ == "__main__":
    main()
