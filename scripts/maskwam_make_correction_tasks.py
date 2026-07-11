#!/usr/bin/env python3
"""Create a human correction task list from MaskWAM first-pass metadata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from maskwam_view_schema import FINAL_ROLE, VIEW_ROLE_ORDER, VIEW_TRAIN_ROLES, base_role


ROLE_ORDER = {**VIEW_ROLE_ORDER, FINAL_ROLE: 99}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
TRAIN_ROLES = set(VIEW_TRAIN_ROLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--annotation_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass_qa16")
    parser.add_argument("--review_dir", default=None)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_html", default=None)
    parser.add_argument("--output_corrections_template", default=None)
    parser.add_argument("--min_area_ratio", type=float, default=0.001)
    parser.add_argument("--max_area_ratio", type=float, default=0.5)
    parser.add_argument("--include_robot_partial", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "sample_id",
        "sample_name",
        "role",
        "name",
        "reason",
        "zero_frame_count",
        "mean_area_ratio",
        "zero_frame_indices",
        "recommended_frame_index",
        "recommended_frame_reason",
        "suggested_action",
        "task_text",
        "review_sheet",
        "click_frames",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rel_link(target: str, output_html: Path) -> str:
    if not target:
        return ""
    target_path = Path(target)
    if not target_path.exists():
        return html.escape(target, quote=True)
    return html.escape(os.path.relpath(target_path.resolve(), output_html.resolve().parent), quote=True)


def write_html(path: str | Path, report: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in report["tasks"]:
        grouped.setdefault(str(task["priority"]), []).append(task)

    sections = []
    for priority in sorted(grouped, key=lambda value: PRIORITY_ORDER.get(value, 99)):
        rows = []
        for task in grouped[priority]:
            sheet_href = rel_link(str(task.get("review_sheet", "")), path)
            sheet_link = f"<a href='{sheet_href}'>sheet</a>" if sheet_href else ""
            click_links = []
            for click_path in str(task.get("click_frames", "")).split():
                href = rel_link(click_path, path)
                label = Path(click_path).stem.rsplit("_frame_", 1)[-1]
                click_links.append(f"<a href='{href}'>F{html.escape(label)}</a>")
            rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(task['sample_name']))}</td>"
                    f"<td>{html.escape(str(task['role']))}</td>"
                    f"<td>{html.escape(str(task['name']))}</td>"
                    f"<td>{html.escape(str(task['reason']))}</td>"
                    f"<td>F{html.escape(str(task.get('recommended_frame_index', '')))} · {html.escape(str(task.get('recommended_frame_reason', '')))}</td>"
                    f"<td>{html.escape(str(task['suggested_action']))}</td>"
                    f"<td>{sheet_link}</td>"
                    f"<td>{' '.join(click_links)}</td>"
                    "</tr>"
                )
        sections.append(
            "\n".join(
                [
                    f"<section><h2>{html.escape(priority)} ({len(rows)})</h2>",
                    "<table>",
                    "<thead><tr><th>sample</th><th>role</th><th>name</th><th>reason</th><th>recommended</th><th>action</th><th>sheet</th><th>frames</th></tr></thead>",
                    "<tbody>",
                    "\n".join(rows),
                    "</tbody></table></section>",
                ]
            )
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MaskWAM Correction Tasks</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f6f7f9; }}
    h1 {{ margin-bottom: 4px; }}
    .summary, section {{ background: #fff; border: 1px solid #dde1e7; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-top: 1px solid #e1e5ea; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #52606d; font-weight: 600; background: #f8fafc; }}
    a {{ color: #145ea8; text-decoration: none; margin-right: 6px; }}
    code {{ background: #eef1f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>MaskWAM Correction Tasks</h1>
  <div class="summary">
    <p><strong>Tasks:</strong> {report['task_count']} | <strong>Samples:</strong> {report['sample_with_task_count']} / {report['sample_count']}</p>
    <p><strong>Priority:</strong> {html.escape(json.dumps(report['by_priority'], sort_keys=True))}</p>
    <p>Open the review sheet first, then use the raw frame links for click coordinates in <code>index.html</code>.</p>
  </div>
  {''.join(sections)}
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def review_paths(review_dir: Path | None, sample_name: str) -> Tuple[str, str]:
    if review_dir is None:
        return "", ""
    sheet = review_dir / "sheets" / f"{sample_name}_review.png"
    click_paths = sorted((review_dir / "click_frames").glob(f"{sample_name}_frame_*.png"))
    return str(sheet), " ".join(str(path) for path in click_paths)


def frame_index_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except Exception:
        return None


def mask_is_nonzero(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("L").getbbox() is not None
    except Exception:
        return False


def zero_frame_indices(mask_dir: str | Path, frame_count: int) -> List[int]:
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return []
    zeros = []
    for idx in range(frame_count):
        path = mask_dir / f"frame_{idx:03d}.png"
        if not path.exists() or not mask_is_nonzero(path):
            zeros.append(idx)
    return zeros


def review_frame_indices(clicks: str) -> List[int]:
    frame_indices = []
    for click_path in clicks.split():
        idx = frame_index_from_path(Path(click_path))
        if idx is not None:
            frame_indices.append(idx)
    return sorted(set(frame_indices))


def nearest_review_frame(frame_idx: int, review_frames: Sequence[int]) -> int:
    if not review_frames:
        return frame_idx
    return min(review_frames, key=lambda value: (abs(value - frame_idx), value))


def recommended_frame(
    zero_indices: Sequence[int],
    frame_count: int,
    clicks: str,
) -> Tuple[int, str]:
    review_frames = review_frame_indices(clicks)
    if not zero_indices:
        target = review_frames[len(review_frames) // 2] if review_frames else 0
        return target, "mask is nonzero on all frames; inspect the middle review frame for area quality"
    if len(zero_indices) >= frame_count:
        target = review_frames[len(review_frames) // 2] if review_frames else frame_count // 2
        return target, "mask is empty on all frames; inspect review frames and start from the first clearly visible frame"
    target = nearest_review_frame(int(zero_indices[0]), review_frames)
    return target, f"nearest review frame to earliest zero mask frame {int(zero_indices[0])}"


def task_reason(
    role: str,
    zero_count: int,
    frame_count: int,
    mean_area: float,
    min_area: float,
    max_area: float,
    include_robot_partial: bool,
) -> Tuple[str, str] | None:
    if role not in TRAIN_ROLES:
        return None
    if zero_count >= frame_count:
        return "critical", f"{role} mask is empty on all {frame_count} frames"
    if zero_count > 0:
        if base_role(role) == "robot" and not include_robot_partial:
            return "medium", f"{role} mask is missing on {zero_count}/{frame_count} frames"
        return "high", f"{role} mask is missing on {zero_count}/{frame_count} frames"
    if mean_area < min_area:
        return "medium", f"{role} mean area ratio is too small ({mean_area:.6f})"
    if mean_area > max_area:
        return "medium", f"{role} mean area ratio is too large ({mean_area:.6f})"
    return None


def suggested_action(priority: str, role: str, zero_count: int, frame_count: int) -> str:
    if zero_count >= frame_count:
        return f"Select prompt `{role}` and add a tight positive box or positive+negative points on a visible key frame."
    if zero_count > 0:
        return f"Add correction on the earliest missing/weak key frame; use negative points if SAM3 over-segments background."
    if priority == "medium":
        return "Inspect visually; add a tighter box or negative points if the mask is too broad or too small."
    return "Inspect visually."


def build_tasks(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_json(args.manifest)
    annotation_dir = Path(args.annotation_dir)
    review_dir = Path(args.review_dir) if args.review_dir else None
    tasks: List[Dict[str, Any]] = []
    samples_with_tasks = set()
    sample_id_filter = {int(value) for value in args.sample_ids} if args.sample_ids is not None else None

    for sample in manifest["samples"]:
        sample_id = int(sample["sample_id"])
        if sample_id_filter is not None and sample_id not in sample_id_filter:
            continue
        sample_name = sample["sample_name"]
        frame_count = len(sample["frame_indices"])
        sample_dir = annotation_dir / sample_name
        meta_path = sample_dir / "annotation_metadata.json"
        if not meta_path.exists():
            sheet, clicks = review_paths(review_dir, sample_name)
            tasks.append(
                {
                    "priority": "critical",
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "role": "",
                    "name": "",
                    "reason": "missing annotation_metadata.json",
                    "zero_frame_count": "",
                    "mean_area_ratio": "",
                    "zero_frame_indices": "",
                    "recommended_frame_index": "",
                    "recommended_frame_reason": "",
                    "suggested_action": "Rerun annotation for this sample.",
                    "task_text": sample.get("task_text", ""),
                    "review_sheet": sheet,
                    "click_frames": clicks,
                }
            )
            samples_with_tasks.add(sample_id)
            continue

        metadata = load_json(meta_path)
        sheet, clicks = review_paths(review_dir, sample_name)
        final_zero = int(metadata.get("final_train_mask", {}).get("zero_frame_count") or 0)
        if final_zero > 0:
            final_zero_indices = zero_frame_indices(sample_dir / "masks" / "final_train_mask", frame_count)
            final_recommended, final_recommended_reason = recommended_frame(final_zero_indices, frame_count, clicks)
            tasks.append(
                {
                    "priority": "critical",
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "role": FINAL_ROLE,
                    "name": FINAL_ROLE,
                    "reason": f"{FINAL_ROLE} has {final_zero}/{frame_count} zero frames",
                    "zero_frame_count": final_zero,
                    "mean_area_ratio": metadata.get(FINAL_ROLE, {}).get("mean_area_ratio", ""),
                    "zero_frame_indices": " ".join(str(idx) for idx in final_zero_indices),
                    "recommended_frame_index": final_recommended,
                    "recommended_frame_reason": final_recommended_reason,
                    "suggested_action": "Fix missing object/target/robot prompts until final_train_mask has no zero frames.",
                    "task_text": sample.get("task_text", ""),
                    "review_sheet": sheet,
                    "click_frames": clicks,
                }
            )
            samples_with_tasks.add(sample_id)

        for prompt in metadata.get("prompt_summaries", []):
            role = prompt.get("role", "")
            name = prompt.get("name", "")
            zero_count = int(prompt.get("zero_frame_count") or 0)
            mean_area = float(prompt.get("mean_area_ratio") or 0.0)
            reason = task_reason(
                role,
                zero_count,
                frame_count,
                mean_area,
                float(args.min_area_ratio),
                float(args.max_area_ratio),
                bool(args.include_robot_partial),
            )
            if reason is None:
                continue
            priority, reason_text = reason
            prompt_zero_indices = zero_frame_indices(prompt.get("mask_dir", ""), frame_count)
            prompt_recommended, prompt_recommended_reason = recommended_frame(prompt_zero_indices, frame_count, clicks)
            tasks.append(
                {
                    "priority": priority,
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                    "role": role,
                    "name": name,
                    "reason": reason_text,
                    "zero_frame_count": zero_count,
                    "mean_area_ratio": mean_area,
                    "zero_frame_indices": " ".join(str(idx) for idx in prompt_zero_indices),
                    "recommended_frame_index": prompt_recommended,
                    "recommended_frame_reason": prompt_recommended_reason,
                    "suggested_action": suggested_action(priority, role, zero_count, frame_count),
                    "task_text": sample.get("task_text", ""),
                    "review_sheet": sheet,
                    "click_frames": clicks,
                }
            )
            samples_with_tasks.add(sample_id)

    tasks.sort(
        key=lambda row: (
            PRIORITY_ORDER.get(str(row["priority"]), 99),
            int(row["sample_id"]),
            ROLE_ORDER.get(str(row["role"]), 99),
            str(row["name"]),
        )
    )
    by_priority: Dict[str, int] = {}
    for task in tasks:
        by_priority[task["priority"]] = by_priority.get(task["priority"], 0) + 1

    return {
        "schema_version": "maskwam_correction_tasks_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "annotation_dir": str(annotation_dir.resolve()),
        "review_dir": str(review_dir.resolve()) if review_dir else None,
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "sample_count": len([sample for sample in manifest["samples"] if sample_id_filter is None or int(sample["sample_id"]) in sample_id_filter]),
        "sample_with_task_count": len(samples_with_tasks),
        "task_count": len(tasks),
        "by_priority": by_priority,
        "tasks": tasks,
    }


def build_corrections_template(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    samples: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for task in tasks:
        role = task.get("role")
        name = task.get("name")
        if role not in TRAIN_ROLES or not name:
            continue
        key = (int(task["sample_id"]), role, name)
        if key in seen:
            continue
        seen.add(key)
        sample_key = str(task["sample_id"])
        sample = samples.setdefault(
            sample_key,
            {
                "status": "needs_correction",
                "notes": f"{task['sample_name']}: {task['task_text']}",
                "prompts": [],
            },
        )
        frame_index = int(task.get("recommended_frame_index") or 0)
        sample["prompts"].append(
            {
                "name": name,
                "role": role,
                "frame_index": frame_index,
                "points_abs": [],
                "point_labels": [],
                "boxes_xywh_abs": [],
                "box_labels": [],
                "todo_reason": task.get("reason", ""),
            }
        )
    return {
        "schema_version": "maskwam_corrections_v1",
        "notes": "Draft generated from first-pass correction tasks. Fill coordinates in review HTML before re-propagation.",
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    report = build_tasks(args)
    write_json(args.output_json, report)
    write_csv(args.output_csv, report["tasks"])
    if args.output_html:
        write_html(args.output_html, report)
    if args.output_corrections_template:
        write_json(args.output_corrections_template, build_corrections_template(report["tasks"]))
    print(args.output_json)
    print(args.output_csv)
    if args.output_html:
        print(args.output_html)
    if args.output_corrections_template:
        print(args.output_corrections_template)
    print(
        f"tasks={report['task_count']} samples={report['sample_with_task_count']}/{report['sample_count']} priorities={report['by_priority']}"
    )


if __name__ == "__main__":
    main()
