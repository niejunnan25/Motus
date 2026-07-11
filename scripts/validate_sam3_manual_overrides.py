#!/usr/bin/env python3
"""Validate SAM3 manual override JSON before running segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manual_overrides", help="Manual override JSON to validate.")
    parser.add_argument("--issue_frames_csv", default=None, help="Optional issue_frames.csv to check frame coverage.")
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=448)
    parser.add_argument("--frame_count", type=int, default=17)
    parser.add_argument(
        "--require_issue_coverage",
        action="store_true",
        help="Treat missing manual coverage for issue frames as errors instead of warnings.",
    )
    parser.add_argument("--output_json", default=None, help="Optional validation report JSON path.")
    return parser.parse_args()


def load_manual_overrides(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    samples = data.get("samples", data)
    if not isinstance(samples, dict):
        raise TypeError(f"Expected top-level samples dict, got {type(samples).__name__}")
    return samples


def load_issue_frame_map(path: str | None) -> Dict[int, List[int]]:
    if not path:
        return {}
    issue_map: Dict[int, List[int]] = {}
    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sample = row.get("sample", "")
            if not sample.startswith("sample_"):
                continue
            sample_id = int(sample.split("_", 1)[1])
            frames = [int(token) for token in row.get("issue_frames", "").split() if token.strip()]
            if frames:
                issue_map[sample_id] = frames
    return issue_map


def cxcywh_norm_to_xyxy_abs(box: Sequence[float], image_w: int, image_h: int) -> List[float]:
    cx, cy, w, h = [float(v) for v in box]
    return [
        (cx - 0.5 * w) * image_w,
        (cy - 0.5 * h) * image_h,
        (cx + 0.5 * w) * image_w,
        (cy + 0.5 * h) * image_h,
    ]


def xyxy_norm_to_xyxy_abs(box: Sequence[float], image_w: int, image_h: int) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return [x0 * image_w, y0 * image_h, x1 * image_w, y1 * image_h]


def entry_box_to_xyxy_abs(entry: Dict, image_w: int, image_h: int) -> List[float]:
    if "xyxy_abs" in entry:
        return [float(v) for v in entry["xyxy_abs"]]
    if "xyxy_norm" in entry:
        return xyxy_norm_to_xyxy_abs(entry["xyxy_norm"], image_w, image_h)
    if "cxcywh_norm" in entry:
        return cxcywh_norm_to_xyxy_abs(entry["cxcywh_norm"], image_w, image_h)
    raise KeyError("entry must contain xyxy_abs, xyxy_norm, or cxcywh_norm")


def validate_box(
    box: Sequence[float],
    image_w: int,
    image_h: int,
    context: str,
    errors: List[str],
    warnings: List[str],
) -> None:
    if len(box) != 4:
        errors.append(f"{context}: expected 4 box values, got {len(box)}")
        return
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 < x0:
        warnings.append(f"{context}: x1 < x0; code will swap, but config should be cleaned")
        x0, x1 = x1, x0
    if y1 < y0:
        warnings.append(f"{context}: y1 < y0; code will swap, but config should be cleaned")
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        errors.append(f"{context}: degenerate box {box}")
    if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
        errors.append(f"{context}: box {box} out of bounds x=[0,{image_w}], y=[0,{image_h}]")


def validate_frame(frame_idx: int, frame_count: int, context: str, errors: List[str]) -> None:
    if frame_idx < 0 or frame_idx >= frame_count:
        errors.append(f"{context}: frame {frame_idx} out of range [0,{frame_count - 1}]")


def frame_coverage(entry: Dict, frame_count: int) -> List[int]:
    if "frames" in entry:
        keyed = sorted(int(key) for key in entry["frames"])
        if not keyed:
            return []
        if entry.get("interpolate", True):
            return list(range(max(0, keyed[0]), min(frame_count - 1, keyed[-1]) + 1))
        return [idx for idx in keyed if 0 <= idx < frame_count]
    if "frame" in entry:
        frame_idx = int(entry["frame"])
        return [frame_idx] if 0 <= frame_idx < frame_count else []
    return list(range(frame_count))


def sample_entries(sample_cfg) -> List[Dict]:
    if not isinstance(sample_cfg, dict):
        return []
    entries = sample_cfg.get("objects", sample_cfg.get("entries", []))
    return entries if isinstance(entries, list) else []


def validate_samples(args: argparse.Namespace, samples: Dict, issue_map: Dict[int, List[int]]) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []
    sample_reports = {}

    for sample_key, sample_cfg in sorted(samples.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999999):
        sample_id = int(sample_key)
        entries = sample_entries(sample_cfg)
        covered_frames = set()
        entry_reports = []
        if not entries:
            warnings.append(f"sample_{sample_id:03d}: no manual entries")
        for entry_idx, entry in enumerate(entries):
            name = entry.get("name", f"manual_{entry_idx}")
            role = entry.get("role", "object")
            context = f"sample_{sample_id:03d}/{role}/{name}"
            coverage = frame_coverage(entry, args.frame_count)
            covered_frames.update(coverage)
            if not coverage:
                warnings.append(f"{context}: entry covers no valid frames")

            if "frames" in entry:
                if not isinstance(entry["frames"], dict):
                    errors.append(f"{context}: frames must be a dict")
                    continue
                for frame_key, frame_entry in entry["frames"].items():
                    frame_idx = int(frame_key)
                    validate_frame(frame_idx, args.frame_count, f"{context}/frame_{frame_idx}", errors)
                    try:
                        box = entry_box_to_xyxy_abs(frame_entry, args.image_width, args.image_height)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{context}/frame_{frame_idx}: {exc}")
                        continue
                    validate_box(box, args.image_width, args.image_height, f"{context}/frame_{frame_idx}", errors, warnings)
            else:
                if "frame" in entry:
                    validate_frame(int(entry["frame"]), args.frame_count, context, errors)
                try:
                    box = entry_box_to_xyxy_abs(entry, args.image_width, args.image_height)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{context}: {exc}")
                    continue
                validate_box(box, args.image_width, args.image_height, context, errors, warnings)

            entry_reports.append(
                {
                    "name": name,
                    "role": role,
                    "covered_frames": coverage,
                }
            )

        sample_reports[f"sample_{sample_id:03d}"] = {
            "entry_count": len(entries),
            "covered_frames": sorted(covered_frames),
            "entries": entry_reports,
        }

    for sample_id, issue_frames in sorted(issue_map.items()):
        sample_key = str(sample_id)
        entries = sample_entries(samples.get(sample_key, samples.get(sample_id, {})))
        covered = set()
        for entry in entries:
            covered.update(frame_coverage(entry, args.frame_count))
        missing = [idx for idx in issue_frames if idx not in covered]
        if not entries:
            message = f"sample_{sample_id:03d}: issue frames {issue_frames} have no manual entries"
        elif missing:
            message = f"sample_{sample_id:03d}: issue frames missing manual coverage {missing}"
        else:
            message = ""
        if message:
            if args.require_issue_coverage:
                errors.append(message)
            else:
                warnings.append(message)

    return {
        "manual_overrides": str(Path(args.manual_overrides).resolve()),
        "issue_frames_csv": str(Path(args.issue_frames_csv).resolve()) if args.issue_frames_csv else None,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "frame_count": args.frame_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "samples": sample_reports,
    }


def main() -> None:
    args = parse_args()
    samples = load_manual_overrides(args.manual_overrides)
    issue_map = load_issue_frame_map(args.issue_frames_csv)
    report = validate_samples(args, samples, issue_map)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"errors={report['error_count']} warnings={report['warning_count']}")
    for warning in report["warnings"]:
        print(f"WARNING {warning}")
    for error in report["errors"]:
        print(f"ERROR {error}")

    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
