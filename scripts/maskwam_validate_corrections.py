#!/usr/bin/env python3
"""Validate MaskWAM correction JSON before re-running SAM3 propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from maskwam_view_schema import VALID_CORRECTION_ROLES, base_role


VALID_ROLES = VALID_CORRECTION_ROLES
VALID_LABELS = {0, 1}
MANUAL_NAMES = {None, "", "manual point", "manual box"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--require_active", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_point(point: Any, width: int, height: int) -> List[str]:
    issues = []
    if not isinstance(point, list) or len(point) != 2 or not all(is_number(value) for value in point):
        return [f"point must be [x, y], got {point!r}"]
    x, y = float(point[0]), float(point[1])
    if x < 0 or x > width or y < 0 or y > height:
        issues.append(f"point out of bounds for {width}x{height}: {point!r}")
    return issues


def validate_box(box: Any, width: int, height: int) -> List[str]:
    issues = []
    if not isinstance(box, list) or len(box) != 4 or not all(is_number(value) for value in box):
        return [f"box must be [x, y, w, h], got {box!r}"]
    x, y, w, h = [float(value) for value in box]
    if w <= 0 or h <= 0:
        issues.append(f"box width/height must be positive: {box!r}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        issues.append(f"box out of bounds for {width}x{height}: {box!r}")
    return issues


def validate_labels(labels: Any, expected_count: int, label_name: str) -> List[str]:
    if expected_count == 0 and labels in (None, []):
        return []
    if labels is None:
        return [f"{label_name} missing for {expected_count} item(s)"]
    if not isinstance(labels, list):
        return [f"{label_name} must be a list"]
    issues = []
    if len(labels) != expected_count:
        issues.append(f"{label_name} count {len(labels)} != {expected_count}")
    for value in labels:
        if value not in VALID_LABELS:
            issues.append(f"{label_name} values must be 0 or 1, got {value!r}")
    return issues


def prompt_matches(entry: Dict[str, Any], sample_prompts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    name = entry.get("name")
    role = entry.get("role")
    prompt_role = base_role(role)
    if name in MANUAL_NAMES:
        return [prompt for prompt in sample_prompts if prompt.get("role") == prompt_role]
    return [
        prompt
        for prompt in sample_prompts
        if prompt.get("name") == name and prompt.get("role") == prompt_role
    ]


def validate_entry(
    sample_id: int,
    prompt_idx: int,
    entry: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    width = int(sample["target_size"]["width"])
    height = int(sample["target_size"]["height"])
    frame_count = len(sample["frame_indices"])
    issues: List[str] = []
    warnings: List[str] = []

    if not isinstance(entry, dict):
        return {
            "sample_id": sample_id,
            "prompt_idx": prompt_idx,
            "active": False,
            "matches": [],
            "issues": [f"prompt entry must be an object, got {type(entry).__name__}"],
            "warnings": [],
        }

    role = entry.get("role")
    if role not in VALID_ROLES:
        issues.append(f"invalid role={role!r}")

    frame_index = entry.get("frame_index", 0)
    if not isinstance(frame_index, int) or isinstance(frame_index, bool):
        issues.append(f"frame_index must be int, got {frame_index!r}")
    elif frame_index < 0 or frame_index >= frame_count:
        issues.append(f"frame_index out of range [0, {frame_count - 1}]: {frame_index}")

    points_abs = entry.get("points_abs") or []
    boxes_abs = entry.get("boxes_xywh_abs") or []
    points_rel = entry.get("points_rel") or []
    boxes_rel = entry.get("boxes_xywh_rel") or []
    accepted_absent = bool(entry.get("accepted_absent"))
    active = bool(points_abs or boxes_abs or points_rel or boxes_rel or accepted_absent)

    if points_abs and not isinstance(points_abs, list):
        issues.append("points_abs must be a list")
        points_abs = []
    if boxes_abs and not isinstance(boxes_abs, list):
        issues.append("boxes_xywh_abs must be a list")
        boxes_abs = []
    if points_rel and not isinstance(points_rel, list):
        issues.append("points_rel must be a list")
        points_rel = []
    if boxes_rel and not isinstance(boxes_rel, list):
        issues.append("boxes_xywh_rel must be a list")
        boxes_rel = []

    for point in points_abs:
        issues.extend(validate_point(point, width, height))
    for box in boxes_abs:
        issues.extend(validate_box(box, width, height))
    for point in points_rel:
        issues.extend(validate_point(point, 1, 1))
    for box in boxes_rel:
        issues.extend(validate_box(box, 1, 1))

    point_count = len(points_abs) + len(points_rel)
    box_count = len(boxes_abs) + len(boxes_rel)
    issues.extend(validate_labels(entry.get("point_labels"), point_count, "point_labels"))
    issues.extend(validate_labels(entry.get("box_labels"), box_count, "box_labels"))

    matches = prompt_matches(entry, sample["prompts"]) if role in VALID_ROLES else []
    match_names = [f"{prompt['role']}:{prompt['name']}" for prompt in matches]
    if active and not matches:
        issues.append(
            f"active correction will not match any configured prompt; name={entry.get('name')!r}, role={role!r}"
        )
    elif active and entry.get("name") in MANUAL_NAMES and len(matches) > 1:
        warnings.append(f"manual role correction matches {len(matches)} prompts: {match_names}")

    return {
        "sample_id": sample_id,
        "sample_name": sample["sample_name"],
        "prompt_idx": prompt_idx,
        "name": entry.get("name"),
        "role": role,
        "frame_index": frame_index,
        "active": active,
        "match_count": len(matches),
        "matches": match_names,
        "point_count": point_count,
        "box_count": box_count,
        "accepted_absent": accepted_absent,
        "issues": issues,
        "warnings": warnings,
    }


def build_report(manifest: Dict[str, Any], corrections: Dict[str, Any], require_active: bool) -> Dict[str, Any]:
    samples_by_id = {str(sample["sample_id"]): sample for sample in manifest["samples"]}
    entries = []
    issues: List[str] = []
    warnings: List[str] = []

    correction_samples = corrections.get("samples")
    if not isinstance(correction_samples, dict):
        return {
            "status": "invalid",
            "issues": ["corrections.samples must be an object"],
            "warnings": [],
            "active_prompt_count": 0,
            "prompt_count": 0,
            "entries": [],
        }

    for sample_key, sample_cfg in correction_samples.items():
        if sample_key not in samples_by_id:
            issues.append(f"unknown sample id: {sample_key}")
            continue
        sample = samples_by_id[sample_key]
        prompts = sample_cfg.get("prompts", []) if isinstance(sample_cfg, dict) else []
        if not isinstance(prompts, list):
            issues.append(f"sample {sample_key} prompts must be a list")
            continue
        for prompt_idx, entry in enumerate(prompts):
            report = validate_entry(int(sample_key), prompt_idx, entry, sample)
            entries.append(report)
            issues.extend(f"sample {sample_key} prompt {prompt_idx}: {issue}" for issue in report["issues"])
            warnings.extend(f"sample {sample_key} prompt {prompt_idx}: {warning}" for warning in report["warnings"])

    active_prompt_count = sum(1 for entry in entries if entry["active"])
    if require_active and active_prompt_count == 0:
        issues.append("no active correction prompts found")

    return {
        "status": "valid" if not issues else "invalid",
        "sample_count": len(correction_samples),
        "prompt_count": len(entries),
        "active_prompt_count": active_prompt_count,
        "matched_active_prompt_count": sum(1 for entry in entries if entry["active"] and entry["match_count"] > 0),
        "issues": issues,
        "warnings": warnings,
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    corrections_path = Path(args.corrections)
    manifest = load_json(manifest_path)
    corrections = load_json(corrections_path)
    report = build_report(manifest, corrections, args.require_active)
    report["manifest"] = str(manifest_path.resolve())
    report["corrections"] = str(corrections_path.resolve())
    if args.output_json:
        write_json(args.output_json, report)
    print(f"status={report['status']} active={report['active_prompt_count']} matched_active={report['matched_active_prompt_count']} issues={len(report['issues'])} warnings={len(report['warnings'])}")
    if args.output_json:
        print(args.output_json)
    if report["issues"]:
        for issue in report["issues"][:20]:
            print(f"ISSUE {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
