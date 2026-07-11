#!/usr/bin/env python3
"""Export MaskWAM-style annotation outputs as trainable mask-sequence manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from maskwam_view_schema import (
    FINAL_ROLE,
    OPTIONAL_BASE_ROLES,
    VIEW_TRAIN_ROLES,
    configured_base_train_roles,
    configured_output_roles,
    configured_view_train_roles,
)

APPROVED_STATUSES = {"ok", "approved", "yes", "y", "true", "1"}
ROLE_MASKS = ["object", "target", "robot", *VIEW_TRAIN_ROLES, *OPTIONAL_BASE_ROLES, FINAL_ROLE]
TRAIN_ROLES = list(VIEW_TRAIN_ROLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--review_csv", default=None)
    parser.add_argument("--require_review_ok", action="store_true")
    parser.add_argument("--allow_final_zero", action="store_true")
    parser.add_argument(
        "--allow_role_zero",
        action="store_true",
        help="Allow configured view-aware train roles to have zero frames. Default blocks formal training export.",
    )
    parser.add_argument("--allow_missing_roles", action="store_true")
    parser.add_argument(
        "--allow_unapproved_fallbacks",
        action="store_true",
        help="Allow fallback/manual box masks without an approved human review row. Default blocks formal training export.",
    )
    parser.add_argument("--warn_min_area_ratio", type=float, default=0.001)
    parser.add_argument("--warn_max_area_ratio", type=float, default=0.5)
    parser.add_argument("--path_mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--relative_to", default=".")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def read_review_statuses(path: str | Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    review_path = Path(path)
    if not review_path.exists():
        raise FileNotFoundError(f"review_csv does not exist: {review_path}")
    with open(review_path, "r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            row.get("sample", ""): row.get("human_status", "").strip().lower()
            for row in rows
            if row.get("sample")
        }


def format_path(path: Path, path_mode: str, relative_to: Path) -> str:
    resolved = path.resolve()
    if path_mode == "absolute":
        return str(resolved)
    return str(resolved.relative_to(relative_to.resolve()))


def png_sequence(directory: Path, frame_count: int) -> List[Path]:
    return [directory / f"frame_{idx:03d}.png" for idx in range(frame_count)]


def image_size(path: Path) -> List[int] | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return [image.width, image.height]


def mask_area(path: Path) -> int:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    return int(mask.sum())


def select_samples(samples: Sequence[Dict[str, Any]], sample_ids: Sequence[int] | None) -> List[Dict[str, Any]]:
    if sample_ids is None:
        return list(samples)
    wanted = {int(value) for value in sample_ids}
    return [sample for sample in samples if int(sample["sample_id"]) in wanted]


def configured_train_roles(sample: Dict[str, Any]) -> List[str]:
    return configured_view_train_roles(sample)


def configured_role_masks(sample: Dict[str, Any]) -> List[str]:
    roles = configured_base_train_roles(sample)
    roles.extend(configured_output_roles(sample, include_context=True))
    seen = set()
    ordered = []
    for role in roles:
        if role not in seen:
            ordered.append(role)
            seen.add(role)
    return ordered


def role_accepted_absent(metadata: Dict[str, Any], role: str, frame_count: int) -> bool:
    role_summary = metadata.get("role_summaries", {}).get(role, {})
    try:
        zero_count = int(role_summary.get("zero_frame_count", -1))
    except (TypeError, ValueError):
        return False
    if zero_count != int(frame_count):
        return False
    prompt_summaries = [
        item
        for item in metadata.get("prompt_summaries", [])
        if item.get("role") == role and item.get("view_derived")
    ]
    if not prompt_summaries:
        return False
    return all(bool(item.get("accepted_absent")) for item in prompt_summaries)


def build_sample_entry(
    sample: Dict[str, Any],
    annotation_dir: Path,
    review_statuses: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sample_name = sample["sample_name"]
    sample_dir = annotation_dir / sample_name
    frame_count = len(sample["frame_indices"])
    expected_size = [
        int(sample["target_size"]["width"]),
        int(sample["target_size"]["height"]),
    ]
    issues: List[str] = []
    warnings: List[str] = []
    review_status = review_statuses.get(sample_name, "")

    metadata_path = sample_dir / "annotation_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    if not metadata_path.exists():
        issues.append("missing annotation_metadata.json")

    rgb_paths = png_sequence(sample_dir / "rgb", frame_count)
    final_paths = png_sequence(sample_dir / "masks" / FINAL_ROLE, frame_count)
    role_paths = {
        role: png_sequence(sample_dir / "masks" / role, frame_count)
        for role in configured_role_masks(sample)
    }

    for path in rgb_paths:
        if not path.exists():
            issues.append(f"missing rgb frame: {path.name}")
    for path in final_paths:
        if not path.exists():
            issues.append(f"missing {FINAL_ROLE} frame: {path.name}")
    if not args.allow_missing_roles:
        for role, paths in role_paths.items():
            for path in paths:
                if not path.exists():
                    issues.append(f"missing {role} frame: {path.name}")

    for path in rgb_paths[:1] + final_paths[:1]:
        size = image_size(path)
        if size is not None and size != expected_size:
            issues.append(f"{path} size {size} != {expected_size}")

    final_areas: List[int] = []
    if all(path.exists() for path in final_paths):
        final_areas = [mask_area(path) for path in final_paths]
        zero_frames = [idx for idx, area in enumerate(final_areas) if area == 0]
        if zero_frames and not args.allow_final_zero:
            issues.append(f"{FINAL_ROLE} has zero frames: {zero_frames}")
        elif zero_frames:
            warnings.append(f"{FINAL_ROLE} has zero frames: {zero_frames}")
    else:
        zero_frames = []

    role_zero_counts = metadata.get("role_summaries", {})
    for role in configured_train_roles(sample):
        zero_count = role_zero_counts.get(role, {}).get("zero_frame_count")
        if zero_count not in (0, "0", None):
            message = f"{role} zero_frame_count={zero_count}"
            if role_accepted_absent(metadata, role, frame_count):
                warnings.append(f"{message}; human accepted role as not visible")
            elif args.allow_role_zero:
                warnings.append(message)
            else:
                issues.append(message)

    prompt_fallbacks = [
        {
            "role": item.get("role"),
            "name": item.get("name"),
            "view": item.get("view", ""),
            "base_role": item.get("base_role", ""),
            "fallback_reason": item.get("fallback_reason", ""),
        }
        for item in metadata.get("prompt_summaries", [])
        if item.get("fallback_used") and not item.get("view_derived")
    ]
    if prompt_fallbacks:
        message = f"{len(prompt_fallbacks)} prompt(s) used fallback"
        if args.allow_unapproved_fallbacks or review_status in APPROVED_STATUSES:
            warnings.append(message)
        else:
            issues.append(f"{message}; requires approved human review or --allow_unapproved_fallbacks")

    if args.require_review_ok and review_status not in APPROVED_STATUSES:
        issues.append(f"review status is not approved: {review_status or '<empty>'}")

    relative_to = Path(args.relative_to)
    path_mode = args.path_mode
    area_ratio = None
    if final_areas:
        pixel_count = int(sample["target_size"]["width"]) * int(sample["target_size"]["height"])
        area_ratio = float(np.mean(final_areas) / float(pixel_count))
        if area_ratio < float(args.warn_min_area_ratio):
            warnings.append(f"final_mean_area_ratio too small: {area_ratio:.6f}")
        if area_ratio > float(args.warn_max_area_ratio):
            warnings.append(f"final_mean_area_ratio too large: {area_ratio:.6f}")

    return {
        "sample_id": int(sample["sample_id"]),
        "sample_name": sample_name,
        "task_text": sample.get("task_text", ""),
        "frame_indices": sample["frame_indices"],
        "source_video_paths": sample["video_paths"],
        "frame_count": frame_count,
        "target_size": sample["target_size"],
        "rgb_paths": [format_path(path, path_mode, relative_to) for path in rgb_paths],
        "mask_paths": [format_path(path, path_mode, relative_to) for path in final_paths],
        "role_mask_paths": {
            role: [format_path(path, path_mode, relative_to) for path in paths]
            for role, paths in role_paths.items()
        },
        "metadata_path": format_path(metadata_path, path_mode, relative_to),
        "review_status": review_status,
        "final_zero_frames": zero_frames,
        "final_mean_area_ratio": area_ratio,
        "prompt_fallbacks": prompt_fallbacks,
        "issues": issues,
        "warnings": warnings,
        "training_ready": not issues,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    annotation_dir = Path(args.annotation_dir)
    manifest = load_json(manifest_path)
    review_statuses = read_review_statuses(args.review_csv)
    samples = select_samples(manifest["samples"], args.sample_ids)
    entries = [
        build_sample_entry(sample, annotation_dir, review_statuses, args)
        for sample in samples
    ]
    ready = [entry for entry in entries if entry["training_ready"]]
    issue_entries = [entry for entry in entries if entry["issues"]]
    warning_entries = [entry for entry in entries if entry["warnings"]]
    payload = {
        "schema_version": "maskwam_train_mask_sequences_v1",
        "view_schema": {
            "enabled": True,
            "train_roles": list(VIEW_TRAIN_ROLES),
            "final_train_mask": "union of view-aware object/target/robot roles",
            "fallback_policy": "fallback/manual masks require approved human review unless --allow_unapproved_fallbacks is set",
        },
        "manifest": str(manifest_path.resolve()),
        "annotation_dir": str(annotation_dir.resolve()),
        "path_mode": args.path_mode,
        "relative_to": str(Path(args.relative_to).resolve()),
        "require_review_ok": bool(args.require_review_ok),
        "allow_final_zero": bool(args.allow_final_zero),
        "allow_role_zero": bool(args.allow_role_zero),
        "allow_unapproved_fallbacks": bool(args.allow_unapproved_fallbacks),
        "warn_min_area_ratio": float(args.warn_min_area_ratio),
        "warn_max_area_ratio": float(args.warn_max_area_ratio),
        "sample_count": len(entries),
        "ready_count": len(ready),
        "issue_count": len(issue_entries),
        "warning_count": len(warning_entries),
        "status": "ready" if len(ready) == len(entries) else "not_ready",
        "samples": entries,
    }
    write_json(args.output_json, payload)
    print(args.output_json)
    print(f"status={payload['status']} ready={payload['ready_count']}/{payload['sample_count']} issues={payload['issue_count']} warnings={payload['warning_count']}")
    if issue_entries:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
