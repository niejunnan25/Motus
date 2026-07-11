#!/usr/bin/env python3
"""Verify MaskWAM-style LIBERO annotation outputs and review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from maskwam_view_schema import (
    FINAL_ROLE,
    OPTIONAL_BASE_ROLES,
    VIEW_TRAIN_ROLES,
    configured_base_train_roles,
    configured_output_roles,
    configured_view_train_roles,
)

TRAIN_ROLES = list(VIEW_TRAIN_ROLES)
OPTIONAL_ROLES = list(OPTIONAL_BASE_ROLES)
ALWAYS_REQUIRED_ROLES = [FINAL_ROLE]
APPROVED_STATUSES = {"ok", "approved", "yes", "y", "true", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--annotation_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass_qa16")
    parser.add_argument("--review_dir", default=None)
    parser.add_argument("--expected_samples", type=int, default=None)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument(
        "--require_nonzero_final",
        action="store_true",
        help="Treat final_train_mask zero frames as failures instead of human-correction flags.",
    )
    return parser.parse_args()


def select_samples(samples: List[Dict[str, Any]], sample_ids: List[int] | None, expected_samples: int | None) -> List[Dict[str, Any]]:
    if sample_ids is not None:
        wanted = {int(value) for value in sample_ids}
        samples = [sample for sample in samples if int(sample["sample_id"]) in wanted]
    if expected_samples is not None:
        samples = samples[:expected_samples]
    return samples


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def image_size(path: Path) -> List[int] | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return [image.width, image.height]


def configured_roles(sample: Dict[str, Any]) -> List[str]:
    # Keep base roles for backward-compatible visual/debug output, but verify
    # view-aware train roles explicitly so wrist failures cannot be hidden by
    # a good main-view mask.
    roles = configured_base_train_roles(sample)
    roles.extend(configured_output_roles(sample, include_context=True))
    seen = set()
    ordered = []
    for role in roles:
        if role not in seen:
            ordered.append(role)
            seen.add(role)
    return ordered


def configured_train_roles(sample: Dict[str, Any]) -> List[str]:
    return configured_view_train_roles(sample)


def verify_annotation_sample(annotation_dir: Path, sample: Dict[str, Any], require_nonzero_final: bool) -> Dict[str, Any]:
    sample_name = sample["sample_name"]
    frame_count = len(sample["frame_indices"])
    sample_dir = annotation_dir / sample_name
    metadata_path = sample_dir / "annotation_metadata.json"
    failures: List[str] = []
    role_counts: Dict[str, int] = {}

    if not metadata_path.exists():
        failures.append("missing annotation_metadata.json")
        metadata = {}
    else:
        metadata = load_json(metadata_path)

    rgb_count = count_pngs(sample_dir / "rgb")
    if rgb_count != frame_count:
        failures.append(f"rgb frame count {rgb_count} != {frame_count}")

    expected_size = [
        int(sample["target_size"]["width"]),
        int(sample["target_size"]["height"]),
    ]
    first_rgb_size = image_size(sample_dir / "rgb" / "frame_000.png")
    if first_rgb_size is not None and first_rgb_size != expected_size:
        failures.append(f"rgb size {first_rgb_size} != {expected_size}")

    expected_roles = configured_roles(sample)
    for role in expected_roles:
        count = count_pngs(sample_dir / "masks" / role)
        role_counts[role] = count
        if count != frame_count:
            failures.append(f"{role} mask count {count} != {frame_count}")
        first_mask_size = image_size(sample_dir / "masks" / role / "frame_000.png")
        if first_mask_size is not None and first_mask_size != expected_size:
            failures.append(f"{role} mask size {first_mask_size} != {expected_size}")

    final_zero = metadata.get("final_train_mask", {}).get("zero_frame_count")
    if require_nonzero_final and final_zero not in (0, "0", None):
        failures.append(f"final_train_mask zero_frame_count={final_zero}")

    role_summaries = metadata.get("role_summaries", {})
    role_zero_counts = {
        role: role_summaries.get(role, {}).get("zero_frame_count")
        for role in configured_train_roles(sample)
    }

    return {
        "sample": sample_name,
        "frame_count": frame_count,
        "expected_roles": expected_roles,
        "rgb_count": rgb_count,
        "role_counts": role_counts,
        "final_zero": final_zero,
        "role_zero_counts": role_zero_counts,
        "failures": failures,
    }


def verify_review(review_dir: Path | None, expected_samples: int) -> Dict[str, Any]:
    if review_dir is None:
        return {"checked": False}
    index = review_dir / "index.html"
    sheets = count_pngs(review_dir / "sheets")
    click_frames = count_pngs(review_dir / "click_frames")
    rows = read_csv(review_dir / "human_review_template.csv")
    diagnostics = review_dir / "auto_diagnostics.csv"
    corrections = review_dir / "corrections_template.json"
    failures = []
    if not index.exists():
        failures.append("missing index.html")
    if sheets != expected_samples:
        failures.append(f"sheet count {sheets} != {expected_samples}")
    if click_frames and click_frames != expected_samples * 5:
        failures.append(f"click frame count {click_frames} != {expected_samples * 5}")
    if rows and len(rows) != expected_samples:
        failures.append(f"human review rows {len(rows)} != {expected_samples}")
    if not diagnostics.exists():
        failures.append("missing auto_diagnostics.csv")
    if not corrections.exists():
        failures.append("missing corrections_template.json")
    approved = [
        row.get("sample", "")
        for row in rows
        if row.get("human_status", "").strip().lower() in APPROVED_STATUSES
    ]
    return {
        "checked": True,
        "index": str(index),
        "sheets": sheets,
        "click_frames": click_frames,
        "human_review_rows": len(rows),
        "approved": approved,
        "failures": failures,
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest)
    annotation_dir = Path(args.annotation_dir)
    manifest = load_json(manifest_path)
    samples = select_samples(manifest["samples"], args.sample_ids, args.expected_samples)
    expected_samples = len(samples)
    sample_reports = [
        verify_annotation_sample(annotation_dir, sample, args.require_nonzero_final)
        for sample in samples
    ]
    failures = []
    missing_or_invalid = [report for report in sample_reports if report["failures"]]
    for report in missing_or_invalid:
        failures.extend(f"{report['sample']}: {failure}" for failure in report["failures"])
    review_report = verify_review(Path(args.review_dir) if args.review_dir else None, expected_samples)
    failures.extend(review_report.get("failures", []))

    samples_with_role_zero = [
        {
            "sample": report["sample"],
            "role_zero_counts": report["role_zero_counts"],
        }
        for report in sample_reports
        if any(value not in (0, "0", None) for value in report["role_zero_counts"].values())
    ]
    samples_with_final_zero = [
        {
            "sample": report["sample"],
            "final_zero": report["final_zero"],
        }
        for report in sample_reports
        if report["final_zero"] not in (0, "0", None)
    ]
    if failures:
        status = "invalid"
    elif samples_with_role_zero or samples_with_final_zero:
        status = "needs_human_corrections"
    elif review_report.get("checked") and len(review_report.get("approved", [])) == expected_samples:
        status = "ready_for_larger_validation"
    else:
        status = "valid_unapproved"

    return {
        "status": status,
        "manifest": str(manifest_path.resolve()),
        "annotation_dir": str(annotation_dir.resolve()),
        "expected_samples": expected_samples,
        "sample_count": len(sample_reports),
        "samples_with_role_zero": samples_with_role_zero,
        "samples_with_final_zero": samples_with_final_zero,
        "review": review_report,
        "failures": failures,
    }


def write_markdown(report: Dict[str, Any], output_md: Path) -> None:
    lines = [
        "### MaskWAM Annotation Verification",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Annotation dir: `{report['annotation_dir']}`",
        f"Sample count: `{report['sample_count']}` / `{report['expected_samples']}`",
        "",
        "### Role Zero Frames",
        "",
    ]
    if report["samples_with_role_zero"]:
        for sample in report["samples_with_role_zero"]:
            lines.append(f"- `{sample['sample']}`: `{sample['role_zero_counts']}`")
    else:
        lines.append("- none")
    lines.extend(["", "### Final Mask Zero Frames", ""])
    if report["samples_with_final_zero"]:
        for sample in report["samples_with_final_zero"]:
            lines.append(f"- `{sample['sample']}`: `{sample['final_zero']}`")
    else:
        lines.append("- none")
    lines.extend(["", "### Review", ""])
    review = report["review"]
    if review.get("checked"):
        lines.extend(
            [
                f"- index: `{review['index']}`",
                f"- sheets: `{review['sheets']}`",
                f"- click frames: `{review['click_frames']}`",
                f"- human review rows: `{review['human_review_rows']}`",
                f"- approved: `{len(review['approved'])}`",
            ]
        )
    else:
        lines.append("- not checked")
    lines.extend(["", "### Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- none")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(output_json)
    if args.output_md:
        write_markdown(report, Path(args.output_md))
        print(args.output_md)
    print(f"status={report['status']}")


if __name__ == "__main__":
    main()
