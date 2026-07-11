#!/usr/bin/env python3
"""Prepare review HTML corrections for MaskWAM SAM3 re-propagation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from maskwam_check_correction_coverage import (
    build_report as build_coverage_report,
    csv_rows as coverage_csv_rows,
    write_csv as write_coverage_csv,
)
from maskwam_merge_corrections import merge_payloads
from maskwam_validate_corrections import build_report as build_validation_report


DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--downloaded_corrections", default=None)
    parser.add_argument("--draft_corrections", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output_corrections", default=None)
    parser.add_argument("--output_coverage_json", default=None)
    parser.add_argument("--output_coverage_csv", default=None)
    parser.add_argument("--output_validation_json", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--allow_empty", action="store_true")
    parser.add_argument(
        "--allow_template_fallback",
        action="store_true",
        help="Allow using corrections_template.json when no browser export is found. Intended only for smoke tests.",
    )
    parser.add_argument("--require_all_critical", action="store_true")
    parser.add_argument("--require_all_high", action="store_true")
    parser.add_argument("--require_all_actionable", action="store_true")
    parser.add_argument("--next_run_name", default="repropagated_qa16")
    parser.add_argument("--next_output_root", default="artifacts/maskwam_libero_annotation_debug16/corrected_runs")
    parser.add_argument("--next_python", default="/mnt/workspace1/users/niejunnan/envs/sam3/bin/python")
    parser.add_argument("--next_cuda_visible_devices", default="")
    parser.add_argument("--next_sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def existing_paths(paths: Sequence[Path]) -> List[Path]:
    return [path for path in paths if path.exists()]


def newest(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def find_downloaded_corrections(review_dir: Path, explicit: str | None, allow_template_fallback: bool = False) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"downloaded corrections not found: {path}")
        return path

    review_export = review_dir / "corrections_from_html.json"
    if review_export.exists():
        return review_export

    download_candidates = [
        Path(path)
        for pattern in [
            str(Path.home() / "Downloads" / "corrections_from_html.json"),
            str(Path.home() / "Downloads" / "corrections_from_html (*).json"),
        ]
        for path in glob.glob(pattern)
    ]
    selected_download = newest(existing_paths(download_candidates))
    if selected_download is not None:
        return selected_download

    if allow_template_fallback:
        template = review_dir / "corrections_template.json"
        if template.exists():
            return template

    raise FileNotFoundError(
        "No corrections JSON found. Pass --downloaded_corrections or save corrections_from_html.json into the review directory."
    )


def count_active_prompts(corrections: Dict[str, Any]) -> int:
    count = 0
    for sample in corrections.get("samples", {}).values():
        for prompt in sample.get("prompts", []):
            if (
                prompt.get("points_abs")
                or prompt.get("points_rel")
                or prompt.get("boxes_xywh_abs")
                or prompt.get("boxes_xywh_rel")
                or prompt.get("accepted_absent")
            ):
                count += 1
    return count


def uncovered_count(coverage: Dict[str, Any], priority: str | None = None) -> int:
    rows = coverage.get("uncovered_actionable", [])
    if priority is None:
        return len(rows)
    return sum(1 for row in rows if row.get("priority") == priority)


def build_next_command(args: argparse.Namespace, output_corrections: Path) -> str:
    sample_args = ""
    if args.sample_ids:
        sample_args = " \\\n  --sample_ids " + " ".join(str(value) for value in args.sample_ids)
    coverage_args = ["  --coverage_require_all_critical \\"]
    if args.require_all_high:
        coverage_args.append("  --coverage_require_all_high \\")
    if args.require_all_actionable:
        coverage_args.append("  --coverage_require_all_actionable \\")
    lines = [
        f"{args.next_python} \\",
        "  scripts/maskwam_run_corrected_pipeline.py \\",
        f"  --manifest {args.manifest} \\",
        f"  --corrections {output_corrections} \\",
        f"  --correction_tasks {args.tasks or str(Path(args.review_dir) / 'correction_tasks.json')} \\",
        *coverage_args,
        f"  --run_name {args.next_run_name} \\",
        f"  --output_root {args.next_output_root} \\",
        f"  --python {args.next_python} \\",
    ]
    if args.next_cuda_visible_devices:
        lines.append(f"  --cuda_visible_devices {args.next_cuda_visible_devices} \\")
    lines.append(f"  --sam3_version {args.next_sam3_version}" + sample_args)
    return "\n".join(lines)


def should_fail(args: argparse.Namespace, validation: Dict[str, Any], coverage: Dict[str, Any]) -> List[str]:
    failures = []
    if validation["status"] != "valid":
        failures.append("validation is invalid")
    if not args.allow_empty and validation["active_prompt_count"] == 0:
        failures.append("no active correction prompts found")
    if args.require_all_critical and uncovered_count(coverage, "critical") > 0:
        failures.append(f"{uncovered_count(coverage, 'critical')} uncovered critical actionable task(s)")
    if args.require_all_high and uncovered_count(coverage, "high") > 0:
        failures.append(f"{uncovered_count(coverage, 'high')} uncovered high actionable task(s)")
    if args.require_all_actionable and uncovered_count(coverage) > 0:
        failures.append(f"{uncovered_count(coverage)} uncovered actionable task(s)")
    return failures


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    manifest_path = Path(args.manifest)
    tasks_path = Path(args.tasks) if args.tasks else review_dir / "correction_tasks.json"
    draft_path = Path(args.draft_corrections) if args.draft_corrections else review_dir / "corrections_tasks_draft.json"
    output_corrections = Path(args.output_corrections) if args.output_corrections else review_dir / "corrections_merged.json"
    output_coverage_json = Path(args.output_coverage_json) if args.output_coverage_json else review_dir / "corrections_merged_coverage.json"
    output_coverage_csv = Path(args.output_coverage_csv) if args.output_coverage_csv else review_dir / "corrections_merged_coverage.csv"
    output_validation_json = Path(args.output_validation_json) if args.output_validation_json else review_dir / "corrections_merged_validation.json"
    summary_json = Path(args.summary_json) if args.summary_json else review_dir / "corrections_prepare_summary.json"

    downloaded_path = find_downloaded_corrections(
        review_dir,
        args.downloaded_corrections,
        allow_template_fallback=bool(args.allow_template_fallback),
    )
    merge_inputs = [path for path in [draft_path, downloaded_path] if path.exists()]
    if not merge_inputs:
        raise FileNotFoundError("No merge inputs found.")

    merged = merge_payloads(
        merge_inputs,
        keep_empty=False,
        notes="Prepared MaskWAM corrections for SAM3 re-propagation.",
    )
    write_json(output_corrections, merged)

    manifest = load_json(manifest_path)
    tasks_payload = load_json(tasks_path)
    validation = build_validation_report(manifest, merged, require_active=not args.allow_empty)
    validation["manifest"] = str(manifest_path.resolve())
    validation["corrections"] = str(output_corrections.resolve())
    write_json(output_validation_json, validation)

    coverage = build_coverage_report(
        tasks_payload,
        merged,
        allow_manual_role_match=True,
        sample_ids=args.sample_ids,
    )
    coverage["tasks"] = str(tasks_path.resolve())
    coverage["corrections"] = str(output_corrections.resolve())
    write_json(output_coverage_json, coverage)
    write_coverage_csv(output_coverage_csv, coverage_csv_rows(coverage))

    failures = should_fail(args, validation, coverage)
    ready_for_repropagate = not failures
    summary = {
        "status": "ready" if ready_for_repropagate else "needs_more_corrections",
        "ready_for_repropagate": ready_for_repropagate,
        "failures": failures,
        "review_dir": str(review_dir.resolve()),
        "downloaded_corrections": str(downloaded_path.resolve()),
        "draft_corrections": str(draft_path.resolve()) if draft_path.exists() else None,
        "output_corrections": str(output_corrections.resolve()),
        "output_coverage_json": str(output_coverage_json.resolve()),
        "output_coverage_csv": str(output_coverage_csv.resolve()),
        "output_validation_json": str(output_validation_json.resolve()),
        "active_prompt_count": count_active_prompts(merged),
        "validation_status": validation["status"],
        "validation_issue_count": len(validation.get("issues", [])),
        "coverage_summary": coverage["summary"],
        "uncovered_critical_count": uncovered_count(coverage, "critical"),
        "uncovered_high_count": uncovered_count(coverage, "high"),
        "uncovered_actionable_count": uncovered_count(coverage),
        "next_repropagate_command": build_next_command(args, output_corrections),
    }
    write_json(summary_json, summary)

    print(f"merged={output_corrections}")
    print(f"validation={validation['status']} active={validation['active_prompt_count']} issues={len(validation.get('issues', []))}")
    print(
        "coverage actionable={covered}/{total} critical_uncovered={critical} high_uncovered={high} unmatched={unmatched}".format(
            covered=coverage["summary"]["covered_actionable_task_count"],
            total=coverage["summary"]["actionable_task_count"],
            critical=summary["uncovered_critical_count"],
            high=summary["uncovered_high_count"],
            unmatched=coverage["summary"]["unmatched_active_correction_count"],
        )
    )
    print(f"summary={summary_json}")
    if ready_for_repropagate:
        print("ready_for_repropagate=true")
        print(summary["next_repropagate_command"])
    else:
        print("ready_for_repropagate=false")
    if failures:
        for failure in failures:
            print(f"ISSUE {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
