#!/usr/bin/env python3
"""Audit a corrected MaskWAM run before using its masks for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ACCEPTED_VERIFICATION_STATUSES = {"valid_unapproved", "ready_for_larger_validation"}
APPROVED_VERIFICATION_STATUS = "ready_for_larger_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        default="artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16",
    )
    parser.add_argument("--expected_samples", type=int, default=16)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--allow_missing_coverage", action="store_true")
    parser.add_argument("--allow_sample_count_mismatch", action="store_true")
    parser.add_argument("--allow_role_zero", action="store_true")
    parser.add_argument("--allow_final_zero", action="store_true")
    parser.add_argument("--allow_train_warnings", action="store_true")
    parser.add_argument("--require_review_approved", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_markdown(path: str | Path, report: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "### MaskWAM Corrected Run Audit",
        "",
        f"Status: `{report['status']}`",
        f"Run dir: `{report['run_dir']}`",
        "",
        "### Summary",
        "",
        f"- pipeline status: `{summary.get('pipeline_status')}`",
        f"- verification status: `{summary.get('verification_status')}`",
        f"- train manifest status: `{summary.get('train_status')}`",
        f"- train ready: `{summary.get('train_ready_count')}/{summary.get('train_sample_count')}`",
        f"- uncovered critical tasks: `{summary.get('uncovered_critical_count')}`",
        f"- active correction prompts: `{summary.get('active_correction_count')}`",
        "",
        "### Issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in report["issues"]] or ["- none"])
    lines.extend(["", "### Warnings", ""])
    lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- none"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_load(path: Path, issues: List[str], label: str) -> Dict[str, Any] | None:
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return None
    try:
        return load_json(path)
    except Exception as exc:
        issues.append(f"could not read {label}: {path}: {exc}")
        return None


def count_uncovered_priority(coverage: Dict[str, Any] | None, priority: str) -> int | None:
    if coverage is None:
        return None
    return sum(1 for row in coverage.get("uncovered_actionable", []) if row.get("priority") == priority)


def count_role_zero(verification: Dict[str, Any] | None) -> int | None:
    if verification is None:
        return None
    return len(verification.get("samples_with_role_zero", []))


def count_final_zero(verification: Dict[str, Any] | None) -> int | None:
    if verification is None:
        return None
    return len(verification.get("samples_with_final_zero", []))


def sample_issue_count(train_manifest: Dict[str, Any] | None) -> int | None:
    if train_manifest is None:
        return None
    return sum(len(sample.get("issues", [])) for sample in train_manifest.get("samples", []))


def sample_warning_count(train_manifest: Dict[str, Any] | None) -> int | None:
    if train_manifest is None:
        return None
    return sum(len(sample.get("warnings", [])) for sample in train_manifest.get("samples", []))


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    issues: List[str] = []
    warnings: List[str] = []

    pipeline_path = run_dir / "pipeline_summary.json"
    coverage_path = run_dir / "correction_coverage.json"
    validation_path = run_dir / "correction_validation.json"
    verification_path = run_dir / "review" / "verification.json"
    train_path = run_dir / "train_mask_sequences.json"

    pipeline = maybe_load(pipeline_path, issues, "pipeline summary")
    validation = maybe_load(validation_path, issues, "correction validation")
    verification = maybe_load(verification_path, issues, "verification report")
    train_manifest = maybe_load(train_path, issues, "train mask manifest")

    coverage: Dict[str, Any] | None = None
    if coverage_path.exists():
        coverage = maybe_load(coverage_path, issues, "correction coverage")
    elif args.allow_missing_coverage:
        warnings.append(f"missing correction coverage: {coverage_path}")
    else:
        issues.append(f"missing correction coverage: {coverage_path}")

    if pipeline is not None and pipeline.get("status") != "complete":
        issues.append(f"pipeline status is not complete: {pipeline.get('status')}")
    if validation is not None and validation.get("status") != "valid":
        issues.append(f"correction validation is not valid: {validation.get('status')}")

    if coverage is not None:
        uncovered_critical = count_uncovered_priority(coverage, "critical") or 0
        if uncovered_critical:
            issues.append(f"uncovered critical actionable task(s): {uncovered_critical}")
        unmatched = coverage.get("summary", {}).get("unmatched_active_correction_count", 0)
        if unmatched:
            warnings.append(f"unmatched active correction prompt(s): {unmatched}")

    if verification is not None:
        verification_status = verification.get("status")
        allowed = {APPROVED_VERIFICATION_STATUS} if args.require_review_approved else ACCEPTED_VERIFICATION_STATUSES
        if verification_status not in allowed:
            issues.append(f"verification status {verification_status!r} not in {sorted(allowed)}")
        if not args.allow_sample_count_mismatch and int(verification.get("sample_count", -1)) != int(args.expected_samples):
            issues.append(f"verification sample_count {verification.get('sample_count')} != {args.expected_samples}")
        role_zero = count_role_zero(verification) or 0
        final_zero = count_final_zero(verification) or 0
        if role_zero and not args.allow_role_zero:
            issues.append(f"samples with role zero masks: {role_zero}")
        elif role_zero:
            warnings.append(f"samples with role zero masks: {role_zero}")
        if final_zero and not args.allow_final_zero:
            issues.append(f"samples with final zero masks: {final_zero}")
        elif final_zero:
            warnings.append(f"samples with final zero masks: {final_zero}")

    if train_manifest is not None:
        if train_manifest.get("status") != "ready":
            issues.append(f"train manifest status is not ready: {train_manifest.get('status')}")
        if not args.allow_sample_count_mismatch and int(train_manifest.get("sample_count", -1)) != int(args.expected_samples):
            issues.append(f"train manifest sample_count {train_manifest.get('sample_count')} != {args.expected_samples}")
        if int(train_manifest.get("ready_count", -1)) != int(train_manifest.get("sample_count", -2)):
            issues.append(f"train ready_count {train_manifest.get('ready_count')} != sample_count {train_manifest.get('sample_count')}")
        train_issue_count = sample_issue_count(train_manifest) or 0
        train_warning_count = sample_warning_count(train_manifest) or 0
        if train_issue_count:
            issues.append(f"train sample issue count: {train_issue_count}")
        if train_warning_count and not args.allow_train_warnings:
            issues.append(f"train sample warning count: {train_warning_count}")
        elif train_warning_count:
            warnings.append(f"train sample warning count: {train_warning_count}")

    summary = {
        "pipeline_status": pipeline.get("status") if pipeline else None,
        "validation_status": validation.get("status") if validation else None,
        "verification_status": verification.get("status") if verification else None,
        "train_status": train_manifest.get("status") if train_manifest else None,
        "train_ready_count": train_manifest.get("ready_count") if train_manifest else None,
        "train_sample_count": train_manifest.get("sample_count") if train_manifest else None,
        "active_correction_count": coverage.get("summary", {}).get("active_correction_count") if coverage else None,
        "covered_actionable_count": coverage.get("summary", {}).get("covered_actionable_task_count") if coverage else None,
        "actionable_task_count": coverage.get("summary", {}).get("actionable_task_count") if coverage else None,
        "uncovered_critical_count": count_uncovered_priority(coverage, "critical"),
        "uncovered_high_count": count_uncovered_priority(coverage, "high"),
        "role_zero_sample_count": count_role_zero(verification),
        "final_zero_sample_count": count_final_zero(verification),
        "train_sample_issue_count": sample_issue_count(train_manifest),
        "train_sample_warning_count": sample_warning_count(train_manifest),
    }

    return {
        "schema_version": "maskwam_corrected_run_audit_v1",
        "status": "ready" if not issues else "not_ready",
        "run_dir": str(run_dir.resolve()),
        "expected_samples": int(args.expected_samples),
        "paths": {
            "pipeline_summary": str(pipeline_path.resolve()),
            "correction_coverage": str(coverage_path.resolve()),
            "correction_validation": str(validation_path.resolve()),
            "verification": str(verification_path.resolve()),
            "train_mask_sequences": str(train_path.resolve()),
        },
        "summary": summary,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json = Path(args.output_json) if args.output_json else Path(args.run_dir) / "audit.json"
    output_md = Path(args.output_md) if args.output_md else Path(args.run_dir) / "audit.md"
    write_json(output_json, report)
    write_markdown(output_md, report)
    summary = report["summary"]
    print(
        "status={status} train={ready}/{total} uncovered_critical={critical} issues={issues} warnings={warnings}".format(
            status=report["status"],
            ready=summary.get("train_ready_count"),
            total=summary.get("train_sample_count"),
            critical=summary.get("uncovered_critical_count"),
            issues=len(report["issues"]),
            warnings=len(report["warnings"]),
        )
    )
    print(output_json)
    print(output_md)
    if report["issues"]:
        for issue in report["issues"][:20]:
            print(f"ISSUE {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
