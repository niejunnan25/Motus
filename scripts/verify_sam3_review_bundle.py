#!/usr/bin/env python3
"""Verify a SAM3 human-review bundle before moving to larger validation."""

from __future__ import annotations

import argparse
import csv
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List


APPROVED_VALUES = {"yes", "y", "true", "1", "approved", "approve", "ok"}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: List[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        data = dict(attrs)
        for key in ("href", "src"):
            ref = data.get(key)
            if ref and not ref.startswith(("http://", "https://", "#")):
                self.refs.append(ref)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default="artifacts/v3_sam3_mask_review_manual_v4_draft")
    parser.add_argument("--expected_samples", type=int, default=16)
    parser.add_argument("--expected_issue_samples", type=int, default=5)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def check_html_links(path: Path) -> Dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "refs": 0, "missing": [str(path)]}
    parser = LinkParser()
    parser.feed(path.read_text(encoding="utf-8"))
    missing = []
    for ref in parser.refs:
        if not (path.parent / ref).resolve().exists():
            missing.append(ref)
    return {"path": str(path), "exists": True, "refs": len(parser.refs), "missing": missing}


def count_files(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def decision_status(decision_csv: Path) -> Dict:
    rows = read_csv(decision_csv)
    missing = []
    rejected = []
    approved = []
    for row in rows:
        sample = row.get("sample", "")
        decision = row.get("human_decision", "").strip()
        approval = row.get("approved_for_larger_validation", "").strip().lower()
        if not decision or not approval:
            missing.append(sample)
        elif approval in APPROVED_VALUES:
            approved.append(sample)
        else:
            rejected.append(sample)
    return {
        "rows": len(rows),
        "approved": approved,
        "missing": missing,
        "rejected": rejected,
        "all_approved": bool(rows) and not missing and not rejected,
    }


def sample_metric(metrics_csv: Path, sample: str, run: str) -> Dict[str, str]:
    for row in read_csv(metrics_csv):
        if row.get("sample") == sample and row.get("run") == run:
            return row
    return {}


def build_report(review_dir: Path, expected_samples: int, expected_issue_samples: int) -> Dict:
    summary_dir = review_dir / "summary"
    dashboard = summary_dir / "sam3_mask_review_dashboard.html"
    decision_pack = summary_dir / "human_decision_pack.html"
    metrics_csv = summary_dir / "review_metrics.csv"
    issue_csv = summary_dir / "issue_frames.csv"
    decision_csv = summary_dir / "human_decision_template.csv"
    outputs_txt = summary_dir / "review_outputs.txt"
    delta_md = summary_dir / "run_delta_summary.md"
    identity_aid = review_dir / "identity_decision_aids" / "sample_009_identity_decision_aid.png"

    counts = {
        "focused": count_files(review_dir / "focused" / "manual-v4-draft", "*_focused_review.png"),
        "compare": count_files(review_dir / "compare", "*_review.png"),
        "top_splits": count_files(review_dir / "focused_view_splits" / "manual-v4-draft" / "top", "*_top_view_review.png"),
        "bottom_splits": count_files(review_dir / "focused_view_splits" / "manual-v4-draft" / "bottom", "*_bottom_view_review.png"),
        "issue_crops": count_files(review_dir / "issue_crops" / "manual-v4-draft", "*_issue_crop_review.png"),
        "operation_crops": count_files(review_dir / "operation_crops", "*_operation_crop_review.png"),
        "identity_aids": count_files(review_dir / "identity_decision_aids", "*_identity_decision_aid.png"),
    }
    expected = {
        "focused": expected_samples,
        "compare": expected_samples,
        "top_splits": expected_samples,
        "bottom_splits": expected_samples,
        "issue_crops": expected_issue_samples,
        "operation_crops": expected_issue_samples,
        "identity_aids": 1,
    }

    required_files = [
        dashboard,
        decision_pack,
        metrics_csv,
        issue_csv,
        decision_csv,
        outputs_txt,
        delta_md,
        identity_aid,
    ]
    missing_required = [str(path) for path in required_files if not path.exists()]

    html_checks = [check_html_links(dashboard), check_html_links(decision_pack)]
    decisions = decision_status(decision_csv)
    sample_002 = sample_metric(metrics_csv, "sample_002", "manual-v4-draft")
    sample_009_issue = next((row for row in read_csv(issue_csv) if row.get("sample") == "sample_009"), {})

    failures = []
    if missing_required:
        failures.extend(f"missing required file: {path}" for path in missing_required)
    for key, expected_count in expected.items():
        if counts.get(key) != expected_count:
            failures.append(f"{key} count {counts.get(key)} != expected {expected_count}")
    for check in html_checks:
        if check["missing"]:
            failures.append(f"{Path(check['path']).name} has missing refs: {check['missing']}")
    if sample_002.get("object_zero") != "0" or sample_002.get("target_zero") != "0":
        failures.append("sample_002 v4 draft did not keep object_zero=0 and target_zero=0")
    if sample_009_issue.get("current_status") != "ambiguous_object_identity":
        failures.append("sample_009 is no longer marked ambiguous_object_identity")

    if failures:
        status = "invalid_bundle"
    elif decisions["all_approved"]:
        status = "ready_for_larger_validation"
    else:
        status = "blocked_by_missing_human_decisions"

    return {
        "review_dir": str(review_dir.resolve()),
        "status": status,
        "counts": counts,
        "expected_counts": expected,
        "missing_required": missing_required,
        "html_checks": html_checks,
        "decision_status": decisions,
        "sample_002_manual_v4_draft": {
            "object_zero": sample_002.get("object_zero", ""),
            "target_zero": sample_002.get("target_zero", ""),
            "final_mean": sample_002.get("final_mean", ""),
            "final_max": sample_002.get("final_max", ""),
        },
        "sample_009_status": sample_009_issue.get("current_status", ""),
        "failures": failures,
    }


def write_markdown(report: Dict, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "### SAM3 Review Bundle Verification",
        "",
        f"Review dir: `{report['review_dir']}`",
        "",
        f"Status: `{report['status']}`",
        "",
        "### Counts",
        "",
        "| artifact | count | expected |",
        "| --- | ---: | ---: |",
    ]
    for key, count in report["counts"].items():
        lines.append(f"| {key} | {count} | {report['expected_counts'].get(key, '')} |")

    lines.extend(["", "### HTML Links", "", "| page | refs | missing |", "| --- | ---: | ---: |"])
    for check in report["html_checks"]:
        lines.append(f"| {Path(check['path']).name} | {check['refs']} | {len(check['missing'])} |")

    decisions = report["decision_status"]
    lines.extend(
        [
            "",
            "### Human Decisions",
            "",
            f"- rows: `{decisions['rows']}`",
            f"- approved: `{len(decisions['approved'])}`",
            f"- missing: `{', '.join(decisions['missing']) or 'none'}`",
            f"- rejected: `{', '.join(decisions['rejected']) or 'none'}`",
            "",
            "### Key Checks",
            "",
            f"- sample_002 manual-v4-draft object_zero: `{report['sample_002_manual_v4_draft']['object_zero']}`",
            f"- sample_002 manual-v4-draft target_zero: `{report['sample_002_manual_v4_draft']['target_zero']}`",
            f"- sample_009 status: `{report['sample_009_status']}`",
            "",
        ]
    )
    if report["failures"]:
        lines.extend(["### Failures", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.extend(
            [
                "### Interpretation",
                "",
                "The bundle is structurally valid. It is still blocked from larger validation until the five human decision rows are explicitly approved.",
            ]
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    summary_dir = review_dir / "summary"
    output_json = Path(args.output_json) if args.output_json else summary_dir / "review_bundle_verification.json"
    output_md = Path(args.output_md) if args.output_md else summary_dir / "review_bundle_verification.md"
    report = build_report(review_dir, args.expected_samples, args.expected_issue_samples)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, output_md)
    print(output_json.resolve())
    print(output_md.resolve())
    print(f"status={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
