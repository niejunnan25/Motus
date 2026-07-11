#!/usr/bin/env python3
"""Summarize metric deltas between two SAM3 review runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


METRIC_FIELDS = [
    "object_zero",
    "object_mean",
    "target_zero",
    "target_mean",
    "final_mean",
    "final_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_csv",
        default="artifacts/v3_sam3_mask_review_manual_v4_draft/summary/review_metrics.csv",
    )
    parser.add_argument("--before_run", default="manual-v3-refined")
    parser.add_argument("--after_run", default="manual-v4-draft")
    parser.add_argument("--issue_frames_csv", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_md", default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        return {(row["sample"], row["run"]): row for row in csv.DictReader(file)}


def read_issue_status(path: str | None) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    issue_path = Path(path)
    if not issue_path.exists():
        return {}
    with open(issue_path, "r", encoding="utf-8", newline="") as file:
        return {row["sample"]: row for row in csv.DictReader(file)}


def parse_num(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_delta(before: str, after: str) -> str:
    b = parse_num(before)
    a = parse_num(after)
    if b is None or a is None:
        return ""
    delta = a - b
    if abs(delta) < 1e-12:
        return "0"
    if before.isdigit() and after.isdigit():
        return f"{int(delta):+d}"
    return f"{delta:+.6f}"


def improvement_note(row: Dict[str, str]) -> str:
    notes: List[str] = []
    for prefix in ("object", "target"):
        before = parse_num(row[f"{prefix}_zero_before"])
        after = parse_num(row[f"{prefix}_zero_after"])
        if before is None or after is None:
            continue
        if before > 0 and after == 0:
            notes.append(f"{prefix}_zero fixed")
        elif after > before:
            notes.append(f"{prefix}_zero worsened")
    if not notes:
        if any(row.get(f"{field}_delta") not in ("", "0") for field in METRIC_FIELDS):
            notes.append("metric changed")
        else:
            notes.append("unchanged")
    return "; ".join(notes)


def build_rows(
    metrics: Dict[Tuple[str, str], Dict[str, str]],
    issue_status: Dict[str, Dict[str, str]],
    before_run: str,
    after_run: str,
) -> List[Dict[str, str]]:
    samples = sorted({sample for sample, _run in metrics})
    rows: List[Dict[str, str]] = []
    for sample in samples:
        before = metrics.get((sample, before_run))
        after = metrics.get((sample, after_run))
        if before is None or after is None:
            continue
        row = {
            "sample": sample,
            "task_text": after.get("task_text", ""),
            "before_run": before_run,
            "after_run": after_run,
            "issue_status": issue_status.get(sample, {}).get("current_status", ""),
            "issue_reasons": issue_status.get(sample, {}).get("reasons", ""),
        }
        for field in METRIC_FIELDS:
            row[f"{field}_before"] = before.get(field, "")
            row[f"{field}_after"] = after.get(field, "")
            row[f"{field}_delta"] = fmt_delta(before.get(field, ""), after.get(field, ""))
        row["note"] = improvement_note(row)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "task_text",
        "before_run",
        "after_run",
        "issue_status",
        "issue_reasons",
    ]
    for field in METRIC_FIELDS:
        fieldnames.extend([f"{field}_before", f"{field}_after", f"{field}_delta"])
    fieldnames.append("note")
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output_md: Path, before_run: str, after_run: str) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    changed = [row for row in rows if row["note"] != "unchanged"]
    issue_rows = [row for row in rows if row.get("issue_status")]
    lines = [
        "### SAM3 Run Delta Summary",
        "",
        f"Before: `{before_run}`",
        "",
        f"After: `{after_run}`",
        "",
        "This summary compares review metrics only. It does not replace visual human QA.",
        "",
        "### Changed Samples",
        "",
    ]
    if changed:
        lines.append("| sample | note | object zero | target zero | final mean | final max |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in changed:
            lines.append(
                "| {sample} | {note} | {ozb} -> {oza} ({ozd}) | {tzb} -> {tza} ({tzd}) | {fmb} -> {fma} ({fmd}) | {fxb} -> {fxa} ({fxd}) |".format(
                    sample=row["sample"],
                    note=row["note"],
                    ozb=row["object_zero_before"],
                    oza=row["object_zero_after"],
                    ozd=row["object_zero_delta"],
                    tzb=row["target_zero_before"],
                    tza=row["target_zero_after"],
                    tzd=row["target_zero_delta"],
                    fmb=row["final_mean_before"],
                    fma=row["final_mean_after"],
                    fmd=row["final_mean_delta"],
                    fxb=row["final_max_before"],
                    fxa=row["final_max_after"],
                    fxd=row["final_max_delta"],
                )
            )
    else:
        lines.append("No metric changes.")

    lines.extend(["", "### Remaining Issue Samples", ""])
    if issue_rows:
        lines.append("| sample | status | reasons | note |")
        lines.append("| --- | --- | --- | --- |")
        for row in issue_rows:
            lines.append(
                f"| {row['sample']} | {row['issue_status']} | {row['issue_reasons']} | {row['note']} |"
            )
    else:
        lines.append("No issue-frame samples listed.")

    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "- A zero-mask count dropping to 0 means the role is numerically present in every sampled frame.",
            "- This does not guarantee semantic correctness; identity ambiguity and overly broad masks still require visual approval.",
            "- For the current v4 draft, the key numeric repair is `sample_002`; the other remaining blockers are human semantic decisions.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    output_csv = Path(args.output_csv) if args.output_csv else metrics_csv.parent / "run_delta_summary.csv"
    output_md = Path(args.output_md) if args.output_md else metrics_csv.parent / "run_delta_summary.md"
    rows = build_rows(
        read_metrics(metrics_csv),
        read_issue_status(args.issue_frames_csv),
        args.before_run,
        args.after_run,
    )
    write_csv(rows, output_csv)
    write_markdown(rows, output_md, args.before_run, args.after_run)
    print(output_csv.resolve())
    print(output_md.resolve())


if __name__ == "__main__":
    main()
