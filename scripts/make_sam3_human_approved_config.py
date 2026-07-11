#!/usr/bin/env python3
"""Create a human-approved SAM3 override config after explicit QA approval."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List


APPROVED_VALUES = {"yes", "y", "true", "1", "approved", "approve", "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_config",
        default="configs/sam3_manual_overrides_debug16.v4_draft.json",
        help="Manual override config to promote after approval.",
    )
    parser.add_argument(
        "--decisions_csv",
        default="artifacts/v3_sam3_mask_review_manual_v4_draft/summary/human_decision_template.csv",
        help="Human decision CSV with approved_for_larger_validation column.",
    )
    parser.add_argument(
        "--output_config",
        default="configs/sam3_manual_overrides_debug16.v4_human_approved.json",
    )
    parser.add_argument(
        "--approval_note",
        default="human_approved_after_16case_visual_qa",
        help="Metadata note written into the output config.",
    )
    parser.add_argument(
        "--allow_rejected",
        action="store_true",
        help="Allow explicitly rejected rows by recording them in metadata. Use only if those cases should be excluded downstream.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_decisions(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def is_approved(value: str) -> bool:
    return value.strip().lower() in APPROVED_VALUES


def validate_decisions(rows: List[Dict[str, str]], allow_rejected: bool) -> Dict[str, List[str]]:
    missing: List[str] = []
    rejected: List[str] = []
    approved: List[str] = []
    for row in rows:
        sample = row.get("sample", "").strip()
        value = row.get("approved_for_larger_validation", "").strip()
        decision = row.get("human_decision", "").strip()
        if not value:
            missing.append(sample or "<unknown>")
            continue
        if is_approved(value):
            approved.append(sample)
            if not decision:
                missing.append(f"{sample}: missing human_decision text")
            continue
        rejected.append(sample)

    errors = []
    if missing:
        errors.append("Missing approvals/decision text: " + ", ".join(missing))
    if rejected and not allow_rejected:
        errors.append("Rejected/not-approved rows: " + ", ".join(rejected))
    if errors:
        raise SystemExit("\n".join(errors))

    return {"approved": approved, "rejected": rejected}


def add_metadata(config: Dict, args: argparse.Namespace, rows: List[Dict[str, str]], status: Dict[str, List[str]]) -> Dict:
    out = copy.deepcopy(config)
    metadata = out.setdefault("metadata", {})
    metadata["approval_note"] = args.approval_note
    metadata["base_config"] = str(Path(args.base_config).resolve())
    metadata["decisions_csv"] = str(Path(args.decisions_csv).resolve())
    metadata["approved_samples"] = status["approved"]
    metadata["rejected_samples"] = status["rejected"]
    metadata["human_decisions"] = [
        {
            "sample": row.get("sample", ""),
            "current_status": row.get("current_status", ""),
            "human_decision": row.get("human_decision", ""),
            "approved_for_larger_validation": row.get("approved_for_larger_validation", ""),
        }
        for row in rows
    ]
    return out


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    decisions_path = Path(args.decisions_csv)
    output_path = Path(args.output_config)

    config = load_json(base_config_path)
    rows = read_decisions(decisions_path)
    status = validate_decisions(rows, args.allow_rejected)
    output = add_metadata(config, args, rows, status)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(output_path.resolve())
    print(f"approved_samples={len(status['approved'])} rejected_samples={len(status['rejected'])}")


if __name__ == "__main__":
    main()
