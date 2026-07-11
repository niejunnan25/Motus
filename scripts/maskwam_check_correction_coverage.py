#!/usr/bin/env python3
"""Check whether MaskWAM human corrections cover the generated task list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from maskwam_view_schema import VIEW_TRAIN_ROLES


TRAIN_ROLES = set(VIEW_TRAIN_ROLES)
MANUAL_NAMES = {None, "", "manual point", "manual box"}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json",
    )
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--require_all_critical", action="store_true")
    parser.add_argument("--require_all_high", action="store_true")
    parser.add_argument("--require_all_actionable", action="store_true")
    parser.add_argument(
        "--no_manual_role_match",
        action="store_true",
        help="Do not let manual role-only corrections cover task prompts with the same role.",
    )
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
        "actionable",
        "aggregate",
        "covered",
        "coverage_type",
        "matched_count",
        "reason",
        "suggested_action",
        "matched_corrections",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def list_value(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def is_active_prompt(prompt: Dict[str, Any]) -> bool:
    return bool(
        list_value(prompt.get("points_abs"))
        or list_value(prompt.get("points_rel"))
        or list_value(prompt.get("boxes_xywh_abs"))
        or list_value(prompt.get("boxes_xywh_rel"))
        or prompt.get("accepted_absent")
    )


def prompt_signature(sample_id: str, prompt_idx: int, prompt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "prompt_idx": prompt_idx,
        "role": prompt.get("role"),
        "name": prompt.get("name"),
        "frame_index": int(prompt.get("frame_index", 0)),
        "point_count": len(list_value(prompt.get("points_abs"))) + len(list_value(prompt.get("points_rel"))),
        "box_count": len(list_value(prompt.get("boxes_xywh_abs"))) + len(list_value(prompt.get("boxes_xywh_rel"))),
        "accepted_absent": bool(prompt.get("accepted_absent")),
    }


def collect_active_corrections(corrections: Dict[str, Any]) -> List[Dict[str, Any]]:
    active = []
    for sample_id, sample_cfg in corrections.get("samples", {}).items():
        if not isinstance(sample_cfg, dict):
            continue
        for prompt_idx, prompt in enumerate(sample_cfg.get("prompts", [])):
            if not isinstance(prompt, dict) or not is_active_prompt(prompt):
                continue
            active.append(prompt_signature(str(sample_id), prompt_idx, prompt))
    active.sort(key=lambda item: (item["sample_id"], str(item["role"]), str(item["name"]), item["frame_index"]))
    return active


def task_key(task: Dict[str, Any]) -> Tuple[int, str, str]:
    return int(task.get("sample_id")), str(task.get("role", "")), str(task.get("name", ""))


def is_actionable_task(task: Dict[str, Any]) -> bool:
    return task.get("role") in TRAIN_ROLES and bool(task.get("name"))


def is_aggregate_task(task: Dict[str, Any]) -> bool:
    return task.get("role") == "final_train_mask"


def correction_matches_task(correction: Dict[str, Any], task: Dict[str, Any], allow_manual_role_match: bool) -> Tuple[bool, str]:
    if int(correction["sample_id"]) != int(task.get("sample_id")):
        return False, ""
    if correction.get("role") != task.get("role"):
        return False, ""
    if correction.get("name") == task.get("name"):
        return True, "accepted_absent" if correction.get("accepted_absent") else "exact"
    if allow_manual_role_match and correction.get("name") in MANUAL_NAMES:
        return True, "manual_role"
    return False, ""


def summarize_counts(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    by_priority: Dict[str, Dict[str, int]] = {}
    by_role: Dict[str, Dict[str, int]] = {}

    def add(bucket: Dict[str, Dict[str, int]], key: str, row: Dict[str, Any]) -> None:
        stats = bucket.setdefault(
            key,
            {
                "tasks": 0,
                "actionable": 0,
                "covered_actionable": 0,
                "uncovered_actionable": 0,
                "aggregate": 0,
                "covered_aggregate": 0,
            },
        )
        stats["tasks"] += 1
        if row["actionable"]:
            stats["actionable"] += 1
            if row["covered"]:
                stats["covered_actionable"] += 1
            else:
                stats["uncovered_actionable"] += 1
        if row["aggregate"]:
            stats["aggregate"] += 1
            if row["covered"]:
                stats["covered_aggregate"] += 1

    for row in rows:
        add(by_priority, str(row["priority"]), row)
        add(by_role, str(row["role"]), row)

    by_priority = dict(sorted(by_priority.items(), key=lambda item: PRIORITY_ORDER.get(item[0], 99)))
    by_role = dict(sorted(by_role.items(), key=lambda item: item[0]))
    return by_priority, by_role


def build_report(
    tasks_payload: Dict[str, Any],
    corrections: Dict[str, Any],
    allow_manual_role_match: bool,
    sample_ids: Sequence[int] | None = None,
) -> Dict[str, Any]:
    tasks = list(tasks_payload.get("tasks", []))
    if sample_ids is not None:
        sample_id_set = {int(value) for value in sample_ids}
        tasks = [task for task in tasks if int(task.get("sample_id")) in sample_id_set]
    active_corrections = collect_active_corrections(corrections)
    rows: List[Dict[str, Any]] = []
    matched_correction_keys = set()

    actionable_matches: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for task in tasks:
        if not is_actionable_task(task):
            continue
        matches = []
        for correction in active_corrections:
            matched, match_type = correction_matches_task(correction, task, allow_manual_role_match)
            if matched:
                item = {**correction, "match_type": match_type}
                matches.append(item)
                matched_correction_keys.add((correction["sample_id"], correction["prompt_idx"]))
        actionable_matches[task_key(task)] = matches

    uncovered_actionable_keys_by_sample: Dict[int, List[Tuple[int, str, str]]] = {}
    for task in tasks:
        if not is_actionable_task(task):
            continue
        key = task_key(task)
        if actionable_matches.get(key):
            continue
        uncovered_actionable_keys_by_sample.setdefault(key[0], []).append(key)

    for idx, task in enumerate(tasks):
        actionable = is_actionable_task(task)
        aggregate = is_aggregate_task(task)
        matches: List[Dict[str, Any]] = []
        coverage_type = "not_actionable"
        covered = False

        if actionable:
            matches = actionable_matches.get(task_key(task), [])
            covered = bool(matches)
            coverage_type = matches[0]["match_type"] if matches else "uncovered"
        elif aggregate:
            sample_id = int(task.get("sample_id"))
            sample_uncovered = uncovered_actionable_keys_by_sample.get(sample_id, [])
            sample_matches = [
                match
                for key, key_matches in actionable_matches.items()
                if key[0] == sample_id
                for match in key_matches
            ]
            covered = bool(sample_matches) and not sample_uncovered
            matches = sample_matches
            coverage_type = "aggregate_all_train_roles_covered" if covered else "aggregate_waiting_for_train_role_fixes"

        rows.append(
            {
                "task_idx": idx,
                "priority": task.get("priority", ""),
                "sample_id": int(task.get("sample_id")),
                "sample_name": task.get("sample_name", ""),
                "role": task.get("role", ""),
                "name": task.get("name", ""),
                "actionable": actionable,
                "aggregate": aggregate,
                "covered": covered,
                "coverage_type": coverage_type,
                "matched_count": len(matches),
                "reason": task.get("reason", ""),
                "suggested_action": task.get("suggested_action", ""),
                "matched_corrections": matches,
            }
        )

    by_priority, by_role = summarize_counts(rows)
    actionable_rows = [row for row in rows if row["actionable"]]
    covered_actionable = [row for row in actionable_rows if row["covered"]]
    uncovered_actionable = [row for row in actionable_rows if not row["covered"]]
    aggregate_rows = [row for row in rows if row["aggregate"]]
    unmatched_active = [
        correction
        for correction in active_corrections
        if (correction["sample_id"], correction["prompt_idx"]) not in matched_correction_keys
    ]

    samples_with_uncovered = sorted({row["sample_id"] for row in uncovered_actionable})
    status = "covered" if not uncovered_actionable else "needs_more_corrections"

    return {
        "schema_version": "maskwam_correction_coverage_v1",
        "status": status,
        "allow_manual_role_match": allow_manual_role_match,
        "sample_ids": list(sample_ids) if sample_ids is not None else None,
        "summary": {
            "task_count": len(rows),
            "actionable_task_count": len(actionable_rows),
            "covered_actionable_task_count": len(covered_actionable),
            "uncovered_actionable_task_count": len(uncovered_actionable),
            "aggregate_task_count": len(aggregate_rows),
            "covered_aggregate_task_count": sum(1 for row in aggregate_rows if row["covered"]),
            "active_correction_count": len(active_corrections),
            "matched_active_correction_count": len(matched_correction_keys),
            "unmatched_active_correction_count": len(unmatched_active),
            "sample_count_with_tasks": len({row["sample_id"] for row in rows}),
            "sample_count_with_uncovered_actionable": len(samples_with_uncovered),
            "samples_with_uncovered_actionable": samples_with_uncovered,
            "by_priority": by_priority,
            "by_role": by_role,
        },
        "coverage_rows": rows,
        "uncovered_actionable": uncovered_actionable,
        "unmatched_active_corrections": unmatched_active,
    }


def csv_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in report["coverage_rows"]:
        rows.append(
            {
                **{key: row.get(key, "") for key in [
                    "priority",
                    "sample_id",
                    "sample_name",
                    "role",
                    "name",
                    "actionable",
                    "aggregate",
                    "covered",
                    "coverage_type",
                    "matched_count",
                    "reason",
                    "suggested_action",
                ]},
                "matched_corrections": json.dumps(row.get("matched_corrections", []), ensure_ascii=False),
            }
        )
    return rows


def unmet_required_priorities(report: Dict[str, Any], require_all_critical: bool, require_all_high: bool) -> List[str]:
    failures = []
    for priority in ("critical", "high"):
        required = (priority == "critical" and require_all_critical) or (priority == "high" and require_all_high)
        if not required:
            continue
        uncovered = [
            row
            for row in report["uncovered_actionable"]
            if row.get("priority") == priority
        ]
        if uncovered:
            failures.append(f"{len(uncovered)} uncovered {priority} actionable task(s)")
    return failures


def main() -> None:
    args = parse_args()
    tasks_path = Path(args.tasks)
    corrections_path = Path(args.corrections)
    tasks_payload = load_json(tasks_path)
    corrections = load_json(corrections_path)
    report = build_report(
        tasks_payload,
        corrections,
        allow_manual_role_match=not args.no_manual_role_match,
        sample_ids=args.sample_ids,
    )
    report["tasks"] = str(tasks_path.resolve())
    report["corrections"] = str(corrections_path.resolve())

    if args.output_json:
        write_json(args.output_json, report)
    if args.output_csv:
        write_csv(args.output_csv, csv_rows(report))

    summary = report["summary"]
    print(
        "status={status} actionable={covered}/{total} active_corrections={active} unmatched_active={unmatched}".format(
            status=report["status"],
            covered=summary["covered_actionable_task_count"],
            total=summary["actionable_task_count"],
            active=summary["active_correction_count"],
            unmatched=summary["unmatched_active_correction_count"],
        )
    )
    if args.output_json:
        print(args.output_json)
    if args.output_csv:
        print(args.output_csv)

    failures = []
    if args.require_all_actionable and report["uncovered_actionable"]:
        failures.append(f"{len(report['uncovered_actionable'])} uncovered actionable task(s)")
    failures.extend(unmet_required_priorities(report, args.require_all_critical, args.require_all_high))
    if failures:
        for failure in failures:
            print(f"ISSUE {failure}")
        for row in report["uncovered_actionable"][:20]:
            print(
                "UNCOVERED {priority} sample={sample_id} {role}:{name} reason={reason}".format(
                    **row
                )
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
