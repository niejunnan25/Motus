#!/usr/bin/env python3
"""Summarize pending MaskWAM human corrections before SAM3 re-propagation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from maskwam_view_schema import FINAL_ROLE, VIEW_TRAIN_ROLES


ACTIONABLE_ROLES = set(VIEW_TRAIN_ROLES)
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json",
    )
    parser.add_argument(
        "--verification",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/verification.json",
    )
    parser.add_argument(
        "--output_json",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.json",
    )
    parser.add_argument(
        "--output_csv",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.csv",
    )
    parser.add_argument(
        "--output_md",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.md",
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


def task_sort_key(task: Dict[str, Any]) -> tuple:
    return (
        int(task.get("sample_id", 999999)),
        PRIORITY_ORDER.get(str(task.get("priority", "low")), 99),
        str(task.get("role", "")),
        str(task.get("name", "")),
    )


def is_actionable(task: Dict[str, Any]) -> bool:
    return str(task.get("role")) in ACTIONABLE_ROLES


def is_prepare_gate(task: Dict[str, Any]) -> bool:
    return is_actionable(task) and task.get("priority") == "critical"


def is_train_ready_gate(task: Dict[str, Any]) -> bool:
    return is_actionable(task)


def prompt_key(task: Dict[str, Any]) -> tuple:
    return (
        int(task.get("sample_id")),
        str(task.get("role", "")),
        str(task.get("name", "")),
    )


def unique_prompt_count(tasks: Iterable[Dict[str, Any]]) -> int:
    return len({prompt_key(task) for task in tasks})


def compact_task(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "priority": task.get("priority"),
        "sample_id": task.get("sample_id"),
        "sample_name": task.get("sample_name"),
        "role": task.get("role"),
        "name": task.get("name"),
        "reason": task.get("reason"),
        "zero_frame_count": task.get("zero_frame_count"),
        "mean_area_ratio": task.get("mean_area_ratio"),
        "zero_frame_indices": task.get("zero_frame_indices"),
        "recommended_frame_index": task.get("recommended_frame_index"),
        "recommended_frame_reason": task.get("recommended_frame_reason"),
        "suggested_action": task.get("suggested_action"),
        "task_text": task.get("task_text"),
        "review_sheet": task.get("review_sheet"),
        "click_frames": task.get("click_frames"),
        "actionable": is_actionable(task),
        "prepare_gate": is_prepare_gate(task),
        "train_ready_gate": is_train_ready_gate(task),
    }


def build_sample_rows(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[int(task["sample_id"])].append(task)

    rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        sample_tasks = sorted(grouped[sample_id], key=task_sort_key)
        actionables = [task for task in sample_tasks if is_actionable(task)]
        prepare_tasks = [task for task in sample_tasks if is_prepare_gate(task)]
        train_ready_tasks = [task for task in sample_tasks if is_train_ready_gate(task)]
        aggregate_tasks = [task for task in sample_tasks if not is_actionable(task)]
        first = sample_tasks[0]
        rows.append(
            {
                "sample_id": sample_id,
                "sample_name": first.get("sample_name"),
                "task_text": first.get("task_text"),
                "task_count": len(sample_tasks),
                "actionable_task_count": len(actionables),
                "prepare_gate_task_count": len(prepare_tasks),
                "prepare_gate_prompt_count": unique_prompt_count(prepare_tasks),
                "train_ready_task_count": len(train_ready_tasks),
                "train_ready_prompt_count": unique_prompt_count(train_ready_tasks),
                "aggregate_task_count": len(aggregate_tasks),
                "priorities": dict(Counter(task.get("priority") for task in sample_tasks)),
                "roles": dict(Counter(task.get("role") for task in sample_tasks)),
                "prepare_gate_targets": "; ".join(
                    f"{task.get('priority')}:{task.get('role')}:{task.get('name')}"
                    for task in prepare_tasks
                ),
                "train_ready_targets": "; ".join(
                    f"{task.get('priority')}:{task.get('role')}:{task.get('name')}"
                    for task in train_ready_tasks
                ),
                "recommended_frames": "; ".join(
                    f"{task.get('role')}:{task.get('name')}@F{task.get('recommended_frame_index')}"
                    for task in train_ready_tasks
                ),
                "aggregate_reasons": "; ".join(task.get("reason", "") for task in aggregate_tasks),
                "review_sheet": first.get("review_sheet"),
            }
        )
    return rows


def build_report(tasks_payload: Dict[str, Any], verification: Dict[str, Any] | None) -> Dict[str, Any]:
    tasks = sorted(tasks_payload.get("tasks", []), key=task_sort_key)
    actionables = [task for task in tasks if is_actionable(task)]
    prepare_tasks = [task for task in tasks if is_prepare_gate(task)]
    train_ready_tasks = [task for task in tasks if is_train_ready_gate(task)]
    aggregate_tasks = [task for task in tasks if not is_actionable(task)]
    sample_rows = build_sample_rows(tasks)

    by_priority = Counter(task.get("priority") for task in tasks)
    actionable_by_priority = Counter(task.get("priority") for task in actionables)
    by_role = Counter(task.get("role") for task in tasks)

    report = {
        "schema_version": "maskwam_pending_corrections_summary_v1",
        "tasks": str(Path(tasks_payload.get("tasks_path", "")).resolve()) if tasks_payload.get("tasks_path") else None,
        "verification_status": verification.get("status") if verification else None,
        "sample_count": tasks_payload.get("sample_count"),
        "sample_with_task_count": tasks_payload.get("sample_with_task_count"),
        "task_count": len(tasks),
        "actionable_task_count": len(actionables),
        "aggregate_task_count": len(aggregate_tasks),
        "prepare_gate": {
            "description": "Minimum active point/box prompts needed before a default critical-only re-propagation attempt.",
            "task_count": len(prepare_tasks),
            "unique_prompt_count": unique_prompt_count(prepare_tasks),
            "sample_count": len({int(task["sample_id"]) for task in prepare_tasks}),
        },
        "train_ready_gate": {
            "description": "Recommended active point/box prompts before the formal 16-case run; use --require_all_actionable.",
            "task_count": len(train_ready_tasks),
            "unique_prompt_count": unique_prompt_count(train_ready_tasks),
            "sample_count": len({int(task["sample_id"]) for task in train_ready_tasks}),
        },
        "by_priority": dict(by_priority),
        "actionable_by_priority": dict(actionable_by_priority),
        "by_role": dict(by_role),
        "sample_rows": sample_rows,
        "pending_tasks": [compact_task(task) for task in tasks],
    }
    if verification:
        report["verification"] = {
            "sample_count": verification.get("sample_count"),
            "samples_with_role_zero": verification.get("samples_with_role_zero", []),
            "samples_with_final_zero": verification.get("samples_with_final_zero", []),
        }
    return report


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sample_name",
        "task_count",
        "actionable_task_count",
        "prepare_gate_task_count",
        "prepare_gate_prompt_count",
        "train_ready_task_count",
        "train_ready_prompt_count",
        "aggregate_task_count",
        "prepare_gate_targets",
        "train_ready_targets",
        "recommended_frames",
        "aggregate_reasons",
        "task_text",
        "review_sheet",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: str | Path, report: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MaskWAM Pending Corrections",
        "",
        f"- verification status: `{report.get('verification_status')}`",
        f"- tasks: `{report['task_count']}`",
        f"- actionable tasks: `{report['actionable_task_count']}`",
        f"- aggregate tasks: `{report['aggregate_task_count']}`",
        f"- prepare gate prompts: `{report['prepare_gate']['unique_prompt_count']}` across `{report['prepare_gate']['sample_count']}` samples",
        f"- train-ready gate prompts: `{report['train_ready_gate']['unique_prompt_count']}` across `{report['train_ready_gate']['sample_count']}` samples",
        "",
        "## Interpretation",
        "",
        "- `prepare gate` is the minimum for the default critical-only re-propagation path.",
        "- `train-ready gate` is the recommended target for the formal 16-case run, because audit can still fail when high/medium view-role masks have zero frames.",
        f"- `{FINAL_ROLE}` rows are aggregate diagnostics; fix the `main_*` / `wrist_*` rows in the same sample instead of selecting `{FINAL_ROLE}` directly.",
        "",
        "## Per-Sample Worklist",
        "",
        "| sample | prepare prompts | train-ready prompts | train-ready targets | recommended frames | task |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in report["sample_rows"]:
        targets = row["train_ready_targets"].replace("|", "/")
        recommended = row["recommended_frames"].replace("|", "/")
        task_text = str(row.get("task_text") or "").replace("|", "/")
        lines.append(
            f"| `{row['sample_name']}` | {row['prepare_gate_prompt_count']} | {row['train_ready_prompt_count']} | {targets} | {recommended} | {task_text} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks_payload = load_json(args.tasks)
    tasks_payload["tasks_path"] = args.tasks
    verification = load_json(args.verification) if args.verification and Path(args.verification).exists() else None
    report = build_report(tasks_payload, verification)
    write_json(args.output_json, report)
    write_csv(args.output_csv, report["sample_rows"])
    write_markdown(args.output_md, report)
    print(
        "tasks={tasks} actionable={actionable} prepare_prompts={prepare} train_ready_prompts={train_ready}".format(
            tasks=report["task_count"],
            actionable=report["actionable_task_count"],
            prepare=report["prepare_gate"]["unique_prompt_count"],
            train_ready=report["train_ready_gate"]["unique_prompt_count"],
        )
    )
    print(args.output_json)
    print(args.output_csv)
    print(args.output_md)


if __name__ == "__main__":
    main()
