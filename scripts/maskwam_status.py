#!/usr/bin/env python3
"""Report the current status of the MaskWAM LIBERO annotation pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from maskwam_check_correction_coverage import build_report as build_coverage_report
from maskwam_merge_corrections import merge_payloads
from maskwam_validate_corrections import build_report as build_validation_report
from maskwam_view_schema import VIEW_TRAIN_ROLES


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")
DEFAULT_MANIFEST = Path("artifacts/maskwam_libero_annotation_debug16/manifest.json")
DEFAULT_REMOTE_REPO = "/mnt/workspace1/users/niejunnan/codebase/Motus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--downloaded_corrections", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--draft_corrections", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--check_remote", action="store_true")
    parser.add_argument("--remote_host", default="7779")
    parser.add_argument("--remote_repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--remote_run_name", default="repropagated_qa16")
    parser.add_argument("--ssh_timeout", type=int, default=20)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def newest(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_corrections(review_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"downloaded corrections not found: {path}")

    review_export = review_dir / "corrections_from_html.json"
    if review_export.exists():
        return review_export

    candidates = [
        Path(path)
        for pattern in [
            str(Path.home() / "Downloads" / "corrections_from_html.json"),
            str(Path.home() / "Downloads" / "corrections_from_html (*).json"),
        ]
        for path in glob.glob(pattern)
    ]
    return newest(candidates)


def build_local_status(
    manifest_path: Path,
    review_dir: Path,
    tasks_path: Path,
    draft_path: Path,
    corrections_path: Path | None,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "review_dir": str(review_dir.resolve()),
        "tasks": str(tasks_path.resolve()),
        "draft_corrections": str(draft_path.resolve()) if draft_path.exists() else None,
        "corrections_source": str(corrections_path.resolve()) if corrections_path else None,
        "pending_summary": str((review_dir / "pending_corrections_summary.md").resolve()),
        "has_browser_export": corrections_path is not None,
        "ready_for_preflight": False,
        "ready_for_formal_repropagate": False,
        "validation": None,
        "coverage": None,
    }

    if not manifest_path.exists():
        status["error"] = f"manifest not found: {manifest_path}"
        return status
    if not tasks_path.exists():
        status["error"] = f"correction tasks not found: {tasks_path}"
        return status

    tasks_payload = load_json(tasks_path)
    manifest = load_json(manifest_path)
    task_count = int(tasks_payload.get("task_count", len(tasks_payload.get("tasks", []))))
    actionable_roles = set(VIEW_TRAIN_ROLES)
    actionable_count = sum(
        1
        for task in tasks_payload.get("tasks", [])
        if task.get("role") in actionable_roles and task.get("name")
    )
    status["task_count"] = task_count
    status["actionable_task_count"] = actionable_count
    status["sample_count_with_tasks"] = int(tasks_payload.get("sample_with_task_count", 0))

    if corrections_path is None:
        return status

    merge_inputs = [path for path in [draft_path, corrections_path] if path.exists()]
    if not merge_inputs:
        status["error"] = "no correction merge inputs found"
        return status

    merged = merge_payloads(
        merge_inputs,
        keep_empty=False,
        notes="Temporary status merge; not written back to the review directory.",
    )
    validation = build_validation_report(manifest, merged, require_active=True)
    coverage = build_coverage_report(tasks_payload, merged, allow_manual_role_match=True)
    coverage_summary = coverage["summary"]
    status["validation"] = {
        "status": validation["status"],
        "active_prompt_count": validation["active_prompt_count"],
        "issue_count": len(validation.get("issues", [])),
    }
    status["coverage"] = {
        "status": coverage["status"],
        "actionable": {
            "covered": coverage_summary["covered_actionable_task_count"],
            "total": coverage_summary["actionable_task_count"],
            "uncovered": coverage_summary["uncovered_actionable_task_count"],
        },
        "aggregate": {
            "covered": coverage_summary["covered_aggregate_task_count"],
            "total": coverage_summary["aggregate_task_count"],
        },
        "samples_with_uncovered_actionable": coverage_summary["samples_with_uncovered_actionable"],
        "unmatched_active_correction_count": coverage_summary["unmatched_active_correction_count"],
    }
    status["ready_for_preflight"] = (
        validation["status"] == "valid"
        and validation["active_prompt_count"] > 0
    )
    status["ready_for_formal_repropagate"] = (
        status["ready_for_preflight"]
        and coverage_summary["uncovered_actionable_task_count"] == 0
    )
    status["uncovered_actionable_preview"] = coverage.get("uncovered_actionable", [])[:20]
    return status


def build_remote_status(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = (
        Path(args.remote_repo)
        / "artifacts/maskwam_libero_annotation_debug16/corrected_runs"
        / args.remote_run_name
    )
    remote_code = (
        "import json; from pathlib import Path; "
        f"run_dir=Path({str(run_dir)!r}); "
        "audit=run_dir/'audit.json'; train=run_dir/'train_mask_sequences.json'; "
        "payload={'run_dir':str(run_dir),'audit_exists':audit.exists(),"
        "'train_mask_sequences_exists':train.exists()}; "
        "payload['audit_status']=json.load(open(audit)).get('status') if audit.exists() else None; "
        "print(json.dumps(payload))"
    )
    command = [
        "ssh",
        args.remote_host,
        f"python3 -c {shlex.quote(remote_code)}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.ssh_timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "ssh_timeout", "run_dir": str(run_dir)}

    if completed.returncode != 0:
        return {
            "status": "ssh_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "run_dir": str(run_dir),
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "parse_failed",
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "run_dir": str(run_dir),
        }
    payload["status"] = "ready" if (
        payload.get("audit_status") == "ready"
        and payload.get("train_mask_sequences_exists")
    ) else "not_ready"
    return payload


def print_status(report: Dict[str, Any]) -> None:
    local = report["local"]
    print("MaskWAM LIBERO annotation status")
    print(f"review_dir={local['review_dir']}")
    print(f"tasks={local.get('task_count', 'unknown')} actionable={local.get('actionable_task_count', 'unknown')}")
    print(f"corrections_source={local.get('corrections_source') or 'MISSING'}")

    validation = local.get("validation")
    coverage = local.get("coverage")
    if validation:
        print(
            "validation={status} active_prompts={active} issues={issues}".format(
                status=validation["status"],
                active=validation["active_prompt_count"],
                issues=validation["issue_count"],
            )
        )
    if coverage:
        actionable = coverage["actionable"]
        print(
            "coverage={status} actionable={covered}/{total} uncovered={uncovered} unmatched_active={unmatched}".format(
                status=coverage["status"],
                covered=actionable["covered"],
                total=actionable["total"],
                uncovered=actionable["uncovered"],
                unmatched=coverage["unmatched_active_correction_count"],
            )
        )
        if actionable["uncovered"]:
            print(f"samples_with_uncovered={coverage['samples_with_uncovered_actionable']}")

    if local.get("ready_for_formal_repropagate"):
        print("next=run formal preflight + SAM3 re-propagation")
    elif local.get("ready_for_preflight"):
        print("next=more corrections are needed before formal --require_all_actionable re-propagation")
    else:
        print("next=finish the HTML review and download corrections_from_html.json")
    print(f"pending_summary={local.get('pending_summary')}")

    remote = report.get("remote")
    if remote:
        print(
            "remote={status} run_dir={run_dir} audit={audit} train_masks={train}".format(
                status=remote.get("status"),
                run_dir=remote.get("run_dir"),
                audit=remote.get("audit_status"),
                train=remote.get("train_mask_sequences_exists"),
            )
        )


def main() -> None:
    args = parse_args()
    review_dir = (REPO_ROOT / args.review_dir).resolve()
    manifest_path = (REPO_ROOT / args.manifest).resolve()
    tasks_path = (REPO_ROOT / args.tasks).resolve() if args.tasks else review_dir / "correction_tasks.json"
    draft_path = (REPO_ROOT / args.draft_corrections).resolve() if args.draft_corrections else review_dir / "corrections_tasks_draft.json"
    corrections_path = find_corrections(review_dir, args.downloaded_corrections)

    report = {
        "schema_version": "maskwam_status_v1",
        "local": build_local_status(
            manifest_path=manifest_path,
            review_dir=review_dir,
            tasks_path=tasks_path,
            draft_path=draft_path,
            corrections_path=corrections_path,
        ),
    }
    if args.check_remote:
        report["remote"] = build_remote_status(args)

    if args.output_json:
        write_json(args.output_json, report)
    print_status(report)

    local = report["local"]
    if local.get("error"):
        print(f"ERROR {local['error']}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
