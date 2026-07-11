#!/usr/bin/env python3
"""Run the full MaskWAM correction loop after human review corrections are exported."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class StepFailed(RuntimeError):
    def __init__(self, message: str, result: Dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/maskwam_libero_annotation_debug16/manifest.json")
    parser.add_argument("--review_dir", default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")
    parser.add_argument("--downloaded_corrections", default=None)
    parser.add_argument("--before_annotation_dir", default="artifacts/maskwam_libero_annotation_debug16/first_pass_qa16")
    parser.add_argument("--run_name", default="repropagated_qa16")
    parser.add_argument("--output_root", default="artifacts/maskwam_libero_annotation_debug16/corrected_runs")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--allow_uncovered_critical", action="store_true")
    parser.add_argument("--require_all_high", action="store_true")
    parser.add_argument("--require_all_actionable", action="store_true")
    parser.add_argument("--allow_train_warnings", action="store_true")
    parser.add_argument("--require_review_approved", action="store_true")
    parser.add_argument("--expected_samples", type=int, default=None)
    parser.add_argument("--stop_after_prepare", action="store_true")
    parser.add_argument("--skip_comparison", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def run_command(command: List[str], env: Dict[str, str], step_name: str, dry_run: bool) -> Dict[str, Any]:
    print(f"\n### {step_name}", flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return {
            "step": step_name,
            "command": command,
            "returncode": None,
            "dry_run": True,
        }
    completed = subprocess.run(command, env=env, check=False)
    result = {
        "step": step_name,
        "command": command,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise StepFailed(f"{step_name} failed with return code {completed.returncode}", result)
    return result


def sample_args(sample_ids: List[int] | None) -> List[str]:
    if not sample_ids:
        return []
    return ["--sample_ids", *[str(value) for value in sample_ids]]


def expected_samples(args: argparse.Namespace) -> int:
    if args.expected_samples is not None:
        return int(args.expected_samples)
    if args.sample_ids:
        return len(args.sample_ids)
    return 16


def maybe_read(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def main() -> None:
    args = parse_args()
    py = args.python
    review_dir = Path(args.review_dir)
    run_dir = Path(args.output_root) / args.run_name
    loop_summary_json = run_dir / "full_correction_loop_summary.json"
    prepare_summary_json = run_dir / "corrections_prepare_summary.json"
    tasks_path = review_dir / "correction_tasks.json"
    merged_corrections = run_dir / "corrections_merged.json"
    comparison_dir = run_dir / "comparison_review"
    expected = expected_samples(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    steps: List[Dict[str, Any]] = []
    status = "complete"
    error = None
    exit_code = 0

    try:
        prepare_cmd = [
            py,
            "scripts/maskwam_prepare_corrections_for_repropagate.py",
            "--manifest",
            args.manifest,
            "--review_dir",
            str(review_dir),
            "--output_corrections",
            str(merged_corrections),
            "--summary_json",
            str(prepare_summary_json),
            "--next_run_name",
            args.run_name,
            "--next_output_root",
            args.output_root,
            "--next_python",
            py,
            "--next_sam3_version",
            args.sam3_version,
            *sample_args(args.sample_ids),
        ]
        if args.cuda_visible_devices is not None:
            prepare_cmd.extend(["--next_cuda_visible_devices", args.cuda_visible_devices])
        if args.downloaded_corrections:
            prepare_cmd.extend(["--downloaded_corrections", args.downloaded_corrections])
        if not args.allow_uncovered_critical:
            prepare_cmd.append("--require_all_critical")
        if args.require_all_high:
            prepare_cmd.append("--require_all_high")
        if args.require_all_actionable:
            prepare_cmd.append("--require_all_actionable")
        steps.append(run_command(prepare_cmd, env, "prepare corrections", args.dry_run))

        if args.stop_after_prepare:
            status = "stopped_after_prepare"
            return

        corrected_cmd = [
            py,
            "scripts/maskwam_run_corrected_pipeline.py",
            "--manifest",
            args.manifest,
            "--corrections",
            str(merged_corrections),
            "--correction_tasks",
            str(tasks_path),
            "--run_name",
            args.run_name,
            "--output_root",
            args.output_root,
            "--python",
            py,
            "--sam3_version",
            args.sam3_version,
            *sample_args(args.sample_ids),
        ]
        if not args.allow_uncovered_critical:
            corrected_cmd.append("--coverage_require_all_critical")
        if args.require_all_high:
            corrected_cmd.append("--coverage_require_all_high")
        if args.require_all_actionable:
            corrected_cmd.append("--coverage_require_all_actionable")
        if args.cuda_visible_devices is not None:
            corrected_cmd.extend(["--cuda_visible_devices", args.cuda_visible_devices])
        steps.append(run_command(corrected_cmd, env, "run corrected pipeline", args.dry_run))

        if not args.skip_comparison:
            comparison_cmd = [
                py,
                "scripts/maskwam_make_correction_comparison_review.py",
                "--manifest",
                args.manifest,
                "--before_annotation_dir",
                args.before_annotation_dir,
                "--after_annotation_dir",
                str(run_dir / "annotations"),
                "--output_dir",
                str(comparison_dir),
                *sample_args(args.sample_ids),
            ]
            steps.append(run_command(comparison_cmd, env, "make correction comparison review", args.dry_run))

        if not args.skip_audit:
            audit_cmd = [
                py,
                "scripts/maskwam_audit_corrected_run.py",
                "--run_dir",
                str(run_dir),
                "--expected_samples",
                str(expected),
            ]
            if args.allow_train_warnings:
                audit_cmd.append("--allow_train_warnings")
            if args.require_review_approved:
                audit_cmd.append("--require_review_approved")
            steps.append(run_command(audit_cmd, env, "audit corrected run", args.dry_run))

    except StepFailed as exc:
        status = "failed"
        error = str(exc)
        steps.append(exc.result)
        exit_code = 1
    except Exception as exc:
        status = "failed"
        error = str(exc)
        exit_code = 1
    finally:
        summary = {
            "schema_version": "maskwam_full_correction_loop_summary_v1",
            "status": status,
            "error": error,
            "dry_run": bool(args.dry_run),
            "manifest": str(Path(args.manifest).resolve()),
            "review_dir": str(review_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "merged_corrections": str(merged_corrections.resolve()),
            "prepare_summary": str(prepare_summary_json.resolve()),
            "comparison_review": str((comparison_dir / "index.html").resolve()),
            "audit_json": str((run_dir / "audit.json").resolve()),
            "sample_ids": args.sample_ids,
            "expected_samples": expected,
            "steps": steps,
            "prepare": maybe_read(prepare_summary_json) if not args.dry_run else None,
            "pipeline": maybe_read(run_dir / "pipeline_summary.json") if not args.dry_run else None,
            "comparison": maybe_read(comparison_dir / "comparison_summary.json") if not args.dry_run else None,
            "audit": maybe_read(run_dir / "audit.json") if not args.dry_run else None,
        }
        write_json(loop_summary_json, summary)
        print(f"\nsummary: {loop_summary_json}", flush=True)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
