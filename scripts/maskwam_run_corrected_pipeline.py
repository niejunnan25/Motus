#!/usr/bin/env python3
"""Run the corrected MaskWAM annotation loop end to end."""

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
    parser.add_argument("--corrections", required=True)
    parser.add_argument(
        "--correction_tasks",
        default="artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_root", default="artifacts/maskwam_libero_annotation_debug16/corrected_runs")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--allow_empty_corrections", action="store_true")
    parser.add_argument("--skip_correction_coverage", action="store_true")
    parser.add_argument("--coverage_require_all_critical", action="store_true")
    parser.add_argument("--coverage_require_all_high", action="store_true")
    parser.add_argument("--coverage_require_all_actionable", action="store_true")
    parser.add_argument("--require_review_ok", action="store_true")
    parser.add_argument("--allow_final_zero", action="store_true")
    parser.add_argument("--allow_role_zero", action="store_true")
    parser.add_argument("--allow_unapproved_fallbacks", action="store_true")
    parser.add_argument("--skip_context", action="store_true", default=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def default_run_name(corrections: Path) -> str:
    stem = corrections.stem
    return stem.replace("corrections_template", "corrected").replace("correction", "corrected")


def run_command(command: List[str], env: Dict[str, str], step_name: str) -> Dict[str, Any]:
    print(f"\n### {step_name}", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, env=env, check=False)
    result = {
        "step": step_name,
        "command": command,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise StepFailed(f"{step_name} failed with return code {completed.returncode}", result)
    return result


def sample_count(manifest_path: Path, sample_ids: List[int] | None) -> int:
    if sample_ids is not None:
        return len(sample_ids)
    manifest = load_json(manifest_path)
    return len(manifest["samples"])


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    corrections = Path(args.corrections)
    run_name = args.run_name or default_run_name(corrections)
    run_dir = Path(args.output_root) / run_name
    annotations_dir = run_dir / "annotations"
    review_dir = run_dir / "review"
    validation_json = run_dir / "correction_validation.json"
    coverage_json = run_dir / "correction_coverage.json"
    coverage_csv = run_dir / "correction_coverage.csv"
    verification_json = review_dir / "verification.json"
    verification_md = review_dir / "verification.md"
    train_manifest = run_dir / "train_mask_sequences.json"
    summary_json = run_dir / "pipeline_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    py = args.python
    sample_args: List[str] = []
    if args.sample_ids is not None:
        sample_args = ["--sample_ids", *[str(value) for value in args.sample_ids]]
    expected = sample_count(manifest, args.sample_ids)

    steps: List[Dict[str, Any]] = []
    exit_code = 0
    try:
        correction_tasks = Path(args.correction_tasks) if args.correction_tasks else None
        if not args.skip_correction_coverage and correction_tasks is not None:
            if correction_tasks.exists():
                coverage_cmd = [
                    py,
                    "scripts/maskwam_check_correction_coverage.py",
                    "--tasks",
                    str(correction_tasks),
                    "--corrections",
                    str(corrections),
                    "--output_json",
                    str(coverage_json),
                    "--output_csv",
                    str(coverage_csv),
                    *sample_args,
                ]
                if args.coverage_require_all_critical:
                    coverage_cmd.append("--require_all_critical")
                if args.coverage_require_all_high:
                    coverage_cmd.append("--require_all_high")
                if args.coverage_require_all_actionable:
                    coverage_cmd.append("--require_all_actionable")
                steps.append(run_command(coverage_cmd, env, "check correction coverage"))
            else:
                print(f"\n### check correction coverage\nskip missing task file: {correction_tasks}", flush=True)

        validate_cmd = [
            py,
            "scripts/maskwam_validate_corrections.py",
            "--manifest",
            str(manifest),
            "--corrections",
            str(corrections),
            "--output_json",
            str(validation_json),
        ]
        if not args.allow_empty_corrections:
            validate_cmd.append("--require_active")
        steps.append(run_command(validate_cmd, env, "validate corrections"))

        annotate_cmd = [
            py,
            "scripts/maskwam_run_sam3_video_annotation.py",
            "--manifest",
            str(manifest),
            "--corrections",
            str(corrections),
            "--output_dir",
            str(annotations_dir),
            "--sam3_version",
            args.sam3_version,
            *sample_args,
        ]
        if args.skip_context:
            annotate_cmd.append("--skip_context")
        steps.append(run_command(annotate_cmd, env, "sam3 re-propagate"))

        steps.append(
            run_command(
                [
                    py,
                    "scripts/maskwam_make_annotation_review.py",
                    "--manifest",
                    str(manifest),
                    "--annotation_dir",
                    str(annotations_dir),
                    "--output_dir",
                    str(review_dir),
                ],
                env,
                "make review pack",
            )
        )

        steps.append(
            run_command(
                [
                    py,
                    "scripts/maskwam_verify_annotation_run.py",
                    "--manifest",
                    str(manifest),
                    "--annotation_dir",
                    str(annotations_dir),
                    "--review_dir",
                    str(review_dir),
                    "--expected_samples",
                    str(expected),
                    *sample_args,
                    "--output_json",
                    str(verification_json),
                    "--output_md",
                    str(verification_md),
                ],
                env,
                "verify annotation run",
            )
        )

        export_cmd = [
            py,
            "scripts/maskwam_export_train_mask_sequences.py",
            "--manifest",
            str(manifest),
            "--annotation_dir",
            str(annotations_dir),
            "--review_csv",
            str(review_dir / "human_review_template.csv"),
            "--output_json",
            str(train_manifest),
            *sample_args,
        ]
        if args.require_review_ok:
            export_cmd.append("--require_review_ok")
        if args.allow_final_zero:
            export_cmd.append("--allow_final_zero")
        if args.allow_role_zero:
            export_cmd.append("--allow_role_zero")
        if args.allow_unapproved_fallbacks:
            export_cmd.append("--allow_unapproved_fallbacks")
        steps.append(run_command(export_cmd, env, "export train mask sequences"))
        status = "complete"
        error = None
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
            "status": status,
            "error": error,
            "manifest": str(manifest.resolve()),
            "corrections": str(corrections.resolve()),
            "correction_tasks": str(Path(args.correction_tasks).resolve()) if args.correction_tasks else None,
            "run_dir": str(run_dir.resolve()),
            "annotations_dir": str(annotations_dir.resolve()),
            "review_dir": str(review_dir.resolve()),
            "validation_json": str(validation_json.resolve()),
            "coverage_json": str(coverage_json.resolve()),
            "coverage_csv": str(coverage_csv.resolve()),
            "verification_json": str(verification_json.resolve()),
            "train_manifest": str(train_manifest.resolve()),
            "sample_ids": args.sample_ids,
            "expected_samples": expected,
            "steps": steps,
        }
        write_json(summary_json, summary)
        print(f"\nsummary: {summary_json}", flush=True)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
