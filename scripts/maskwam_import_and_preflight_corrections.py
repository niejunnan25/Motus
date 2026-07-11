#!/usr/bin/env python3
"""Import browser-exported MaskWAM corrections, sync to 7779, and run preflight."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--downloaded_corrections", default=None)
    parser.add_argument("--remote_host", default="7779")
    parser.add_argument("--remote_repo", default="/mnt/workspace1/users/niejunnan/codebase/Motus")
    parser.add_argument("--remote_python", default="/mnt/workspace1/users/niejunnan/envs/sam3/bin/python")
    parser.add_argument("--remote_sam3_repo", default="/mnt/workspace1/users/niejunnan/codebase/sam3")
    parser.add_argument("--hf_home", default="/mnt/workspace1/users/niejunnan/cache/huggingface")
    parser.add_argument("--run_name", default="repropagated_qa16_preflight")
    parser.add_argument("--full_run_name", default="repropagated_qa16")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--skip_sync", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument(
        "--run_full_after_preflight",
        action="store_true",
        help="Run the full GPU re-propagation command only after preflight succeeds.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--allow_incomplete_actionable",
        action="store_true",
        help="Do not pass --require_all_actionable. Intended only for debugging.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_summaries(paths: Sequence[Path], payload: Dict[str, Any]) -> None:
    for path in paths:
        write_json(path, payload)


def newest(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def find_corrections(review_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"downloaded corrections not found: {path}")
        return path

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
    selected = newest([path for path in candidates if path.exists()])
    if selected is not None:
        return selected

    raise FileNotFoundError(
        "No corrections_from_html JSON found. Pass --downloaded_corrections or download from the review HTML first."
    )


def validate_corrections(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    issues: List[str] = []
    if not isinstance(payload, dict):
        issues.append("corrections JSON root must be an object")
    elif not isinstance(payload.get("samples"), dict):
        issues.append("corrections JSON must contain a samples object")
    active = 0
    if not issues:
        for sample in payload.get("samples", {}).values():
            for prompt in sample.get("prompts", []):
                if (
                    prompt.get("points_abs")
                    or prompt.get("points_rel")
                    or prompt.get("boxes_xywh_abs")
                    or prompt.get("boxes_xywh_rel")
                ):
                    active += 1
    if issues:
        raise ValueError("; ".join(issues))
    return {
        "schema_version": payload.get("schema_version"),
        "sample_count": len(payload.get("samples", {})),
        "active_prompt_count": active,
    }


def run_command(command: List[str], dry_run: bool) -> Dict[str, Any]:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, check=False)
    return {"command": command, "returncode": completed.returncode, "dry_run": False}


def relative_to_repo(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT.resolve())


def build_remote_loop_command(
    args: argparse.Namespace,
    review_rel: Path,
    remote_corrections_rel: Path,
    run_name: str,
    stop_after_prepare: bool,
) -> List[str]:
    full_cmd = [
        args.remote_python,
        "scripts/maskwam_run_full_correction_loop.py",
        "--review_dir",
        str(review_rel),
        "--downloaded_corrections",
        str(remote_corrections_rel),
        "--run_name",
        run_name,
        "--python",
        args.remote_python,
        "--cuda_visible_devices",
        args.cuda_visible_devices,
        "--sam3_version",
        args.sam3_version,
    ]
    if stop_after_prepare:
        full_cmd.append("--stop_after_prepare")
    if not args.allow_incomplete_actionable:
        full_cmd.append("--require_all_actionable")
    if args.sample_ids:
        full_cmd.extend(["--sample_ids", *[str(value) for value in args.sample_ids]])

    lines = [
        f"cd {shlex.quote(args.remote_repo)}",
        f"export HF_HOME={shlex.quote(args.hf_home)}",
        "export HF_HUB_OFFLINE=1",
        f"export PYTHONPATH={shlex.quote(args.remote_sam3_repo)}:$PYTHONPATH",
        " ".join(shlex.quote(part) for part in full_cmd),
    ]
    return ["ssh", args.remote_host, "bash -lc " + shlex.quote("\n".join(lines))]


def main() -> None:
    args = parse_args()
    if args.run_full_after_preflight and args.skip_preflight:
        raise SystemExit("--run_full_after_preflight requires preflight; remove --skip_preflight.")

    review_dir = (REPO_ROOT / args.review_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    destination = review_dir / "corrections_from_html.json"
    latest_summary_path = review_dir / "corrections_import_preflight_summary.json"
    run_summary_name = args.full_run_name if args.run_full_after_preflight else args.run_name
    run_summary_path = review_dir.parent / "corrected_runs" / run_summary_name / "local_import_preflight_summary.json"
    summary_paths = [latest_summary_path, run_summary_path]
    summary: Dict[str, Any] = {
        "schema_version": "maskwam_import_preflight_summary_v1",
        "status": "complete",
        "error": None,
        "dry_run": bool(args.dry_run),
        "review_dir": str(review_dir),
        "local_corrections": str(destination),
        "remote_host": args.remote_host,
        "remote_repo": args.remote_repo,
        "run_name": args.run_name,
        "full_run_name": args.full_run_name,
        "latest_summary": str(latest_summary_path),
        "run_summary": str(run_summary_path),
        "steps": [],
    }

    try:
        source = find_corrections(review_dir, args.downloaded_corrections)
        validation = validate_corrections(source)
        summary["source_corrections"] = str(source.resolve())
        summary["validation"] = validation

        if source.resolve() != destination.resolve():
            print(f"import {source} -> {destination}", flush=True)
            if not args.dry_run:
                shutil.copyfile(source, destination)
        else:
            print(f"use existing {destination}", flush=True)

        review_rel = relative_to_repo(review_dir)
        local_corrections_rel = relative_to_repo(destination)
        remote_corrections_abs = str(Path(args.remote_repo) / local_corrections_rel)
        summary["remote_corrections"] = remote_corrections_abs

        if not args.skip_sync:
            rsync_cmd = [
                "rsync",
                "-av",
                str(destination),
                f"{args.remote_host}:{remote_corrections_abs}",
            ]
            result = run_command(rsync_cmd, args.dry_run)
            summary["steps"].append({"step": "rsync corrections", **result})
            if result["returncode"] not in (0, None):
                raise RuntimeError(f"rsync failed with return code {result['returncode']}")

        if not args.skip_preflight:
            remote_cmd = build_remote_loop_command(
                args,
                review_rel,
                local_corrections_rel,
                run_name=args.run_name,
                stop_after_prepare=True,
            )
            result = run_command(remote_cmd, args.dry_run)
            summary["steps"].append({"step": "remote preflight", **result})
            if result["returncode"] not in (0, None):
                raise RuntimeError(f"remote preflight failed with return code {result['returncode']}")

        if args.run_full_after_preflight:
            remote_cmd = build_remote_loop_command(
                args,
                review_rel,
                local_corrections_rel,
                run_name=args.full_run_name,
                stop_after_prepare=False,
            )
            result = run_command(remote_cmd, args.dry_run)
            summary["steps"].append({"step": "remote full repropagate", **result})
            if result["returncode"] not in (0, None):
                raise RuntimeError(f"remote full repropagate failed with return code {result['returncode']}")

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_summaries(summary_paths, summary)
        print(f"summary={latest_summary_path}", flush=True)
        print(f"run_summary={run_summary_path}", flush=True)
        print(f"ERROR {exc}", flush=True)
        raise SystemExit(1)

    write_summaries(summary_paths, summary)
    print(f"summary={latest_summary_path}", flush=True)
    print(f"run_summary={run_summary_path}", flush=True)


if __name__ == "__main__":
    main()
