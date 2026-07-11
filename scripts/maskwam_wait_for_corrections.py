#!/usr/bin/env python3
"""Wait for browser-exported MaskWAM corrections, then run import/preflight."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = Path("artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--downloaded_corrections", default=None)
    parser.add_argument("--downloads_dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--timeout_sec", type=float, default=0.0, help="0 means wait forever.")
    parser.add_argument("--poll_sec", type=float, default=2.0)
    parser.add_argument("--stable_sec", type=float, default=2.0)
    parser.add_argument(
        "--accept_existing",
        action="store_true",
        help="Accept an existing corrections file instead of requiring a file modified after this watcher starts.",
    )
    parser.add_argument("--run_name", default="repropagated_qa16_preflight")
    parser.add_argument("--full_run_name", default="repropagated_qa16")
    parser.add_argument("--run_full_after_preflight", action="store_true")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--sam3_version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--remote_host", default="7779")
    parser.add_argument("--remote_repo", default="/mnt/workspace1/users/niejunnan/codebase/Motus")
    parser.add_argument("--remote_python", default="/mnt/workspace1/users/niejunnan/envs/sam3/bin/python")
    parser.add_argument("--remote_sam3_repo", default="/mnt/workspace1/users/niejunnan/codebase/sam3")
    parser.add_argument("--hf_home", default="/mnt/workspace1/users/niejunnan/cache/huggingface")
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument("--allow_incomplete_actionable", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def candidate_paths(args: argparse.Namespace, review_dir: Path) -> List[Path]:
    if args.downloaded_corrections:
        return [Path(args.downloaded_corrections).expanduser()]

    downloads_dir = Path(args.downloads_dir).expanduser()
    patterns = [
        str(review_dir / "corrections_from_html.json"),
        str(downloads_dir / "corrections_from_html.json"),
        str(downloads_dir / "corrections_from_html (*).json"),
    ]
    paths = [Path(path) for pattern in patterns for path in glob.glob(pattern)]
    unique = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def active_prompt_count(path: Path) -> int:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), dict):
        raise ValueError("corrections JSON must contain a samples object")
    active = 0
    for sample in payload.get("samples", {}).values():
        if not isinstance(sample, dict):
            continue
        for prompt in sample.get("prompts", []):
            if not isinstance(prompt, dict):
                continue
            if (
                prompt.get("points_abs")
                or prompt.get("points_rel")
                or prompt.get("boxes_xywh_abs")
                or prompt.get("boxes_xywh_rel")
            ):
                active += 1
    return active


def newest_existing(paths: Sequence[Path], start_time: float, accept_existing: bool) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not accept_existing:
        existing = [path for path in existing if path.stat().st_mtime >= start_time]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def is_stable(path: Path, stable_sec: float) -> bool:
    first = path.stat()
    time.sleep(max(0.0, stable_sec))
    if not path.exists():
        return False
    second = path.stat()
    return first.st_size == second.st_size and first.st_mtime == second.st_mtime


def wait_for_corrections(args: argparse.Namespace, review_dir: Path, start_time: float) -> Path:
    deadline = None if args.timeout_sec <= 0 else start_time + args.timeout_sec
    while True:
        selected = newest_existing(
            candidate_paths(args, review_dir),
            start_time=start_time,
            accept_existing=bool(args.accept_existing or args.downloaded_corrections),
        )
        if selected is not None:
            print(f"candidate={selected}", flush=True)
            if is_stable(selected, args.stable_sec):
                active = active_prompt_count(selected)
                print(f"stable corrections found: {selected} active_prompts={active}", flush=True)
                return selected
            print("candidate is still changing; keep waiting", flush=True)

        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for corrections_from_html.json after {args.timeout_sec} seconds")
        time.sleep(max(0.25, args.poll_sec))


def build_import_command(args: argparse.Namespace, corrections: Path) -> List[str]:
    command = [
        sys.executable,
        "scripts/maskwam_import_and_preflight_corrections.py",
        "--review_dir",
        args.review_dir,
        "--downloaded_corrections",
        str(corrections),
        "--remote_host",
        args.remote_host,
        "--remote_repo",
        args.remote_repo,
        "--remote_python",
        args.remote_python,
        "--remote_sam3_repo",
        args.remote_sam3_repo,
        "--hf_home",
        args.hf_home,
        "--run_name",
        args.run_name,
        "--full_run_name",
        args.full_run_name,
        "--cuda_visible_devices",
        args.cuda_visible_devices,
        "--sam3_version",
        args.sam3_version,
    ]
    if args.run_full_after_preflight:
        command.append("--run_full_after_preflight")
    if args.allow_incomplete_actionable:
        command.append("--allow_incomplete_actionable")
    if args.dry_run:
        command.append("--dry_run")
    if args.sample_ids:
        command.extend(["--sample_ids", *[str(value) for value in args.sample_ids]])
    return command


def main() -> None:
    args = parse_args()
    review_dir = (REPO_ROOT / args.review_dir).resolve()
    summary_path = (
        Path(args.output_json)
        if args.output_json
        else review_dir / "corrections_wait_summary.json"
    )
    start_time = time.time()
    summary: Dict[str, Any] = {
        "schema_version": "maskwam_corrections_wait_summary_v1",
        "status": "complete",
        "error": None,
        "started_at_unix": start_time,
        "review_dir": str(review_dir),
        "selected_corrections": None,
        "command": None,
        "returncode": None,
        "dry_run": bool(args.dry_run),
    }

    try:
        corrections = wait_for_corrections(args, review_dir, start_time=start_time)
        summary["selected_corrections"] = str(corrections.resolve())
        command = build_import_command(args, corrections)
        summary["command"] = command
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        if args.dry_run:
            summary["returncode"] = None
        else:
            completed = subprocess.run(command, check=False, cwd=REPO_ROOT)
            summary["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"import/preflight command failed with return code {completed.returncode}")
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        print(f"summary={summary_path}", flush=True)
        print(f"ERROR {exc}", flush=True)
        raise SystemExit(1)

    write_json(summary_path, summary)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
