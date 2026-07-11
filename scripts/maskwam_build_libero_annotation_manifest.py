#!/usr/bin/env python3
"""Build a MaskWAM-style LIBERO mask annotation manifest from fixed VGM windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VALID_ROLES = {"object", "target", "robot", "context"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/maskwam_libero_annotation_debug16.json",
        help="Annotation pipeline config.",
    )
    parser.add_argument("--windows_json", default=None, help="Override windows JSON path.")
    parser.add_argument("--output", default=None, help="Output manifest path.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def normalize_video_paths(window: Dict[str, Any]) -> List[str]:
    paths = window.get("video_paths", window.get("video_path"))
    if paths is None:
        raise KeyError("Window must contain video_path or video_paths")
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("Window video_paths is empty")
    return [str(path) for path in paths]


def validate_prompt(prompt: Dict[str, Any], sample_id: int, prompt_idx: int) -> Dict[str, Any]:
    role = prompt.get("role")
    if role not in VALID_ROLES:
        raise ValueError(f"sample {sample_id} prompt {prompt_idx} has invalid role={role!r}")
    text = str(prompt.get("text", "")).strip()
    if not text:
        raise ValueError(f"sample {sample_id} prompt {prompt_idx} is missing text")
    name = str(prompt.get("name") or text).strip()
    return {
        "name": name,
        "role": role,
        "text": text,
        "source": str(prompt.get("source", "config")),
    }


def build_manifest(config: Dict[str, Any], windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_size = config.get("target_size", {})
    height = int(target_size.get("height", 448))
    width = int(target_size.get("width", 224))
    sample_ids = [int(value) for value in config.get("sample_ids", range(len(windows)))]
    sample_cfgs = config.get("samples", {})

    samples = []
    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= len(windows):
            raise IndexError(f"sample_id={sample_id} out of range for {len(windows)} windows")
        window = dict(windows[sample_id])
        sample_cfg = sample_cfgs.get(str(sample_id), {})
        prompts = [
            validate_prompt(prompt, sample_id, idx)
            for idx, prompt in enumerate(sample_cfg.get("prompts", []))
        ]
        if not prompts:
            raise ValueError(f"No prompts configured for sample {sample_id}")
        frame_indices = [int(idx) for idx in window.get("frame_indices", [])]
        if not frame_indices:
            raise ValueError(f"Window {sample_id} has no frame_indices")

        samples.append(
            {
                "sample_id": sample_id,
                "sample_name": f"sample_{sample_id:03d}",
                "task_text": str(window.get("task_text", "")),
                "frame_indices": frame_indices,
                "video_paths": normalize_video_paths(window),
                "prompts": prompts,
                "target_size": {"height": height, "width": width},
                "view_layout": str(config.get("view_layout", "vertical")),
                "review_frame_indices": [int(idx) for idx in config.get("review_frame_indices", [0, 4, 8, 12, 16])],
                "window": window,
            }
        )

    return {
        "schema_version": config.get("schema_version", "maskwam_libero_annotation_v1"),
        "description": config.get("description", ""),
        "mask_semantics": config.get("mask_semantics", {}),
        "review_statuses": config.get("review_statuses", []),
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    config_path = resolve_path(args.config, repo_root)
    config = load_json(config_path)
    windows_path = resolve_path(args.windows_json or config["windows_json"], repo_root)
    windows = load_json(windows_path)
    if not isinstance(windows, list):
        raise TypeError(f"Expected {windows_path} to contain a list")

    manifest = build_manifest(config, windows)
    manifest["config_path"] = str(config_path.resolve())
    manifest["windows_json"] = str(windows_path.resolve())

    output = args.output
    if output is None:
        output_root = resolve_path(config.get("output_root", "artifacts/maskwam_libero_annotation_debug16"), repo_root)
        output_path = output_root / "manifest.json"
    else:
        output_path = resolve_path(output, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(output_path)


if __name__ == "__main__":
    main()
