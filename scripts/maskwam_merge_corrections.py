#!/usr/bin/env python3
"""Merge and normalize MaskWAM correction JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep_empty", action="store_true")
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


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


def normalize_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": prompt.get("name") or "manual point",
        "role": prompt.get("role") or "object",
        "frame_index": int(prompt.get("frame_index", 0)),
        "points_abs": list_value(prompt.get("points_abs")),
        "point_labels": list_value(prompt.get("point_labels")),
        "boxes_xywh_abs": list_value(prompt.get("boxes_xywh_abs")),
        "box_labels": list_value(prompt.get("box_labels")),
        **({"points_rel": list_value(prompt.get("points_rel"))} if prompt.get("points_rel") else {}),
        **({"boxes_xywh_rel": list_value(prompt.get("boxes_xywh_rel"))} if prompt.get("boxes_xywh_rel") else {}),
        **({"todo_reason": prompt.get("todo_reason")} if prompt.get("todo_reason") else {}),
        **({"accepted_absent": True} if prompt.get("accepted_absent") else {}),
        **({"absent_reason": prompt.get("absent_reason")} if prompt.get("absent_reason") else {}),
        **({"force_manual_mask": True} if prompt.get("force_manual_mask") else {}),
    }


def prompt_key(prompt: Dict[str, Any]) -> Tuple[str, str, int, str, str, str, str]:
    return (
        str(prompt.get("role", "")),
        str(prompt.get("name", "")),
        int(prompt.get("frame_index", 0)),
        json.dumps(prompt.get("points_abs", []), sort_keys=True),
        json.dumps(prompt.get("point_labels", []), sort_keys=True),
        json.dumps(prompt.get("boxes_xywh_abs", []), sort_keys=True),
        json.dumps(prompt.get("box_labels", []), sort_keys=True),
        str(bool(prompt.get("force_manual_mask"))),
        str(bool(prompt.get("accepted_absent"))),
    )


def merge_payloads(paths: Iterable[str | Path], keep_empty: bool, notes: str | None) -> Dict[str, Any]:
    merged_samples: Dict[str, Dict[str, Any]] = {}
    seen_prompts = set()
    source_paths = [str(Path(path).resolve()) for path in paths]

    for path in paths:
        payload = load_json(path)
        for sample_id, sample_cfg in payload.get("samples", {}).items():
            sample_key = str(sample_id)
            target = merged_samples.setdefault(
                sample_key,
                {
                    "status": sample_cfg.get("status", "needs_correction"),
                    "notes": sample_cfg.get("notes", ""),
                    "prompts": [],
                },
            )
            if sample_cfg.get("status"):
                target["status"] = sample_cfg["status"]
            if sample_cfg.get("notes") and sample_cfg.get("notes") not in target["notes"]:
                target["notes"] = (target["notes"] + " | " + sample_cfg["notes"]).strip(" |")
            for prompt in sample_cfg.get("prompts", []):
                if not isinstance(prompt, dict):
                    continue
                normalized = normalize_prompt(prompt)
                if not keep_empty and not is_active_prompt(normalized):
                    continue
                key = (sample_key, prompt_key(normalized))
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
                target["prompts"].append(normalized)

    for sample_id in list(merged_samples):
        prompts = merged_samples[sample_id]["prompts"]
        prompts.sort(key=lambda item: (int(item.get("frame_index", 0)), str(item.get("role", "")), str(item.get("name", ""))))
        if not prompts and not keep_empty:
            del merged_samples[sample_id]

    return {
        "schema_version": "maskwam_corrections_v1",
        "notes": notes or "Merged MaskWAM corrections. Empty draft prompts are dropped unless --keep_empty is used.",
        "sources": source_paths,
        "samples": dict(sorted(merged_samples.items(), key=lambda item: int(item[0]))),
    }


def main() -> None:
    args = parse_args()
    merged = merge_payloads(args.inputs, args.keep_empty, args.notes)
    write_json(args.output, merged)
    active_count = sum(
        1
        for sample in merged["samples"].values()
        for prompt in sample.get("prompts", [])
        if is_active_prompt(prompt)
    )
    prompt_count = sum(len(sample.get("prompts", [])) for sample in merged["samples"].values())
    print(args.output)
    print(f"samples={len(merged['samples'])} prompts={prompt_count} active={active_count}")


if __name__ == "__main__":
    main()
