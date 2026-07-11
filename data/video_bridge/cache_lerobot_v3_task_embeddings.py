#!/usr/bin/env python3
"""Cache WAN UMT5 embeddings for LeRobot v3 task metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_wan_to_path() -> None:
    bak_root = str((repo_root() / "bak").resolve())
    if bak_root not in sys.path:
        sys.path.insert(0, bak_root)


def resolve_wan_dir(wan_path: str) -> Path:
    path = Path(wan_path)
    if (path / "models_t5_umt5-xxl-enc-bf16.pth").exists():
        return path
    nested = path / "Wan2.2-TI2V-5B"
    if (nested / "models_t5_umt5-xxl-enc-bf16.pth").exists():
        return nested
    raise FileNotFoundError(
        "Could not find models_t5_umt5-xxl-enc-bf16.pth under "
        f"{path} or {nested}"
    )


def load_task_texts(dataset_root: Path) -> Dict[int, str]:
    import pandas as pd

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"LeRobot v3 tasks parquet not found: {tasks_path}")

    tasks_df = pd.read_parquet(tasks_path)
    task_texts: Dict[int, str] = {}
    for row_index, row in tasks_df.iterrows():
        if "task_index" in row:
            task_index = int(row["task_index"])
        else:
            task_index = int(row_index)

        task_text = None
        for column in ("task", "text", "instruction", "language_instruction"):
            if column in row and row[column] is not None:
                task_text = str(row[column])
                break
        if task_text is None:
            task_text = str(row_index)
        task_texts[task_index] = task_text

    if not task_texts:
        raise RuntimeError(f"No task texts found in {tasks_path}")
    return dict(sorted(task_texts.items()))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_caption_sidecar(
    caption_path: Path,
    caption_field: str,
    dataset_task_texts: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if not caption_path.exists():
        raise FileNotFoundError(f"Task caption sidecar not found: {caption_path}")

    with caption_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"Caption sidecar must contain a top-level tasks list: {caption_path}")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Caption sidecar must contain a non-empty string version: {caption_path}")

    entries: Dict[int, Dict[str, Any]] = {}
    for position, raw_entry in enumerate(payload["tasks"]):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Caption sidecar tasks[{position}] must be an object")
        if "task_index" not in raw_entry:
            raise KeyError(f"Caption sidecar tasks[{position}] is missing task_index")
        task_index = int(raw_entry["task_index"])
        if task_index in entries:
            raise ValueError(f"Duplicate task_index={task_index} in {caption_path}")
        entries[task_index] = raw_entry

    dataset_indices = set(dataset_task_texts)
    sidecar_indices = set(entries)
    missing = sorted(dataset_indices - sidecar_indices)
    extra = sorted(sidecar_indices - dataset_indices)
    if missing or extra:
        raise ValueError(
            f"Caption sidecar task coverage does not match the dataset; missing={missing}, extra={extra}"
        )

    caption_texts: Dict[int, str] = {}
    for task_index, original_text in dataset_task_texts.items():
        entry = entries[task_index]
        sidecar_original = entry.get("original_en")
        if sidecar_original is None:
            raise KeyError(f"Caption sidecar task_index={task_index} is missing original_en")
        if str(sidecar_original) != original_text:
            raise ValueError(
                f"Caption sidecar original_en mismatch for task_index={task_index}: "
                f"{sidecar_original!r} != {original_text!r}"
            )
        caption_text = entry.get(caption_field)
        if not isinstance(caption_text, str) or not caption_text.strip():
            raise ValueError(
                f"Caption sidecar task_index={task_index} has an empty/non-string {caption_field!r}"
            )
        caption_texts[task_index] = caption_text.strip()

    source = {
        "mode": "sidecar",
        "version": version,
        "path": str(caption_path.resolve()),
        "sha256": file_sha256(caption_path),
        "caption_field": caption_field,
    }
    return caption_texts, entries, source


def validate_existing_cache(
    output_dir: Path,
    caption_texts: Dict[int, str],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    existing = [output_dir / f"task_{task_index:06d}.pt" for task_index in caption_texts]
    if not any(path.exists() for path in existing):
        return

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Embedding files already exist under {output_dir}, but manifest.json is missing. "
            "Use a new output directory or pass --overwrite."
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    manifest_texts = {}
    for entry in manifest.get("tasks", []):
        if not isinstance(entry, dict) or "task_index" not in entry:
            continue
        text = entry.get("caption_text", entry.get("task_text"))
        if isinstance(text, str):
            manifest_texts[int(entry["task_index"])] = text

    mismatched = [
        task_index
        for task_index, caption_text in caption_texts.items()
        if manifest_texts.get(task_index) != caption_text
    ]
    if mismatched:
        raise RuntimeError(
            "Existing embeddings were generated from different or untraceable captions for task indices "
            f"{mismatched}. Use a new output directory or pass --overwrite."
        )


def init_t5_encoder(wan_dir: Path, device: str, text_len: int) -> Any:
    add_wan_to_path()
    from wan.modules.t5 import T5EncoderModel

    checkpoint_path = wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = wan_dir / "google" / "umt5-xxl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"T5 checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"T5 tokenizer dir not found: {tokenizer_path}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
    )


def encode_text(encoder: Any, text: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        output = encoder([text], device)
    if isinstance(output, list):
        embedding = output[0]
    elif isinstance(output, torch.Tensor):
        embedding = output
    else:
        raise TypeError(f"Unexpected T5 output type: {type(output)}")

    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    if embedding.ndim != 2:
        raise ValueError(f"Expected [seq, dim] embedding, got shape={tuple(embedding.shape)}")
    return embedding.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache WAN UMT5 task embeddings for LeRobot v3 datasets")
    parser.add_argument("--dataset_root", required=True, help="LeRobot v3 root containing meta/tasks.parquet")
    parser.add_argument(
        "--wan_path",
        default=os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT"),
        help="Wan2.2-TI2V-5B directory, or its parent",
    )
    parser.add_argument("--output_dir", default=None, help="Output dir, default: <dataset_root>/umt5_wan_tasks")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default: cuda if available")
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument(
        "--task_caption_file",
        default=None,
        help="Optional JSON sidecar containing a complete task_index -> caption mapping",
    )
    parser.add_argument(
        "--caption_field",
        default="caption_en",
        help="Text field to encode from each sidecar task entry (default: caption_en)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.wan_path:
        raise ValueError("--wan_path is required unless WAN_PATH or WAN_ROOT is set")

    dataset_root = Path(args.dataset_root)
    if args.task_caption_file and not args.output_dir:
        raise ValueError("--task_caption_file requires an explicit --output_dir outside the dataset root")
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "umt5_wan_tasks"
    if args.task_caption_file:
        try:
            output_dir.resolve().relative_to(dataset_root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Sidecar embeddings must be stored outside the immutable dataset root: {dataset_root}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    wan_dir = resolve_wan_dir(args.wan_path)

    dataset_task_texts = load_task_texts(dataset_root)
    sidecar_entries: Dict[int, Dict[str, Any]] = {}
    if args.task_caption_file:
        task_texts, sidecar_entries, caption_source = load_caption_sidecar(
            Path(args.task_caption_file),
            args.caption_field,
            dataset_task_texts,
        )
    else:
        task_texts = dataset_task_texts
        caption_source = {
            "mode": "dataset_tasks",
            "version": None,
            "path": str((dataset_root / "meta" / "tasks.parquet").resolve()),
            "sha256": file_sha256(dataset_root / "meta" / "tasks.parquet"),
            "caption_field": "task_text",
        }

    validate_existing_cache(output_dir, task_texts, overwrite=args.overwrite)
    missing = [
        task_index
        for task_index in task_texts
        if args.overwrite or not (output_dir / f"task_{task_index:06d}.pt").exists()
    ]

    manifest = {
        "schema_version": 2,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "text_len": args.text_len,
        "wan_path": str(wan_dir.resolve()),
        "embedding_pattern": "task_{task_index:06d}.pt",
        "caption_source": caption_source,
        "tasks": [
            {
                "task_index": task_index,
                "original_task_text": dataset_task_texts[task_index],
                "caption_text": task_text,
                "caption_field": caption_source["caption_field"],
                "embedding_path": str(output_dir / f"task_{task_index:06d}.pt"),
                **(
                    {
                        key: sidecar_entries[task_index][key]
                        for key in ("original_zh", "caption_zh")
                        if key in sidecar_entries[task_index]
                    }
                    if task_index in sidecar_entries
                    else {}
                ),
            }
            for task_index, task_text in task_texts.items()
        ],
    }

    if missing:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder = init_t5_encoder(wan_dir, device=device, text_len=args.text_len)
        for task_index in missing:
            output_path = output_dir / f"task_{task_index:06d}.pt"
            embedding = encode_text(encoder, task_texts[task_index], device=device)
            temporary_path = output_dir / f".{output_path.name}.{os.getpid()}.tmp"
            torch.save(embedding, temporary_path)
            os.replace(temporary_path, output_path)
            print(f"cached task {task_index:06d}: {output_path}")
    else:
        print(f"all {len(task_texts)} task embeddings already exist under {output_dir}")

    manifest_path = output_dir / "manifest.json"
    temporary_manifest_path = output_dir / f".manifest.json.{os.getpid()}.tmp"
    with temporary_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    os.replace(temporary_manifest_path, manifest_path)
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
