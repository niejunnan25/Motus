#!/usr/bin/env python3
"""Cache WAN UMT5 embeddings for LeRobot v3 task metadata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.wan_path:
        raise ValueError("--wan_path is required unless WAN_PATH or WAN_ROOT is set")

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "umt5_wan_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_texts = load_task_texts(dataset_root)
    missing = [
        task_index
        for task_index in task_texts
        if args.overwrite or not (output_dir / f"task_{task_index:06d}.pt").exists()
    ]

    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "text_len": args.text_len,
        "tasks": [
            {
                "task_index": task_index,
                "task_text": task_text,
                "embedding_path": str(output_dir / f"task_{task_index:06d}.pt"),
            }
            for task_index, task_text in task_texts.items()
        ],
    }

    if missing:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        wan_dir = resolve_wan_dir(args.wan_path)
        encoder = init_t5_encoder(wan_dir, device=device, text_len=args.text_len)
        for task_index in missing:
            output_path = output_dir / f"task_{task_index:06d}.pt"
            embedding = encode_text(encoder, task_texts[task_index], device=device)
            torch.save(embedding, output_path)
            print(f"cached task {task_index:06d}: {output_path}")
    else:
        print(f"all {len(task_texts)} task embeddings already exist under {output_dir}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
