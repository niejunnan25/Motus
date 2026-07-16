#!/usr/bin/env python3
"""Parse a VGM training terminal log and render a reproducible loss curve."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_PATTERN = re.compile(
    r"Step (?P<step>\d+)/(?P<total>\d+), Loss: (?P<loss>[0-9.eE+-]+) "
    r"\(Video: (?P<video>[0-9.eE+-]+), Middle: (?P<middle>[0-9.eE+-]+)\), "
    r"LR: (?P<lr>[0-9.eE+-]+), Time: (?P<seconds>[0-9.eE+-]+)s"
)


def parse_log(path: Path) -> List[Dict[str, float | int]]:
    by_step: Dict[int, Dict[str, float | int]] = {}
    text = path.read_text(errors="replace").replace("\r", "\n")
    for match in STEP_PATTERN.finditer(text):
        values = match.groupdict()
        step = int(values["step"])
        by_step[step] = {
            "step": step,
            "total_steps": int(values["total"]),
            "loss": float(values["loss"]),
            "video_loss": float(values["video"]),
            "middle_loss": float(values["middle"]),
            "learning_rate": float(values["lr"]),
            "step_seconds": float(values["seconds"]),
        }
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise ValueError(f"No VGM step records found in {path}")
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < window:
        return result
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_log(Path(args.log))
    steps = np.asarray([int(row["step"]) for row in rows])
    losses = np.asarray([float(row["loss"]) for row in rows])
    smoothed = rolling_mean(losses, args.window)

    csv_rows = []
    for row, mean in zip(rows, smoothed):
        csv_rows.append({**row, f"loss_mean_{args.window}": mean})
    with (output_dir / "training_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "records": len(rows),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        f"final_loss_mean_{args.window}": float(smoothed[-1]),
        "minimum_loss": float(losses.min()),
        "median_step_seconds_excluding_first": float(
            np.median([float(row["step_seconds"]) for row in rows[1:]])
        ),
    }
    with (output_dir / "training_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    axes[0].plot(steps, losses, color="#4C78A8", alpha=0.18, linewidth=0.7, label="Instantaneous loss")
    axes[0].plot(
        steps,
        smoothed,
        color="#D62728",
        linewidth=2.2,
        label=f"{args.window}-step mean",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Flow-matching loss (log scale)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    late = steps >= min(1000, int(steps[-1] * 0.2))
    axes[1].plot(steps[late], losses[late], color="#4C78A8", alpha=0.12, linewidth=0.7)
    axes[1].plot(steps[late], smoothed[late], color="#D62728", linewidth=2.2)
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Late-stage loss")
    axes[1].grid(alpha=0.25)
    for checkpoint in range(1000, int(steps[-1]) + 1, 1000):
        for axis in axes:
            axis.axvline(checkpoint, color="#777777", linewidth=0.8, alpha=0.4)
        index = int(np.searchsorted(steps, checkpoint))
        if index < len(steps) and np.isfinite(smoothed[index]):
            is_last = checkpoint == int(steps[-1])
            axes[1].annotate(
                f"{checkpoint}: {smoothed[index]:.4f}",
                (steps[index], smoothed[index]),
                xytext=((-6 if is_last else 4), 8),
                textcoords="offset points",
                fontsize=9,
                ha="right" if is_last else "left",
            )
    fig.suptitle("PushT V1-proper VGM training | 7 GPUs | global batch 14", fontsize=17)
    fig.savefig(output_dir / "training_loss_curve.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
