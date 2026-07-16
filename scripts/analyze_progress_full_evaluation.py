#!/usr/bin/env python3
"""Build a unified A0-A4/Layerwise/B0-B5 Progress evaluation report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_progress_stage2_evals import (
    COLORS,
    RunData,
    load_run,
    paired_statistics,
    parse_run,
    render_comparison_case,
    render_individual_case,
    validate_runs,
    write_csv,
)

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "A0": {"family": "single-frame serial", "epochs": 60},
    "A1": {"family": "single-frame patch1", "epochs": 60},
    "A2": {"family": "single-frame split views", "epochs": 60},
    "A3": {"family": "single-frame token match", "epochs": 60},
    "A4": {"family": "single-frame combined", "epochs": 20},
    "Layerwise": {"family": "single-frame layerwise WVM", "epochs": 20},
    "B0": {"family": "single-frame A2 control", "epochs": 60},
    "B1": {"family": "two-frame fused", "epochs": 60},
    "B2": {"family": "two-frame joint", "epochs": 60},
    "B3": {"family": "joint + absolute", "epochs": 60},
    "B4": {"family": "joint + delta", "epochs": 60},
    "B5": {"family": "joint + absolute + delta", "epochs": 60},
}

GOLDEN_CASES = {
    13: "branch switch",
    21: "long plateau",
    37: "branch switch",
    45: "VOC blind-spot collapse",
    50: "VOC protocol artifact",
    62: "endpoint collapse",
    66: "multi-stage aliasing",
    91: "multi-object aliasing",
    26: "positive control",
    43: "positive control",
    83: "positive control",
    87: "positive control",
}

CORE_METRICS = [
    ("mae", "Task-macro MAE", True),
    ("rmse", "Task-macro RMSE", True),
    ("spearman", "Task-macro Spearman", False),
    ("ordering_accuracy", "Ordering accuracy", False),
    ("end_error", "Endpoint error", True),
]

STABILITY_METRICS = [
    ("monotonic_violation_rate", "Monotonic violation", True),
    ("backward_step_rate", "Backward-step rate", True),
    ("predicted_jump_p95", "Progress jump P95", True),
    ("predicted_jump_p99", "Progress jump P99", True),
    ("max_predicted_jump", "Maximum progress jump", True),
    ("sequential_delta_mae", "Sequential delta MAE", True),
]

PAIR_METRICS = [
    ("balanced_pair_current_mae", "Balanced current MAE", True),
    ("balanced_pair_previous_mae", "Balanced previous MAE", True),
    ("balanced_pair_delta_mae", "Balanced delta MAE", True),
    ("balanced_pair_direction_accuracy", "Direction accuracy", False),
    ("balanced_pair_wrong_direction_rate", "Wrong-direction rate", True),
    ("previous_shuffle_progress_change", "Previous-shuffle progress change", False),
    ("previous_shuffle_alignment_tv", "Previous-shuffle alignment TV", False),
    ("previous_shuffle_effect_rate_001", "Shuffle effect rate > 0.01", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--baseline", default="A2")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--benchmark_json", type=Path, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def episode_number(name: str) -> int:
    match = re.search(r"episode_(\d+)$", name)
    if match is None:
        raise ValueError(f"Cannot parse episode number from {name!r}")
    return int(match.group(1))


def event_metrics(run: RunData, episode_name: str) -> Dict[str, Any]:
    episode = run.episodes[episode_name]
    prediction = episode["prediction"].numpy()
    target = episode["target"].numpy()
    alignment = episode["alignment_probabilities"].numpy()
    errors = np.abs(prediction - target)
    progress_jumps = np.diff(prediction)
    slots = alignment.argmax(axis=1)
    slot_jumps = np.diff(slots)
    max_progress_jump = (
        float(np.abs(progress_jumps).max()) if progress_jumps.size else 0.0
    )
    max_slot_jump = int(np.abs(slot_jumps).max()) if slot_jumps.size else 0
    return {
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.square(prediction - target).mean())),
        "max_absolute_error": float(errors.max()),
        "max_progress_jump": max_progress_jump,
        "max_backward_jump": (
            float(np.maximum(-progress_jumps, 0.0).max())
            if progress_jumps.size
            else 0.0
        ),
        "max_slot_jump": max_slot_jump,
        "backward_step_rate": (
            float((progress_jumps < -1e-4).mean()) if progress_jumps.size else 0.0
        ),
        "end_error": float(errors[-1]),
        "catastrophic": bool(
            errors.max() >= 0.25 or max_progress_jump >= 0.20 or max_slot_jump >= 20
        ),
    }


def metric_value(run: RunData, metric: str) -> float:
    value = run.metrics["task_macro"].get(metric)
    return float(value) if value is not None else math.nan


def summary_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "model": run.name,
            **MODEL_METADATA.get(run.name, {"family": "unknown", "epochs": ""}),
            "episodes": int(run.metrics["episodes"]),
            "queries": int(run.metrics["queries"]),
            "tasks": int(run.metrics["tasks"]),
        }
        for metric, _, _ in CORE_METRICS + STABILITY_METRICS + PAIR_METRICS:
            row[f"task_macro_{metric}"] = metric_value(run, metric)
        events = [event_metrics(run, name) for name in run.episodes]
        row["catastrophic_episodes"] = sum(
            bool(item["catastrophic"]) for item in events
        )
        row["catastrophic_episode_rate"] = row["catastrophic_episodes"] / len(events)
        row["episode_mean_max_absolute_error"] = float(
            np.mean([item["max_absolute_error"] for item in events])
        )
        row["episode_mean_max_slot_jump"] = float(
            np.mean([item["max_slot_jump"] for item in events])
        )
        rows.append(row)
    return rows


def golden_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    episode_lookup = {episode_number(name): name for name in runs[0].episodes}
    rows: List[Dict[str, Any]] = []
    for number, category in GOLDEN_CASES.items():
        episode_name = episode_lookup.get(number)
        if episode_name is None:
            continue
        for run in runs:
            rows.append(
                {
                    "episode": number,
                    "episode_name": episode_name,
                    "category": category,
                    "model": run.name,
                    **event_metrics(run, episode_name),
                }
            )
    return rows


def plot_metric_grid(
    runs: Sequence[RunData],
    metrics: Sequence[tuple[str, str, bool]],
    output_path: Path,
    *,
    title: str,
) -> None:
    columns = 3
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 5.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    colors = COLORS[: len(runs)]
    x = np.arange(len(runs))
    for axis, (metric, label, lower) in zip(axes.flat, metrics):
        values = [metric_value(run, metric) for run in runs]
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, [run.name for run in runs], rotation=35, ha="right")
        axis.set_title(f"{label} ({'lower' if lower else 'higher'} is better)")
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=18)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_catastrophic(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    names = [str(row["model"]) for row in rows]
    values = [int(row["catastrophic_episodes"]) for row in rows]
    figure, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    bars = axis.bar(np.arange(len(rows)), values, color=COLORS[: len(rows)])
    axis.set_xticks(np.arange(len(rows)), names, rotation=35, ha="right")
    axis.set_ylabel("Episodes triggering a catastrophic threshold")
    axis.set_title(
        "Catastrophic cases: error >= .25, progress jump >= .20, or slot jump >= 20"
    )
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_golden_heatmap(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    baseline: str,
    output_path: Path,
) -> None:
    episodes = sorted({int(row["episode"]) for row in rows})
    lookup = {(int(row["episode"]), str(row["model"])): row for row in rows}
    matrix = np.zeros((len(episodes), len(models)), dtype=np.float64)
    for row_index, episode in enumerate(episodes):
        base = float(lookup[(episode, baseline)]["mae"])
        for column, model in enumerate(models):
            matrix[row_index, column] = float(lookup[(episode, model)]["mae"]) - base
    limit = max(float(np.abs(matrix).max()), 1e-4)
    figure, axis = plt.subplots(
        figsize=(max(13, len(models) * 1.05), max(7, len(episodes) * 0.6)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(models)), models, rotation=35, ha="right")
    axis.set_yticks(
        np.arange(len(episodes)),
        [f"{episode:03d} {GOLDEN_CASES[episode]}" for episode in episodes],
    )
    axis.set_title(f"Golden-case MAE delta versus {baseline} (blue is better)")
    for row_index in range(len(episodes)):
        for column in range(len(models)):
            axis.text(
                column,
                row_index,
                f"{matrix[row_index, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(matrix[row_index, column]) > 0.55 * limit
                    else "black"
                ),
            )
    figure.colorbar(image, ax=axis, label="Episode MAE delta")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_golden_cases(runs: Sequence[RunData], output_dir: Path) -> None:
    lookup = {episode_number(name): name for name in runs[0].episodes}
    groups = {
        "A_variants": [
            run for run in runs if run.name.startswith("A") or run.name == "Layerwise"
        ],
        "B_variants": [run for run in runs if run.name.startswith("B")],
    }
    for episode, category in GOLDEN_CASES.items():
        episode_name = lookup.get(episode)
        if episode_name is None:
            continue
        row = {
            "episode_name": episode_name,
            "task_index": runs[0].episodes[episode_name]["task_index"],
            "categories": category,
        }
        case_dir = output_dir / f"episode_{episode:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for group_name, group_runs in groups.items():
            if group_runs:
                render_comparison_case(group_runs, row, case_dir / f"{group_name}.png")
        for index, run in enumerate(runs):
            render_individual_case(
                run,
                row,
                COLORS[index],
                case_dir / "by_model" / f"{run.name}.png",
            )


def manual_review_rows(
    runs: Sequence[RunData], benchmark_json: Path | None
) -> List[Dict[str, Any]]:
    tasks: Dict[int, str] = {}
    if benchmark_json is not None:
        payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
        tasks = {
            index: str(task)
            for index, task in enumerate(payload.get("sample_path_task", []))
        }
    rows: List[Dict[str, Any]] = []
    for episode_name in sorted(runs[0].episodes, key=episode_number):
        number = episode_number(episode_name)
        for run in runs:
            rows.append(
                {
                    "episode": number,
                    "episode_name": episode_name,
                    "task": tasks.get(number, ""),
                    "model": run.name,
                    "phase_accuracy_0_2": "",
                    "temporal_continuity_0_2": "",
                    "interaction_alignment_0_2": "",
                    "endpoint_stability_0_2": "",
                    "win_tie_loss_vs_baseline": "",
                    "failure_tags": "",
                    "timestamp_notes": "",
                }
            )
    return rows


def report_text(
    dataset_name: str,
    runs: Sequence[RunData],
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    paired: Sequence[Mapping[str, Any]],
) -> str:
    ranked = sorted(rows, key=lambda row: float(row["task_macro_mae"]))
    lines = [
        f"# Unified Progress Evaluation: {dataset_name}",
        "",
        f"Models: {len(runs)}; episodes: {ranked[0]['episodes']}; queries: {ranked[0]['queries']}; tasks: {ranked[0]['tasks']}.",
        f"All runs passed fixed-window validation against baseline `{baseline}`.",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Model | Epoch | MAE | RMSE | Spearman | Ordering | Backward | P99 jump | Catastrophic episodes | End error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['epochs']} | "
            f"{float(row['task_macro_mae']):.6f} | {float(row['task_macro_rmse']):.6f} | "
            f"{float(row['task_macro_spearman']):.6f} | "
            f"{float(row['task_macro_ordering_accuracy']):.6f} | "
            f"{float(row['task_macro_backward_step_rate']):.6f} | "
            f"{float(row['task_macro_predicted_jump_p99']):.6f} | "
            f"{int(row['catastrophic_episodes'])} | "
            f"{float(row['task_macro_end_error']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"## Paired MAE versus {baseline}",
            "",
            "| Model | Level | Delta | 95% CI | Win rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row["metric"] != "mae":
            continue
        lines.append(
            f"| {row['model']} | {row['level']} | "
            f"{float(row['mean_delta_model_minus_baseline']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{float(row['win_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Raw training loss is not compared across single and joint objectives. This report uses common Progress outputs, sequence stability, pair diagnostics, and fixed-case regression. Visual evidence and downstream RL evaluation remain separate gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [load_run(*parse_run(spec)) for spec in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verification = validate_runs(runs, args.baseline)
    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    (args.output_dir / "windows_match.txt").write_text(
        "same_fixed_windows=true\n"
        f"episodes={verification['episodes']}\n"
        f"baseline={args.baseline}\n",
        encoding="utf-8",
    )

    rows = summary_rows(runs)
    write_csv(args.output_dir / "full_metrics_summary.csv", rows)
    (args.output_dir / "full_metrics_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    paired = paired_statistics(
        runs,
        args.baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(args.output_dir / "paired_statistics.csv", paired)

    golden = golden_rows(runs)
    write_csv(args.output_dir / "golden_case_metrics.csv", golden)
    manual = manual_review_rows(runs, args.benchmark_json)
    write_csv(args.output_dir / "manual_review_template.csv", manual)

    plot_metric_grid(
        runs,
        CORE_METRICS,
        args.output_dir / "core_metrics.png",
        title="Progress absolute-position metrics",
    )
    plot_metric_grid(
        runs,
        STABILITY_METRICS,
        args.output_dir / "stability_metrics.png",
        title="Progress temporal stability metrics",
    )
    pair_runs = [run for run in runs if run.name.startswith("B") and run.name != "B0"]
    if pair_runs:
        plot_metric_grid(
            pair_runs,
            PAIR_METRICS,
            args.output_dir / "pair_diagnostics.png",
            title="Two-frame pair diagnostics",
        )
    plot_catastrophic(rows, args.output_dir / "catastrophic_episodes.png")
    if golden:
        plot_golden_heatmap(
            golden,
            [run.name for run in runs],
            args.baseline,
            args.output_dir / "golden_case_mae_delta.png",
        )
        render_golden_cases(runs, args.output_dir / "golden_cases")
    (args.output_dir / "REPORT_AUTO.md").write_text(
        report_text(args.output_dir.name, runs, rows, args.baseline, paired),
        encoding="utf-8",
    )
    print(f"Unified Progress analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
