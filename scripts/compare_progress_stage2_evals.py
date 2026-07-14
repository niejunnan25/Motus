#!/usr/bin/env python3
"""Compare Progress Stage-2 evaluations on identical cached episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOWER_IS_BETTER = {
    "mae": True,
    "rmse": True,
    "voc_pearson": False,
    "ordering_accuracy": False,
    "monotonic_violation_rate": True,
    "end_error": True,
}

PLOT_METRICS = [
    ("mae", "MAE", True),
    ("rmse", "RMSE", True),
    ("voc_pearson", "Pearson correlation", False),
    ("ordering_accuracy", "Ordering accuracy", False),
    ("monotonic_violation_rate", "Monotonic violation", True),
    ("end_error", "End error", True),
]

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


@dataclass
class RunData:
    name: str
    root: Path
    metrics: Dict[str, Any]
    episode_metrics: Dict[str, Dict[str, Any]]
    episodes: Dict[str, Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=EVAL_DIR",
        help="Evaluation name and directory. Repeat once per model.",
    )
    parser.add_argument("--baseline", default="A0")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--cases_per_category", type=int, default=5)
    parser.add_argument("--max_cases", type=int, default=48)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=EVAL_DIR, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing run name in {value!r}")
    return name, Path(path).expanduser().resolve()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return value.item()
    return value


def load_run(name: str, root: Path) -> RunData:
    required = ["metrics.json", "episode_metrics.csv", "episode_outputs.pt"]
    missing = [filename for filename in required if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} is missing {missing} under {root}")

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    with (root / "episode_metrics.csv").open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    episode_metrics: Dict[str, Dict[str, Any]] = {}
    for raw_row in csv_rows:
        episode_name = str(raw_row["episode_name"])
        if episode_name in episode_metrics:
            raise ValueError(f"Duplicate episode {episode_name!r} in {root}")
        row: Dict[str, Any] = {
            "episode_name": episode_name,
            "task_index": int(raw_row["task_index"]),
            "num_queries": int(raw_row["num_queries"]),
        }
        row.update(
            {
                key: float(value)
                for key, value in raw_row.items()
                if key not in {"episode_name", "task_index", "num_queries"}
            }
        )
        episode_metrics[episode_name] = row

    payload = torch_load(root / "episode_outputs.pt")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported episode output schema in {root}")
    episodes: Dict[str, Dict[str, Any]] = {}
    for raw_episode in payload["episodes"]:
        episode_name = str(raw_episode["episode_name"])
        if episode_name in episodes:
            raise ValueError(f"Duplicate episode output {episode_name!r} in {root}")
        episodes[episode_name] = {
            "episode_name": episode_name,
            "task_index": int(scalar(raw_episode["task_index"])),
            "total_frames": int(scalar(raw_episode["total_frames"])),
            "frame_indices": torch.as_tensor(raw_episode["frame_indices"]).long(),
            "target": torch.as_tensor(raw_episode["target"]).float(),
            "prediction": torch.as_tensor(raw_episode["prediction"]).float(),
            "alignment_probabilities": torch.as_tensor(
                raw_episode["alignment_probabilities"]
            ).float(),
        }
    if set(episode_metrics) != set(episodes):
        raise ValueError(f"CSV and tensor episode sets differ for {name}")
    return RunData(name, root, metrics, episode_metrics, episodes)


def validate_runs(runs: Sequence[RunData], baseline: str) -> Dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required")
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"Run names must be unique: {names}")
    if baseline not in names:
        raise ValueError(f"Baseline {baseline!r} is not one of {names}")

    reference = runs[names.index(baseline)]
    reference_names = list(reference.episodes)
    verification: Dict[str, Any] = {
        "same_fixed_windows": True,
        "baseline": baseline,
        "episodes": len(reference_names),
        "runs": {},
    }
    for run in runs:
        if set(run.episodes) != set(reference_names):
            missing = sorted(set(reference_names) - set(run.episodes))
            extra = sorted(set(run.episodes) - set(reference_names))
            raise ValueError(
                f"Episode set mismatch for {run.name}: missing={missing[:5]} "
                f"extra={extra[:5]}"
            )
        max_target_delta = 0.0
        max_probability_sum_error = 0.0
        out_of_range_predictions = 0
        query_count = 0
        for episode_name in reference_names:
            expected = reference.episodes[episode_name]
            actual = run.episodes[episode_name]
            for key in ("task_index", "total_frames"):
                if actual[key] != expected[key]:
                    raise ValueError(
                        f"{run.name}/{episode_name} differs in {key}: "
                        f"{actual[key]} != {expected[key]}"
                    )
            if not torch.equal(actual["frame_indices"], expected["frame_indices"]):
                raise ValueError(f"Frame indices differ for {run.name}/{episode_name}")
            if actual["target"].shape != expected["target"].shape:
                raise ValueError(f"Target shape differs for {run.name}/{episode_name}")
            target_delta = float((actual["target"] - expected["target"]).abs().max())
            max_target_delta = max(max_target_delta, target_delta)
            if target_delta > 1e-6:
                raise ValueError(f"Targets differ for {run.name}/{episode_name}")
            prediction = actual["prediction"]
            alignment = actual["alignment_probabilities"]
            if prediction.ndim != 1 or alignment.shape != (prediction.numel(), 53):
                raise ValueError(
                    f"Unexpected output shape for {run.name}/{episode_name}: "
                    f"prediction={tuple(prediction.shape)} alignment={tuple(alignment.shape)}"
                )
            if not torch.isfinite(prediction).all() or not torch.isfinite(alignment).all():
                raise ValueError(f"Non-finite output for {run.name}/{episode_name}")
            probability_error = float((alignment.sum(dim=-1) - 1.0).abs().max())
            max_probability_sum_error = max(
                max_probability_sum_error, probability_error
            )
            if probability_error > 1e-3:
                raise ValueError(
                    f"Alignment probabilities do not sum to one for "
                    f"{run.name}/{episode_name}: {probability_error}"
                )
            out_of_range_predictions += int(
                ((prediction < 0.0) | (prediction > 1.0)).sum()
            )
            query_count += prediction.numel()
        verification["runs"][run.name] = {
            "episodes": len(run.episodes),
            "queries": query_count,
            "max_target_delta": max_target_delta,
            "max_probability_sum_error": max_probability_sum_error,
            "out_of_range_predictions": out_of_range_predictions,
        }
    return verification


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "model": run.name,
            "episodes": run.metrics["episodes"],
            "queries": run.metrics["queries"],
            "tasks": len(run.metrics["per_task"]),
            "query_weighted_mae": run.metrics["query_weighted"]["mae"],
            "query_weighted_rmse": run.metrics["query_weighted"]["rmse"],
        }
        for scope in ("episode_mean", "task_macro"):
            for key, value in run.metrics[scope].items():
                row[f"{scope}_{key}"] = value
        rows.append(row)
    return rows


def per_episode_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    episode_names = sorted(runs[0].episodes)
    rows: List[Dict[str, Any]] = []
    for episode_name in episode_names:
        reference = runs[0].episodes[episode_name]
        row: Dict[str, Any] = {
            "episode_name": episode_name,
            "task_index": reference["task_index"],
            "num_queries": reference["target"].numel(),
        }
        for run in runs:
            for metric in LOWER_IS_BETTER:
                row[f"{run.name}_{metric}"] = run.episode_metrics[episode_name][
                    metric
                ]
        rows.append(row)
    return rows


def per_task_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    task_indices = sorted(
        {int(row["task_index"]) for row in runs[0].episode_metrics.values()}
    )
    rows: List[Dict[str, Any]] = []
    for task_index in task_indices:
        row: Dict[str, Any] = {"task_index": task_index}
        for run in runs:
            task_rows = [
                metrics
                for metrics in run.episode_metrics.values()
                if int(metrics["task_index"]) == task_index
            ]
            row[f"{run.name}_episodes"] = len(task_rows)
            for metric in LOWER_IS_BETTER:
                row[f"{run.name}_{metric}"] = float(
                    np.mean([metrics[metric] for metrics in task_rows])
                )
        rows.append(row)
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("Bootstrap requires a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 1000
    for start in range(0, samples, batch):
        end = min(start + batch, samples)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def metric_vector(run: RunData, episode_names: Sequence[str], metric: str) -> np.ndarray:
    return np.asarray(
        [float(run.episode_metrics[name][metric]) for name in episode_names],
        dtype=np.float64,
    )


def task_metric_vector(
    run: RunData, task_indices: Sequence[int], metric: str
) -> np.ndarray:
    values: List[float] = []
    for task_index in task_indices:
        task_values = [
            float(row[metric])
            for row in run.episode_metrics.values()
            if int(row["task_index"]) == task_index
        ]
        values.append(float(np.mean(task_values)))
    return np.asarray(values, dtype=np.float64)


def paired_statistics(
    runs: Sequence[RunData],
    baseline_name: str,
    samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    run_by_name = {run.name: run for run in runs}
    baseline = run_by_name[baseline_name]
    episode_names = sorted(baseline.episodes)
    task_indices = sorted(
        {int(row["task_index"]) for row in baseline.episode_metrics.values()}
    )
    rows: List[Dict[str, Any]] = []
    counter = 0
    for run in runs:
        if run.name == baseline_name:
            continue
        for metric, lower_is_better in LOWER_IS_BETTER.items():
            for level in ("episode", "task"):
                if level == "episode":
                    baseline_values = metric_vector(baseline, episode_names, metric)
                    run_values = metric_vector(run, episode_names, metric)
                else:
                    baseline_values = task_metric_vector(
                        baseline, task_indices, metric
                    )
                    run_values = task_metric_vector(run, task_indices, metric)
                delta = run_values - baseline_values
                mean_delta, ci_low, ci_high = bootstrap_mean_ci(
                    delta, samples=samples, seed=seed + counter
                )
                counter += 1
                wins = delta < 0.0 if lower_is_better else delta > 0.0
                rows.append(
                    {
                        "model": run.name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "level": level,
                        "n": delta.size,
                        "baseline_mean": float(baseline_values.mean()),
                        "model_mean": float(run_values.mean()),
                        "mean_delta_model_minus_baseline": mean_delta,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "lower_is_better": lower_is_better,
                        "win_rate": float(wins.mean()),
                        "tie_rate": float((delta == 0.0).mean()),
                        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                    }
                )
    return rows


def plot_overall_metrics(runs: Sequence[RunData], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    x = np.arange(len(runs))
    for axis, (metric, title, lower_is_better) in zip(axes.flat, PLOT_METRICS):
        values = [float(run.metrics["task_macro"][metric]) for run in runs]
        bars = axis.bar(x, values, color=COLORS[: len(runs)], width=0.68)
        axis.set_xticks(x, [run.name for run in runs], fontsize=11)
        axis.set_title(f"{title} ({'lower' if lower_is_better else 'higher'} is better)")
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    figure.suptitle("Progress Stage-2: task-macro offline metrics", fontsize=18)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def per_task_mae(runs: Sequence[RunData]) -> tuple[List[int], np.ndarray]:
    tasks = sorted({int(task) for run in runs for task in run.metrics["per_task"]})
    matrix = np.full((len(tasks), len(runs)), np.nan, dtype=np.float64)
    for column, run in enumerate(runs):
        for row, task in enumerate(tasks):
            task_metrics = run.metrics["per_task"].get(str(task))
            if task_metrics is not None:
                matrix[row, column] = float(task_metrics["mae"])
    return tasks, matrix


def plot_per_task_heatmap(
    runs: Sequence[RunData], baseline_name: str, output_path: Path
) -> None:
    tasks, matrix = per_task_mae(runs)
    baseline_index = [run.name for run in runs].index(baseline_name)
    delta = matrix - matrix[:, [baseline_index]]
    height = max(12.0, 0.34 * len(tasks))
    figure, axes = plt.subplots(1, 2, figsize=(15, height), constrained_layout=True)
    absolute = axes[0].imshow(matrix, aspect="auto", cmap="magma_r")
    limit = float(np.nanmax(np.abs(delta)))
    if not math.isfinite(limit) or limit == 0.0:
        limit = 1e-6
    relative = axes[1].imshow(
        delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    for axis in axes:
        axis.set_xticks(np.arange(len(runs)), [run.name for run in runs])
        axis.set_yticks(np.arange(len(tasks)), [str(task) for task in tasks])
        axis.set_ylabel("Task index")
    axes[0].set_title("Per-task MAE")
    axes[1].set_title(f"Per-task MAE delta vs {baseline_name} (blue is better)")
    figure.colorbar(absolute, ax=axes[0], label="MAE")
    figure.colorbar(relative, ax=axes[1], label="MAE delta")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_paired_episode_delta(
    runs: Sequence[RunData], baseline_name: str, output_path: Path
) -> None:
    baseline = next(run for run in runs if run.name == baseline_name)
    names = sorted(baseline.episodes)
    variants = [run for run in runs if run.name != baseline_name]
    deltas = [
        metric_vector(run, names, "mae") - metric_vector(baseline, names, "mae")
        for run in variants
    ]
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    parts = axis.violinplot(deltas, showmeans=True, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], COLORS[1:]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    axis.axhline(0.0, color="black", linewidth=1.5, linestyle="--")
    axis.set_xticks(np.arange(1, len(variants) + 1), [run.name for run in variants])
    axis.set_ylabel(f"Episode MAE delta vs {baseline_name}")
    axis.set_title("Paired episode comparison (negative is better)")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def progress_bin_errors(
    runs: Sequence[RunData], bins: int = 10
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    centers = (np.arange(bins) + 0.5) / bins
    values: Dict[str, np.ndarray] = {}
    for run in runs:
        sums = np.zeros(bins, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
        for episode in run.episodes.values():
            target = episode["target"].numpy()
            error = np.abs(episode["prediction"].numpy() - target)
            indices = np.minimum((target * bins).astype(np.int64), bins - 1)
            for index in range(bins):
                selected = indices == index
                sums[index] += error[selected].sum()
                counts[index] += int(selected.sum())
        values[run.name] = sums / np.maximum(counts, 1)
    return centers, values


def progress_bin_rows(runs: Sequence[RunData], bins: int = 10) -> List[Dict[str, Any]]:
    centers, values = progress_bin_errors(runs, bins=bins)
    rows: List[Dict[str, Any]] = []
    for index, center in enumerate(centers):
        row: Dict[str, Any] = {
            "bin": index,
            "progress_low": index / bins,
            "progress_high": (index + 1) / bins,
            "progress_center": center,
        }
        for run in runs:
            row[f"{run.name}_mae"] = float(values[run.name][index])
        rows.append(row)
    return rows


def backward_jump_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    thresholds = (1e-4, 1e-3, 1e-2, 5e-2)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        jumps = torch.cat(
            [
                episode["prediction"][:-1] - episode["prediction"][1:]
                for episode in run.episodes.values()
                if episode["prediction"].numel() > 1
            ]
        )
        row: Dict[str, Any] = {
            "model": run.name,
            "transitions": jumps.numel(),
            "mean_positive_backward_jump": float(jumps.clamp_min(0.0).mean()),
            "maximum_backward_jump": float(jumps.max()),
        }
        for threshold in thresholds:
            key = f"rate_gt_{threshold:g}".replace("-", "m").replace(".", "p")
            row[key] = float((jumps > threshold).float().mean())
        rows.append(row)
    return rows


def plot_backward_jump_rates(
    runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    thresholds = (1e-3, 1e-2, 5e-2)
    x = np.arange(len(thresholds), dtype=np.float64)
    width = 0.18
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for index, (run, color, row) in enumerate(zip(runs, COLORS, rows)):
        values = []
        for threshold in thresholds:
            key = f"rate_gt_{threshold:g}".replace("-", "m").replace(".", "p")
            values.append(float(row[key]))
        axis.bar(
            x + (index - (len(runs) - 1) / 2) * width,
            values,
            width=width,
            label=run.name,
            color=color,
        )
    axis.set_xticks(x, [f"> {threshold:g}" for threshold in thresholds])
    axis.set_xlabel("Single-step progress decrease threshold")
    axis.set_ylabel("Fraction of adjacent transitions")
    axis.set_title("Backward progress jumps: jitter versus severe collapse")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=min(4, len(runs)))
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_error_by_progress_bin(runs: Sequence[RunData], output_path: Path) -> None:
    centers, values = progress_bin_errors(runs)
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for run, color in zip(runs, COLORS):
        axis.plot(
            centers,
            values[run.name],
            marker="o",
            linewidth=2.2,
            label=run.name,
            color=color,
        )
    axis.set_xlabel("Ground-truth progress")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("Error over task progress")
    axis.set_xticks(np.linspace(0.0, 1.0, 11))
    axis.grid(alpha=0.25)
    axis.legend(ncol=min(4, len(runs)))
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_episode_scatter(
    runs: Sequence[RunData], baseline_name: str, output_path: Path
) -> None:
    baseline = next(run for run in runs if run.name == baseline_name)
    variants = [run for run in runs if run.name != baseline_name]
    names = sorted(baseline.episodes)
    figure, axes = plt.subplots(
        1, len(variants), figsize=(6 * len(variants), 5.5), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    baseline_values = metric_vector(baseline, names, "mae")
    for axis, run, color in zip(axes, variants, COLORS[1:]):
        run_values = metric_vector(run, names, "mae")
        limit = max(float(baseline_values.max()), float(run_values.max())) * 1.04
        axis.scatter(baseline_values, run_values, s=24, alpha=0.65, color=color)
        axis.plot([0.0, limit], [0.0, limit], color="black", linestyle="--")
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_xlabel(f"{baseline_name} episode MAE")
        axis.set_ylabel(f"{run.name} episode MAE")
        axis.set_title(f"{run.name} vs {baseline_name}")
        axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def ranked_names(scores: Mapping[str, float], descending: bool) -> List[str]:
    return sorted(scores, key=lambda name: scores[name], reverse=descending)


def select_cases(
    runs: Sequence[RunData],
    baseline_name: str,
    per_category: int,
    max_cases: int,
) -> List[Dict[str, Any]]:
    baseline = next(run for run in runs if run.name == baseline_name)
    episode_names = sorted(baseline.episodes)
    categories: Dict[str, List[str]] = {name: [] for name in episode_names}
    selection_order: List[str] = []

    def add(category: str, ordered_names: Iterable[str]) -> None:
        for episode_name in list(ordered_names)[:per_category]:
            categories[episode_name].append(category)
            if episode_name not in selection_order:
                selection_order.append(episode_name)

    mean_mae = {
        name: float(np.mean([run.episode_metrics[name]["mae"] for run in runs]))
        for name in episode_names
    }
    add("common_hard", ranked_names(mean_mae, descending=True))

    for run in runs:
        if run.name == baseline_name:
            continue
        delta = {
            name: float(
                run.episode_metrics[name]["mae"]
                - baseline.episode_metrics[name]["mae"]
            )
            for name in episode_names
        }
        add(f"{run.name}_improves_vs_{baseline_name}", ranked_names(delta, False))
        add(f"{run.name}_regresses_vs_{baseline_name}", ranked_names(delta, True))

    endpoint = {
        name: max(float(run.episode_metrics[name]["end_error"]) for run in runs)
        for name in episode_names
    }
    monotonic = {
        name: max(
            float(run.episode_metrics[name]["monotonic_violation_rate"])
            for run in runs
        )
        for name in episode_names
    }
    disagreement = {
        name: float(
            np.stack([run.episodes[name]["prediction"].numpy() for run in runs])
            .std(axis=0)
            .mean()
        )
        for name in episode_names
    }
    max_frame_error = {
        name: max(
            float(
                (run.episodes[name]["prediction"] - run.episodes[name]["target"])
                .abs()
                .max()
            )
            for run in runs
        )
        for name in episode_names
    }
    add("endpoint_failure", ranked_names(endpoint, True))
    add("monotonic_failure", ranked_names(monotonic, True))
    add("model_disagreement", ranked_names(disagreement, True))
    add("largest_single_frame_error", ranked_names(max_frame_error, True))

    selected = selection_order[:max_cases]
    rows: List[Dict[str, Any]] = []
    for episode_name in selected:
        episode = baseline.episodes[episode_name]
        row: Dict[str, Any] = {
            "episode_name": episode_name,
            "task_index": episode["task_index"],
            "num_queries": episode["target"].numel(),
            "categories": ";".join(categories[episode_name]),
            "mean_model_mae": mean_mae[episode_name],
            "max_endpoint_error": endpoint[episode_name],
            "max_monotonic_violation": monotonic[episode_name],
            "mean_model_disagreement": disagreement[episode_name],
            "max_single_frame_error": max_frame_error[episode_name],
        }
        for run in runs:
            row[f"{run.name}_mae"] = run.episode_metrics[episode_name]["mae"]
        rows.append(row)
    return rows


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("._") or "episode"


def case_title(
    *, model: str | None, episode_name: str, task_index: int, categories: str
) -> str:
    prefix = f"{model} | " if model is not None else ""
    category_line = textwrap.fill(f"Categories: {categories}", width=115)
    return f"{prefix}{episode_name} | Task {task_index}\n{category_line}"


def alignment_vmax(runs: Sequence[RunData], episode_name: str) -> float:
    values = np.concatenate(
        [run.episodes[episode_name]["alignment_probabilities"].numpy().ravel() for run in runs]
    )
    return max(float(np.quantile(values, 0.995)), 0.05)


def draw_curve(axis: Any, run: RunData, episode_name: str, color: str) -> None:
    episode = run.episodes[episode_name]
    frames = episode["frame_indices"].numpy()
    target = episode["target"].numpy()
    prediction = episode["prediction"].numpy()
    axis.plot(frames, target, color="#111827", linewidth=2.4, label="GT")
    axis.plot(frames, prediction, color=color, linewidth=2.2, label=run.name)
    axis.fill_between(
        frames, target, prediction, color=color, alpha=0.12, linewidth=0
    )
    axis.set_ylim(-0.04, 1.04)
    axis.set_xlabel("Original episode frame")
    axis.set_ylabel("Progress")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left")
    metrics = run.episode_metrics[episode_name]
    axis.set_title(
        f"{run.name}: MAE={metrics['mae']:.4f}, Pearson={metrics['voc_pearson']:.3f}, "
        f"mono={metrics['monotonic_violation_rate']:.3f}, end={metrics['end_error']:.3f}"
    )


def draw_alignment(
    axis: Any, run: RunData, episode_name: str, vmax: float
) -> Any:
    episode = run.episodes[episode_name]
    frames = episode["frame_indices"].numpy()
    target = episode["target"].numpy()
    alignment = episode["alignment_probabilities"].numpy()
    image = axis.imshow(
        alignment,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        extent=(-0.5, alignment.shape[1] - 0.5, frames[0], frames[-1]),
    )
    axis.plot(
        target * (alignment.shape[1] - 1),
        frames,
        color="white",
        linewidth=1.5,
        linestyle="--",
        label="GT slot",
    )
    axis.set_xlabel("Generated trajectory slot")
    axis.set_ylabel("Original episode frame")
    axis.set_title(f"{run.name}: p(memory slot | current frame)")
    axis.legend(loc="upper left", fontsize=9)
    return image


def render_comparison_case(
    runs: Sequence[RunData], row: Mapping[str, Any], output_path: Path
) -> None:
    episode_name = str(row["episode_name"])
    vmax = alignment_vmax(runs, episode_name)
    figure, axes = plt.subplots(
        len(runs),
        2,
        figsize=(20, 5.0 * len(runs)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for index, (run, color) in enumerate(zip(runs, COLORS)):
        draw_curve(axes[index, 0], run, episode_name, color)
        image = draw_alignment(axes[index, 1], run, episode_name, vmax)
    episode = runs[0].episodes[episode_name]
    figure.suptitle(
        case_title(
            model=None,
            episode_name=episode_name,
            task_index=int(episode["task_index"]),
            categories=str(row["categories"]),
        ),
        fontsize=16,
    )
    if image is not None:
        figure.colorbar(image, ax=axes[:, 1], label="Alignment probability", shrink=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, facecolor="white")
    plt.close(figure)


def render_individual_case(
    run: RunData, row: Mapping[str, Any], color: str, output_path: Path
) -> None:
    episode_name = str(row["episode_name"])
    vmax = max(
        float(
            np.quantile(
                run.episodes[episode_name]["alignment_probabilities"].numpy(), 0.995
            )
        ),
        0.05,
    )
    figure, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    draw_curve(axes[0], run, episode_name, color)
    image = draw_alignment(axes[1], run, episode_name, vmax)
    episode = run.episodes[episode_name]
    figure.suptitle(
        case_title(
            model=run.name,
            episode_name=episode_name,
            task_index=int(episode["task_index"]),
            categories=str(row["categories"]),
        ),
        fontsize=15,
    )
    figure.colorbar(image, ax=axes[1], label="Alignment probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_cases(
    runs: Sequence[RunData], rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    comparison_dir = output_dir / "cases" / "comparison"
    managed_dirs = [comparison_dir] + [
        output_dir / "cases" / "by_model" / run.name for run in runs
    ]
    for directory in managed_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        for stale_plot in directory.glob("*.png"):
            stale_plot.unlink()
    for index, row in enumerate(rows):
        filename = f"{index:03d}_{safe_filename(str(row['episode_name']))}.png"
        render_comparison_case(runs, row, comparison_dir / filename)
        for run_index, (run, color) in enumerate(zip(runs, COLORS)):
            render_individual_case(
                run,
                row,
                color,
                output_dir / "cases" / "by_model" / run.name / filename,
            )


def report_text(
    runs: Sequence[RunData],
    baseline_name: str,
    summary: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> str:
    ranked = sorted(summary, key=lambda row: float(row["task_macro_mae"]))
    lines = [
        "# Progress Stage-2 Offline Evaluation",
        "",
        f"All {len(runs)} models were evaluated on the same {ranked[0]['episodes']} "
        f"episodes and {ranked[0]['queries']} single-frame queries.",
        "Primary ranking uses task-macro MAE; lower is better.",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Model | Task-macro MAE | RMSE | Pearson | Spearman | Ordering | Mono violation | End error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row['model']} | {float(row['task_macro_mae']):.6f} | "
            f"{float(row['task_macro_rmse']):.6f} | "
            f"{float(row['task_macro_voc_pearson']):.6f} | "
            f"{float(row['task_macro_spearman']):.6f} | "
            f"{float(row['task_macro_ordering_accuracy']):.6f} | "
            f"{float(row['task_macro_monotonic_violation_rate']):.6f} | "
            f"{float(row['task_macro_end_error']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"## Paired MAE Statistics vs {baseline_name}",
            "",
            "Delta is model minus baseline, so negative MAE delta is better.",
            "",
            "| Model | Level | Mean delta | 95% bootstrap CI | Win rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row["metric"] != "mae":
            continue
        lines.append(
            f"| {row['model']} | {row['level']} | "
            f"{float(row['mean_delta_model_minus_baseline']):.6f} | "
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] | "
            f"{float(row['win_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Visual Outputs",
            "",
            "- `overall_metrics.png`: task-macro metric comparison.",
            "- `per_task_mae_heatmap.png`: per-task MAE and delta against baseline.",
            "- `paired_episode_mae_delta.png`: paired episode-level MAE distributions.",
            "- `error_by_progress_bin.png`: error from task start to task end.",
            "- `backward_jump_rates.png`: minor jitter and severe backward-jump rates.",
            "- `episode_mae_scatter.png`: episode-level model-versus-baseline scatter.",
            f"- `cases/comparison/`: {len(selected)} multi-model failure and improvement cases.",
            "- `cases/by_model/<model>/`: the same selected cases at larger per-model scale.",
            "",
            "Case selection is metric-driven rather than hand-picked: common hard cases, "
            "improvements and regressions versus baseline, endpoint failures, monotonic "
            "failures, model disagreement, and largest single-frame errors.",
            "",
            "## Scope",
            "",
            "These results measure supervised progress estimation on the supplied evaluation "
            "episodes. They do not by themselves establish held-out generalization, downstream "
            "RL reward quality, or robustness to failed/out-of-distribution trajectories.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_specs = [parse_run(value) for value in args.run]
    runs = [load_run(name, path) for name, path in run_specs]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verification = validate_runs(runs, args.baseline)
    (args.output_dir / "windows_match.txt").write_text(
        "same_fixed_windows=true\n"
        f"episodes={verification['episodes']}\n"
        f"baseline={args.baseline}\n",
        encoding="utf-8",
    )
    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    summary = summary_rows(runs)
    write_csv(args.output_dir / "metrics_summary.csv", summary)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "per_episode_metrics.csv", per_episode_rows(runs))
    write_csv(args.output_dir / "per_task_metrics.csv", per_task_rows(runs))
    write_csv(
        args.output_dir / "error_by_progress_bin.csv", progress_bin_rows(runs)
    )
    backward_jumps = backward_jump_rows(runs)
    write_csv(args.output_dir / "backward_jump_rates.csv", backward_jumps)
    paired = paired_statistics(
        runs,
        args.baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(args.output_dir / "paired_statistics.csv", paired)
    (args.output_dir / "paired_statistics.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )

    selected = select_cases(
        runs,
        args.baseline,
        per_category=args.cases_per_category,
        max_cases=args.max_cases,
    )
    write_csv(args.output_dir / "case_selection.csv", selected)
    (args.output_dir / "case_selection.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    plot_overall_metrics(runs, args.output_dir / "overall_metrics.png")
    plot_per_task_heatmap(
        runs, args.baseline, args.output_dir / "per_task_mae_heatmap.png"
    )
    plot_paired_episode_delta(
        runs, args.baseline, args.output_dir / "paired_episode_mae_delta.png"
    )
    plot_error_by_progress_bin(
        runs, args.output_dir / "error_by_progress_bin.png"
    )
    plot_backward_jump_rates(
        runs, backward_jumps, args.output_dir / "backward_jump_rates.png"
    )
    plot_episode_scatter(
        runs, args.baseline, args.output_dir / "episode_mae_scatter.png"
    )
    render_cases(runs, selected, args.output_dir)
    (args.output_dir / "REPORT.md").write_text(
        report_text(runs, args.baseline, summary, paired, selected),
        encoding="utf-8",
    )
    print(f"Comparison complete: {args.output_dir}")


if __name__ == "__main__":
    main()
