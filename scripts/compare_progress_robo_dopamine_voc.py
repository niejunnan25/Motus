#!/usr/bin/env python3
"""Directly compare Progress and Robo-Dopamine VOC+ on identical episodes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUR_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
    "#15803d",
    "#7e22ce",
    "#c2410c",
    "#0e7490",
]
OFFICIAL_COLOR = "#374151"


@dataclass(frozen=True)
class VocRun:
    name: str
    family: str
    values: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress_episode_csv",
        type=Path,
        required=True,
        help="progress_interval_per_episode.csv produced by the Progress evaluator.",
    )
    parser.add_argument(
        "--official",
        action="append",
        required=True,
        metavar="NAME=RESULT_JSON",
        help="Official forward result_interval_*_voc+_*.json. Repeat per model.",
    )
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--bootstrap_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError(f"Missing model name in {value!r}")
    return name.strip(), Path(raw_path).expanduser().resolve()


def normalize_episode_name(name: str) -> str:
    prefix = "robo_dopamine_bench/"
    return name[len(prefix) :] if name.startswith(prefix) else name


def load_progress_runs(path: Path, interval: int) -> List[VocRun]:
    grouped: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["interval"]) != interval:
                continue
            model = str(row["model"])
            episode = normalize_episode_name(str(row["episode_name"]))
            values = grouped.setdefault(model, {})
            if episode in values:
                raise ValueError(f"Duplicate Progress result for {model}/{episode}")
            values[episode] = float(row["spearman"])
    if not grouped:
        raise RuntimeError(f"No interval={interval} rows found in {path}")
    return [VocRun(name, "Progress", values) for name, values in grouped.items()]


def load_official_run(name: str, path: Path, interval: int) -> VocRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    if int(meta.get("interval", -1)) != interval:
        raise ValueError(
            f"Official interval mismatch for {name}: {meta.get('interval')} != {interval}"
        )
    if bool(meta.get("inverse", False)):
        raise ValueError(f"Expected forward VOC+ result for {name}, got inverse result")
    values: Dict[str, float] = {}
    for row in payload.get("results", []):
        episode = normalize_episode_name(str(row["name"]))
        if episode in values:
            raise ValueError(f"Duplicate official result for {name}/{episode}")
        values[episode] = float(row["voc"])
    if not values:
        raise RuntimeError(f"No per-episode official results found in {path}")
    expected_count = int(meta.get("num_voc", len(values)))
    if len(values) != expected_count:
        raise ValueError(
            f"Official result count mismatch for {name}: {len(values)} != {expected_count}"
        )
    expected_mean = float(meta.get("avg_voc", np.mean(list(values.values()))))
    actual_mean = float(np.mean(list(values.values())))
    if not np.isclose(actual_mean, expected_mean, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Official result mean mismatch for {name}: {actual_mean} != {expected_mean}"
        )
    return VocRun(name, "Robo-Dopamine", values)


def validate_episode_sets(runs: Sequence[VocRun]) -> List[str]:
    reference = set(runs[0].values)
    for run in runs[1:]:
        actual = set(run.values)
        if actual != reference:
            missing = sorted(reference - actual)
            extra = sorted(actual - reference)
            raise ValueError(
                f"Episode set differs for {run.name}: missing={missing[:5]}, extra={extra[:5]}"
            )
    return sorted(reference)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 10000
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_runs(
    runs: Sequence[VocRun],
    episodes: Sequence[str],
    *,
    interval: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in runs:
        values = np.asarray([run.values[name] for name in episodes], dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values, samples=bootstrap_samples, rng=rng)
        rows.append(
            {
                "interval": interval,
                "model": run.name,
                "family": run.family,
                "episodes": len(episodes),
                "mean_voc_plus": float(values.mean()),
                "std_voc_plus": float(values.std(ddof=1)),
                "standard_error": float(values.std(ddof=1) / np.sqrt(values.size)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def paired_rows(
    progress_runs: Sequence[VocRun],
    official_runs: Sequence[VocRun],
    episodes: Sequence[str],
    *,
    interval: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summaries: List[Dict[str, object]] = []
    per_episode: List[Dict[str, object]] = []
    tolerance = 1e-9
    for progress in progress_runs:
        for official in official_runs:
            progress_values = np.asarray(
                [progress.values[name] for name in episodes], dtype=np.float64
            )
            official_values = np.asarray(
                [official.values[name] for name in episodes], dtype=np.float64
            )
            delta = progress_values - official_values
            ci_low, ci_high = bootstrap_mean_ci(
                delta, samples=bootstrap_samples, rng=rng
            )
            win = delta > tolerance
            tie = np.abs(delta) <= tolerance
            loss = delta < -tolerance
            summaries.append(
                {
                    "interval": interval,
                    "progress_model": progress.name,
                    "official_model": official.name,
                    "episodes": len(episodes),
                    "progress_mean_voc_plus": float(progress_values.mean()),
                    "official_mean_voc_plus": float(official_values.mean()),
                    "mean_delta_progress_minus_official": float(delta.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": int(win.sum()),
                    "ties": int(tie.sum()),
                    "losses": int(loss.sum()),
                    "win_rate": float(win.mean()),
                    "tie_rate": float(tie.mean()),
                    "loss_rate": float(loss.mean()),
                }
            )
            for index, episode in enumerate(episodes):
                per_episode.append(
                    {
                        "interval": interval,
                        "progress_model": progress.name,
                        "official_model": official.name,
                        "episode_name": episode,
                        "progress_voc_plus": float(progress_values[index]),
                        "official_voc_plus": float(official_values[index]),
                        "delta_progress_minus_official": float(delta[index]),
                    }
                )
    return summaries, per_episode


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_means(
    rows: Sequence[Mapping[str, object]], output_path: Path, *, interval: int
) -> None:
    labels = [str(row["model"]) for row in rows]
    means = np.asarray([float(row["mean_voc_plus"]) for row in rows])
    low = np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows])
    colors = []
    progress_index = 0
    for row in rows:
        if row["family"] == "Progress":
            colors.append(OUR_COLORS[progress_index % len(OUR_COLORS)])
            progress_index += 1
        else:
            colors.append(OFFICIAL_COLOR)
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    bars = axis.bar(x, means, color=colors, alpha=0.9)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="none",
        ecolor="#111827",
        capsize=4,
        linewidth=1.5,
    )
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=45,
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(min(0.94, float(low.min()) - 0.005), 1.005)
    axis.set_ylabel("Episode-mean Spearman VOC+")
    axis.set_title(
        f"Direct comparison: identical LIBERO episodes and interval={interval} states"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=190, facecolor="white")
    plt.close(figure)


def plot_delta_heatmap(
    rows: Sequence[Mapping[str, object]],
    progress_names: Sequence[str],
    official_names: Sequence[str],
    output_path: Path,
    *,
    interval: int,
) -> None:
    lookup = {
        (str(row["progress_model"]), str(row["official_model"])): float(
            row["mean_delta_progress_minus_official"]
        )
        for row in rows
    }
    values = np.asarray(
        [
            [lookup[(progress, official)] for official in official_names]
            for progress in progress_names
        ]
    )
    limit = max(float(np.abs(values).max()), 1e-3)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        np.arange(len(official_names)), official_names, rotation=25, ha="right"
    )
    axis.set_yticks(np.arange(len(progress_names)), progress_names)
    axis.set_xlabel("Official Robo-Dopamine model")
    axis.set_ylabel("Progress model")
    axis.set_title(
        f"Paired VOC+ delta: Progress minus Robo-Dopamine (interval={interval})"
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:+.6f}",
                ha="center",
                va="center",
                color="white" if abs(values[row, column]) > 0.45 * limit else "#111827",
                fontsize=10,
            )
    figure.colorbar(image, ax=axis, label="Mean paired VOC+ delta")
    figure.savefig(output_path, dpi=190, facecolor="white")
    plt.close(figure)


def report_text(
    summary: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    *,
    interval: int,
) -> str:
    ranked = sorted(summary, key=lambda row: float(row["mean_voc_plus"]), reverse=True)
    lines = [
        "# Direct Robo-Dopamine VOC+ Comparison",
        "",
        f"All models use the same LIBERO episodes, the same official interval={interval} "
        "sampled after-states, and the same episode-mean Spearman VOC+ aggregation.",
        "",
        "## Model Means",
        "",
        "| Rank | Model | Family | Episodes | VOC+ | 95% bootstrap CI |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['family']} | {row['episodes']} | "
            f"{float(row['mean_voc_plus']):.6f} | "
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "Delta is Progress minus Robo-Dopamine; a negative value favors Robo-Dopamine.",
            "",
            "| Progress | Official | Mean delta | 95% bootstrap CI | Win / Tie / Loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['progress_model']} | {row['official_model']} | "
            f"{float(row['mean_delta_progress_minus_official']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{row['wins']} / {row['ties']} / {row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a direct comparison of the released forward VOC+ statistic. Model inputs "
            "remain different: Robo-Dopamine predicts relative hops from before/after multi-view "
            "pairs, whereas Progress predicts absolute position from one or two current observations "
            "and generated trajectory memory. The Progress model has no native reverse-hop output, "
            "so official VOC- is not synthesized.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.interval < 1:
        raise ValueError("interval must be positive")
    progress_runs = load_progress_runs(
        args.progress_episode_csv.resolve(), args.interval
    )
    official_runs = [
        load_official_run(name, path, args.interval)
        for name, path in (parse_named_path(value) for value in args.official)
    ]
    all_runs = [*progress_runs, *official_runs]
    episodes = validate_episode_sets(all_runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = summarize_runs(
        all_runs,
        episodes,
        interval=args.interval,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    paired, per_episode = paired_rows(
        progress_runs,
        official_runs,
        episodes,
        interval=args.interval,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    write_csv(args.output_dir / "direct_voc_summary.csv", summary)
    write_csv(args.output_dir / "paired_voc_comparison.csv", paired)
    write_csv(args.output_dir / "paired_voc_per_episode.csv", per_episode)
    (args.output_dir / "direct_voc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "paired_voc_comparison.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )
    plot_means(
        summary,
        args.output_dir / "direct_voc_means.png",
        interval=args.interval,
    )
    plot_delta_heatmap(
        paired,
        [run.name for run in progress_runs],
        [run.name for run in official_runs],
        args.output_dir / "paired_voc_delta_heatmap.png",
        interval=args.interval,
    )
    (args.output_dir / "REPORT.md").write_text(
        report_text(summary, paired, interval=args.interval), encoding="utf-8"
    )
    print(f"Direct VOC+ comparison complete: {args.output_dir}")


if __name__ == "__main__":
    main()
