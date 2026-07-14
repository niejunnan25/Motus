#!/usr/bin/env python3
"""Summarize A0-A3 Progress outputs under Robo-Dopamine LIBERO sampling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=EVAL_DIR",
        help="A Progress evaluation directory containing episode_outputs.pt.",
    )
    parser.add_argument(
        "--official_result",
        action="append",
        default=[],
        metavar="NAME=SUMMARY_JSON",
        help="Optional official GRM summary_voc.json for a separate protocol table.",
    )
    parser.add_argument("--intervals", type=int, nargs="+", default=[30, 10, 5])
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise ValueError(f"Missing name in {value!r}")
    return name.strip(), Path(raw_path).expanduser().resolve()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_run(name: str, root: Path) -> Dict[str, Any]:
    outputs_path = root / "episode_outputs.pt"
    metrics_path = root / "metrics.json"
    if not outputs_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"{name} needs episode_outputs.pt and metrics.json under {root}"
        )
    payload = torch_load(outputs_path)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported episode output schema in {outputs_path}")
    episodes: Dict[str, Dict[str, Any]] = {}
    for item in payload["episodes"]:
        episode_name = str(item["episode_name"])
        if episode_name in episodes:
            raise ValueError(f"Duplicate episode {episode_name!r} in {outputs_path}")
        episodes[episode_name] = {
            "task_index": int(torch.as_tensor(item["task_index"]).item()),
            "total_frames": int(torch.as_tensor(item["total_frames"]).item()),
            "frame_indices": torch.as_tensor(item["frame_indices"]).long(),
            "target": torch.as_tensor(item["target"]).float(),
            "prediction": torch.as_tensor(item["prediction"]).float(),
        }
    return {
        "name": name,
        "root": root,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "episodes": episodes,
    }


def validate_runs(runs: Sequence[Mapping[str, Any]]) -> None:
    reference = runs[0]["episodes"]
    for run in runs:
        episodes = run["episodes"]
        if set(episodes) != set(reference):
            raise ValueError(f"Episode set differs for {run['name']}")
        for episode_name, expected in reference.items():
            actual = episodes[episode_name]
            if actual["task_index"] != expected["task_index"]:
                raise ValueError(f"Task differs for {run['name']}/{episode_name}")
            if actual["total_frames"] != expected["total_frames"]:
                raise ValueError(f"Frame count differs for {run['name']}/{episode_name}")
            if not torch.equal(actual["frame_indices"], expected["frame_indices"]):
                raise ValueError(f"Frame indices differ for {run['name']}/{episode_name}")
            if not torch.allclose(actual["target"], expected["target"], atol=1e-7):
                raise ValueError(f"Progress targets differ for {run['name']}/{episode_name}")


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool) -> float:
    if left.size < 2:
        return 0.0
    if ranked:
        left = rankdata(left)
        right = rankdata(right)
    left = left.astype(np.float64) - float(np.mean(left))
    right = right.astype(np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def make_sample_indices(num_frames: int, count: int) -> List[int]:
    """Mirror Robo-Dopamine eval/evaluation_grm.py exactly."""
    if num_frames < 1:
        return []
    if count <= 1:
        return [0]
    return [round(index * (num_frames - 1) / (count - 1)) for index in range(count)]


def sampled_episode_metrics(episode: Mapping[str, Any], interval: int) -> Dict[str, Any]:
    total_frames = int(episode["total_frames"])
    sample_count = total_frames // int(interval)
    # Official GRM predicts a hop for each transition, so its VOC vector omits
    # the initial state. Use the same sampled after-states for absolute Progress.
    sampled_indices = make_sample_indices(total_frames, sample_count)[1:]
    position = {
        int(frame): offset
        for offset, frame in enumerate(episode["frame_indices"].tolist())
    }
    missing = [frame for frame in sampled_indices if frame not in position]
    if missing:
        raise ValueError(
            f"Cache is missing official interval={interval} query frames: {missing[:5]}"
        )
    offsets = torch.tensor([position[frame] for frame in sampled_indices], dtype=torch.long)
    prediction = episode["prediction"].index_select(0, offsets).numpy()
    target = episode["target"].index_select(0, offsets).numpy()
    error = prediction - target
    differences = np.diff(prediction)
    return {
        "sample_count_with_initial": sample_count,
        "evaluated_states": len(sampled_indices),
        "degenerate_voc": len(sampled_indices) < 2,
        "spearman": correlation(prediction, target, ranked=True),
        "pearson": correlation(prediction, target, ranked=False),
        "mae": float(np.mean(np.abs(error))) if error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0,
        "monotonic_violation_rate": (
            float(np.mean(differences < -1e-4)) if differences.size else 0.0
        ),
    }


def aggregate_interval(
    run: Mapping[str, Any], interval: int
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for episode_name in sorted(run["episodes"]):
        episode = run["episodes"][episode_name]
        row = {
            "model": run["name"],
            "interval": interval,
            "episode_name": episode_name,
            "task_index": int(episode["task_index"]),
            "total_frames": int(episode["total_frames"]),
            **sampled_episode_metrics(episode, interval),
        }
        rows.append(row)
    task_ids = sorted({int(row["task_index"]) for row in rows})
    metrics = ("spearman", "pearson", "mae", "rmse", "monotonic_violation_rate")
    summary: Dict[str, Any] = {
        "model": run["name"],
        "interval": interval,
        "episodes": len(rows),
        "tasks": len(task_ids),
        "evaluated_states": sum(int(row["evaluated_states"]) for row in rows),
        "degenerate_voc_episodes": sum(bool(row["degenerate_voc"]) for row in rows),
        "perfect_non_degenerate_voc_ceiling": 1.0
        - sum(bool(row["degenerate_voc"]) for row in rows) / len(rows),
    }
    for metric in metrics:
        summary[f"episode_mean_{metric}"] = float(
            np.mean([float(row[metric]) for row in rows])
        )
        task_values = []
        for task_id in task_ids:
            values = [float(row[metric]) for row in rows if int(row["task_index"]) == task_id]
            task_values.append(float(np.mean(values)))
        summary[f"task_macro_{metric}"] = float(np.mean(task_values))
    return summary, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_official_results(specs: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        name, path = parse_named_path(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        subset = summary.get("libero_test_100")
        if subset is None and len(summary) == 1:
            subset = next(iter(summary.values()))
        if not isinstance(subset, dict):
            raise ValueError(f"No LIBERO summary found in {path}")
        voc_plus = subset.get("voc+")
        voc_minus = subset.get("voc-")
        rows.append(
            {
                "model": name,
                "result_path": str(path),
                "interval": int(payload.get("meta", {}).get("interval", 30)),
                "voc_plus": voc_plus,
                "voc_minus": voc_minus,
                "voc_mean": (
                    0.5 * (float(voc_plus) + float(voc_minus))
                    if voc_plus is not None and voc_minus is not None
                    else None
                ),
            }
        )
    return rows


def plot_interval_metrics(
    runs: Sequence[Mapping[str, Any]],
    intervals: Sequence[int],
    summaries: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    lookup = {(row["model"], int(row["interval"])): row for row in summaries}
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    x = np.arange(len(intervals), dtype=np.float64)
    width = 0.8 / len(runs)
    for run_index, (run, color) in enumerate(zip(runs, COLORS)):
        offsets = x + (run_index - (len(runs) - 1) / 2) * width
        voc = [lookup[(run["name"], interval)]["episode_mean_spearman"] for interval in intervals]
        mae = [lookup[(run["name"], interval)]["episode_mean_mae"] for interval in intervals]
        axes[0].bar(offsets, voc, width=width, label=run["name"], color=color)
        axes[1].bar(offsets, mae, width=width, label=run["name"], color=color)
    for axis in axes:
        axis.set_xticks(x, [str(interval) for interval in intervals])
        axis.set_xlabel("Robo-Dopamine interval (smaller is denser)")
        axis.grid(axis="y", alpha=0.22)
        axis.legend()
    axes[0].set_ylabel("Episode-mean Spearman VOC")
    axes[0].set_title("Absolute Progress ordering under released-bench sampling")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Episode-mean absolute Progress error")
    axes[1].set_title("Absolute calibration error on the same sampled states")
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def report(
    runs: Sequence[Mapping[str, Any]],
    intervals: Sequence[int],
    summaries: Sequence[Mapping[str, Any]],
    official: Sequence[Mapping[str, Any]],
) -> str:
    lookup = {(row["model"], int(row["interval"])): row for row in summaries}
    lines = [
        "# Robo-Dopamine-Bench LIBERO Progress Evaluation",
        "",
        "A0-A3 consume one current frame and a cached 53-slot generated trajectory, then output absolute Progress. Official GRM consumes before/after multi-view pairs and outputs a relative hop. The tables are therefore kept separate; only forward chronological ordering is a shared diagnostic.",
        "",
        "## Absolute Progress Results",
        "",
        "| Interval | Model | Episodes | States | Degenerate VOC episodes | Spearman | Pearson | MAE | RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for interval in intervals:
        for run in runs:
            row = lookup[(run["name"], interval)]
            lines.append(
                f"| {interval} | {run['name']} | {row['episodes']} | {row['evaluated_states']} | "
                f"{row['degenerate_voc_episodes']} | {row['episode_mean_spearman']:.6f} | "
                f"{row['episode_mean_pearson']:.6f} | {row['episode_mean_mae']:.6f} | "
                f"{row['episode_mean_rmse']:.6f} |"
            )
    lines.extend(
        [
            "",
            "At interval 30, episodes shorter than 90 frames yield only one predicted after-state. The released evaluator assigns VOC=0 because correlation is undefined; `perfect_non_degenerate_voc_ceiling` records this protocol ceiling.",
        ]
    )
    if official:
        lines.extend(
            [
                "",
                "## Official Relative-Hop GRM Results",
                "",
                "| Model | Interval | VOC+ | VOC- | Mean |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in official:
            lines.append(
                f"| {row['model']} | {row['interval']} | {float(row['voc_plus']):.6f} | "
                f"{float(row['voc_minus']):.6f} | {float(row['voc_mean']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Official `voc-` reverses pair direction and asks GRM for regress hops. An absolute single-frame Progress model has no equivalent native reverse-hop output, so no synthetic A0-A3 `voc-` is reported. Dense MAE, ordering, monotonicity, and per-case alignment maps remain the primary A0-A3 diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if any(interval < 1 for interval in args.intervals):
        raise ValueError("Intervals must be positive")
    runs = [load_run(*parse_named_path(spec)) for spec in args.run]
    validate_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    for interval in args.intervals:
        for run in runs:
            summary, rows = aggregate_interval(run, interval)
            summaries.append(summary)
            episode_rows.extend(rows)
    official = load_official_results(args.official_result)

    write_csv(args.output_dir / "progress_interval_summary.csv", summaries)
    write_csv(args.output_dir / "progress_interval_per_episode.csv", episode_rows)
    write_csv(args.output_dir / "official_grm_summary.csv", official)
    (args.output_dir / "progress_interval_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    plot_interval_metrics(
        runs, args.intervals, summaries, args.output_dir / "interval_voc_mae.png"
    )
    (args.output_dir / "REPORT.md").write_text(
        report(runs, args.intervals, summaries, official), encoding="utf-8"
    )
    print(f"Robo-Dopamine LIBERO summary complete: {args.output_dir}")


if __name__ == "__main__":
    main()
