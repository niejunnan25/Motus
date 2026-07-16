#!/usr/bin/env python3
"""Compare training-path modes with repeated 50-step VGM samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pusht_trajectory_metrics import (
    SECTOR_NAMES,
    categorical_distribution,
    effective_mode_count,
    endpoint_descriptor,
    generated_approach_mode,
    jensen_shannon_divergence,
    nearest_path_rms,
    pairwise_path_rms,
    resample_path,
    state_approach_mode,
)
from summarize_pusht_vgm_multimodal import task_validity


MODE_COLORS = {
    "right": "#c44e52",
    "down": "#4c72b0",
    "left": "#55a868",
    "up": "#8172b2",
    "missing": "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--nearest_train", type=int, default=32)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def condition_groups(step_dir: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in read_json(step_dir / "sample_metrics.json"):
        result.setdefault(int(row["condition_number"]), []).append(row)
    return result


def condition_dir(step_dir: Path, condition_number: int) -> Path:
    matches = list(step_dir.glob(f"condition_{condition_number:02d}_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one condition_{condition_number:02d}_* directory, found {matches}"
        )
    return matches[0]


def load_train_episodes(dataset_dir: Path) -> list[dict[str, Any]]:
    episodes = []
    for row in read_manifest(dataset_dir / "train" / "manifest.jsonl"):
        payload = np.load(dataset_dir / row["metadata_path"])
        state = payload["state"].astype(np.float64)
        mode = state_approach_mode(state, payload["n_contacts"])
        episodes.append(
            {
                "row": row,
                "state": state,
                "descriptor": endpoint_descriptor(state),
                "mode": str(mode["sector"]),
                "mode_index": int(mode["sector_index"]),
                "pusher_path": state[:, :2] / 512.0,
                "block_path": state[:, 2:4] / 512.0,
            }
        )
    return episodes


def nearest_indices(
    query: np.ndarray, references: np.ndarray, count: int
) -> np.ndarray:
    scale = references.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_query = (query - references.mean(axis=0)) / scale
    normalized_reference = (references - references.mean(axis=0)) / scale
    distance = np.linalg.norm(normalized_reference - normalized_query, axis=-1)
    return np.argsort(distance)[: min(count, len(distance))]


def load_generated_samples(
    step_dir: Path,
    condition_number: int,
    rows: Sequence[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    directory = condition_dir(step_dir, condition_number)
    samples = []
    for row in sorted(rows, key=lambda value: int(value["seed"])):
        tracks = np.load(
            directory / f"seed_{int(row['seed']):04d}" / "trajectory_tracks.npz"
        )
        mode = generated_approach_mode(
            tracks["pred_pusher_xy"],
            tracks["pred_block_xy"],
            tracks["pred_surface_distance"],
            contact_distance_max=float(thresholds["moving_contact_distance_max"]),
        )
        diagnostic = task_validity(row, tracks, thresholds)
        samples.append(
            {
                "row": row,
                "tracks": tracks,
                "mode": str(mode["sector"]),
                "mode_index": int(mode["sector_index"]),
                "mode_from_contact": bool(mode["from_contact"]),
                "strict_valid": bool(row["valid"]),
                "task_valid": bool(diagnostic["valid"]),
            }
        )
    return samples


def pca_projection(
    paths: Sequence[np.ndarray], fit_count: int | None = None
) -> np.ndarray:
    features = []
    for path in paths:
        sampled = resample_path(path, count=17)
        if sampled is None:
            features.append(np.full(34, np.nan))
        else:
            features.append(sampled.reshape(-1))
    matrix = np.stack(features)
    finite_rows = np.isfinite(matrix).all(axis=-1)
    result = np.full((len(matrix), 2), np.nan, dtype=np.float64)
    fit_count = len(matrix) if fit_count is None else fit_count
    fit_rows = finite_rows & (np.arange(len(matrix)) < fit_count)
    if fit_rows.sum() < 2:
        return result
    fit_matrix = matrix[fit_rows]
    mean = fit_matrix.mean(axis=0)
    scale = fit_matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_fit = (fit_matrix - mean) / scale
    _, _, vh = np.linalg.svd(normalized_fit, full_matrices=False)
    components = vh[: min(2, len(vh))]
    projected = ((matrix[finite_rows] - mean) / scale) @ components.T
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    result[finite_rows] = projected
    return result


def compute_metrics(
    train_neighbors: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    train_labels = [sample["mode"] for sample in train_neighbors]
    generated_contact = [sample for sample in generated if sample["mode_from_contact"]]
    generated_task = [sample for sample in generated_contact if sample["task_valid"]]
    generated_strict = [
        sample for sample in generated_contact if sample["strict_valid"]
    ]
    generated_labels = [sample["mode"] for sample in generated_task]
    train_modes = set(train_labels)
    generated_modes = set(generated_labels)
    generated_paths = [sample["tracks"]["pred_pusher_xy"] for sample in generated_task]
    train_paths = [sample["pusher_path"] for sample in train_neighbors]
    return {
        "train_neighbor_count": len(train_neighbors),
        "train_mode_count": len(train_modes),
        "train_effective_modes": effective_mode_count(train_labels),
        "generated_samples": len(generated),
        "generated_contact_rate": len(generated_contact) / len(generated),
        "strict_valid_rate": len(generated_strict) / len(generated),
        "task_valid_rate": len(generated_task) / len(generated),
        "task_valid_mode_count": len(generated_modes),
        "task_valid_effective_modes": effective_mode_count(generated_labels),
        "task_valid_mode_coverage": (
            len(train_modes & generated_modes) / len(train_modes)
            if train_modes
            else float("nan")
        ),
        "task_valid_mode_precision": (
            len(train_modes & generated_modes) / len(generated_modes)
            if generated_modes
            else float("nan")
        ),
        "task_valid_pusher_apd": pairwise_path_rms(generated_paths),
        "task_valid_nearest_train_path_rms": nearest_path_rms(
            generated_paths, train_paths
        ),
        "task_valid_mode_js_divergence": jensen_shannon_divergence(
            categorical_distribution(train_labels),
            categorical_distribution(generated_labels),
        ),
    }


def draw_endpoint_pair(axis: Any, gt_frames: np.ndarray, title: str) -> None:
    pair = np.concatenate([gt_frames[0], gt_frames[-1]], axis=1)
    axis.imshow(pair)
    axis.axvline(gt_frames.shape[2] - 0.5, color="white", linewidth=2)
    axis.text(
        0.25,
        0.04,
        "first",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axis.text(
        0.75,
        0.04,
        "goal",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    axis.set_title(title, fontsize=14)
    axis.axis("off")


def draw_paths(
    axis: Any,
    background: np.ndarray,
    samples: Sequence[dict[str, Any]],
    generated: bool,
    title: str,
) -> None:
    axis.imshow(background, extent=(0, 1, 1, 0), alpha=0.72)
    for sample in samples:
        color = MODE_COLORS[sample["mode"]]
        if generated:
            pusher = sample["tracks"]["pred_pusher_xy"]
            block = sample["tracks"]["pred_block_xy"]
            if sample["strict_valid"]:
                alpha, linewidth = 1.0, 2.8
            elif sample["task_valid"]:
                alpha, linewidth = 0.9, 2.2
            else:
                alpha, linewidth = 0.35, 1.5
            label = f"seed {int(sample['row']['seed'])}: {sample['mode']}"
        else:
            pusher = sample["pusher_path"]
            block = sample["block_path"]
            alpha, linewidth = 0.55, 1.45
            label = None
        axis.plot(
            pusher[:, 0],
            pusher[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        axis.plot(
            block[:, 0],
            block[:, 1],
            color=color,
            alpha=alpha * 0.65,
            linewidth=1.1,
            linestyle="--",
        )
        if generated and np.isfinite(pusher).all(axis=-1).any():
            finite = np.flatnonzero(np.isfinite(pusher).all(axis=-1))
            axis.scatter(
                pusher[finite[0], 0],
                pusher[finite[0], 1],
                color=color,
                s=20,
                marker="o",
            )
            axis.scatter(
                pusher[finite[-1], 0],
                pusher[finite[-1], 1],
                color=color,
                s=32,
                marker="*",
            )
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=14)
    axis.set_xlabel("solid: pusher; dashed: T block", fontsize=10)


def draw_mode_bars(
    axis: Any,
    train_neighbors: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
) -> None:
    train_counts = Counter(sample["mode"] for sample in train_neighbors)
    generated_counts = Counter(
        sample["mode"]
        for sample in generated
        if sample["task_valid"] and sample["mode_from_contact"]
    )
    x = np.arange(len(SECTOR_NAMES))
    width = 0.38
    train_probability = np.asarray(
        [train_counts[name] for name in SECTOR_NAMES], dtype=float
    )
    train_probability /= max(train_probability.sum(), 1.0)
    generated_probability = np.asarray(
        [generated_counts[name] for name in SECTOR_NAMES], dtype=float
    )
    generated_probability /= max(generated_probability.sum(), 1.0)
    axis.bar(
        x - width / 2, train_probability, width, label="nearest train", color="#767676"
    )
    axis.bar(
        x + width / 2,
        generated_probability,
        width,
        label="task-valid VGM",
        color=[MODE_COLORS[name] for name in SECTOR_NAMES],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(x, SECTOR_NAMES)
    axis.set_ylim(0, 1)
    axis.set_ylabel("probability")
    axis.set_title("Contact-side mode distribution", fontsize=14)
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.2)


def draw_embedding(
    axis: Any,
    train_neighbors: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
) -> None:
    paths = [sample["pusher_path"] for sample in train_neighbors] + [
        sample["tracks"]["pred_pusher_xy"] for sample in generated
    ]
    train_count = len(train_neighbors)
    projection = pca_projection(paths, fit_count=train_count)
    for index, sample in enumerate(train_neighbors):
        axis.scatter(
            projection[index, 0],
            projection[index, 1],
            color=MODE_COLORS[sample["mode"]],
            alpha=0.45,
            s=24,
            marker="o",
        )
    for offset, sample in enumerate(generated):
        point = projection[train_count + offset]
        marker = "*" if sample["task_valid"] else "x"
        axis.scatter(
            point[0],
            point[1],
            color=MODE_COLORS[sample["mode"]],
            s=95,
            marker=marker,
            edgecolor="black" if marker == "*" else None,
            linewidth=0.5,
        )
        axis.annotate(
            str(int(sample["row"]["seed"])),
            point,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_title("Pusher-path PCA: train circles, VGM stars/x", fontsize=14)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.2)


def draw_metrics(axis: Any, metrics: dict[str, Any]) -> None:
    axis.axis("off")
    rows = (
        ("Nearest-train modes", f"{metrics['train_mode_count']} / 4"),
        ("Train effective modes", f"{metrics['train_effective_modes']:.2f}"),
        ("Generated contact rate", f"{metrics['generated_contact_rate']:.1%}"),
        ("Strict valid rate", f"{metrics['strict_valid_rate']:.1%}"),
        ("Task-valid rate", f"{metrics['task_valid_rate']:.1%}"),
        ("Task-valid modes", f"{metrics['task_valid_mode_count']} / 4"),
        ("Valid mode coverage", f"{metrics['task_valid_mode_coverage']:.1%}"),
        ("Pusher APD", f"{metrics['task_valid_pusher_apd']:.4f}"),
        (
            "Nearest-train path RMS",
            f"{metrics['task_valid_nearest_train_path_rms']:.4f}",
        ),
        ("Mode JS divergence", f"{metrics['task_valid_mode_js_divergence']:.4f}"),
    )
    table = axis.table(
        cellText=rows,
        colLabels=("Metric", "Value"),
        loc="center",
        cellLoc="left",
        colWidths=(0.7, 0.3),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)
    axis.set_title("Validity-aware multimodality metrics", fontsize=14)


def plot_condition(
    output_path: Path,
    condition_number: int,
    episode_name: str,
    gt_frames: np.ndarray,
    train_neighbors: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    draw_endpoint_pair(
        axes[0, 0], gt_frames, f"Condition {condition_number:02d}: {episode_name}"
    )
    draw_paths(
        axes[0, 1],
        gt_frames[0],
        train_neighbors,
        generated=False,
        title=f"{len(train_neighbors)} nearest training endpoint pairs",
    )
    draw_paths(
        axes[0, 2],
        gt_frames[0],
        generated,
        generated=True,
        title="50-step VGM: 8 seeds",
    )
    draw_mode_bars(axes[1, 0], train_neighbors, generated)
    draw_embedding(axes[1, 1], train_neighbors, generated)
    draw_metrics(axes[1, 2], metrics)
    figure.suptitle(
        "Same endpoints: training-supported path modes versus generated modes",
        fontsize=19,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_global_training_atlas(
    output_path: Path,
    episodes: Sequence[dict[str, Any]],
    condition_metrics: Sequence[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for episode in episodes:
        relative = episode["pusher_path"] - episode["block_path"][0]
        axes[0, 0].plot(
            relative[:, 0],
            relative[:, 1],
            color=MODE_COLORS[episode["mode"]],
            alpha=0.22,
            linewidth=1.0,
        )
    axes[0, 0].scatter([0], [0], marker="x", color="black", s=80)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title(
        "All training pusher paths relative to initial T center", fontsize=14
    )

    counts = Counter(episode["mode"] for episode in episodes)
    axes[0, 1].bar(
        SECTOR_NAMES,
        [counts[name] for name in SECTOR_NAMES],
        color=[MODE_COLORS[name] for name in SECTOR_NAMES],
    )
    axes[0, 1].set_title("Training first-contact modes", fontsize=14)
    axes[0, 1].set_ylabel("episodes")

    projection = pca_projection([episode["pusher_path"] for episode in episodes])
    for index, episode in enumerate(episodes):
        axes[1, 0].scatter(
            projection[index, 0],
            projection[index, 1],
            color=MODE_COLORS[episode["mode"]],
            alpha=0.65,
            s=26,
        )
    axes[1, 0].set_title("Training pusher-path PCA", fontsize=14)
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    axes[1, 0].grid(alpha=0.2)

    x = np.arange(len(condition_metrics))
    width = 0.35
    axes[1, 1].bar(
        x - width / 2,
        [row["train_mode_count"] for row in condition_metrics],
        width,
        label="nearest-train modes",
        color="#777777",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["task_valid_mode_count"] for row in condition_metrics],
        width,
        label="task-valid VGM modes",
        color="#4c72b0",
    )
    axes[1, 1].set_xticks(
        x, [f"C{index:02d}" for index in range(len(condition_metrics))]
    )
    axes[1, 1].set_ylim(0, 4.3)
    axes[1, 1].set_ylabel("distinct contact-side modes")
    axes[1, 1].set_title("Held-out endpoint neighborhoods versus VGM", fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "PushT training multimodality and held-out 50-step VGM coverage", fontsize=18
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_inference_path_atlas(
    output_path: Path,
    condition_records: Sequence[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for axis, record in zip(axes.flat, condition_records):
        draw_paths(
            axis,
            record["gt_frames"][0],
            record["generated"],
            generated=True,
            title=(
                f"C{record['condition_number']:02d} {record['episode_name']} | "
                f"valid modes={record['metrics']['task_valid_mode_count']}"
            ),
        )
    figure.suptitle(
        "50-step VGM path samples: eight seeds per held-out endpoint pair",
        fontsize=18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    step_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if int(read_json(step_dir / "run_summary.json")["num_inference_steps"]) != 50:
        raise ValueError(
            f"multimodality visualization requires the formal 50-step run: {step_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_episodes = load_train_episodes(dataset_dir)
    train_descriptors = np.stack([episode["descriptor"] for episode in train_episodes])
    test_rows = {
        row["episode_name"]: row
        for row in read_manifest(dataset_dir / "test" / "manifest.jsonl")
    }
    thresholds = read_json(step_dir / "calibration.json")["thresholds"]
    groups = condition_groups(step_dir)
    windows = {
        int(row["condition_number"]): row
        for row in read_json(step_dir / "windows.json")
    }

    metrics_rows = []
    sample_rows = []
    condition_records = []
    for condition_number in sorted(groups):
        window = windows[condition_number]
        episode_name = str(window["episode_name"])
        test_payload = np.load(dataset_dir / test_rows[episode_name]["metadata_path"])
        query_descriptor = endpoint_descriptor(test_payload["state"])
        neighbor_indices = nearest_indices(
            query_descriptor, train_descriptors, args.nearest_train
        )
        train_neighbors = [train_episodes[index] for index in neighbor_indices]
        generated = load_generated_samples(
            step_dir,
            condition_number,
            groups[condition_number],
            thresholds,
        )
        first_tracks = generated[0]["tracks"]
        metrics = compute_metrics(train_neighbors, generated)
        metrics.update(
            {"condition_number": condition_number, "episode_name": episode_name}
        )
        metrics_rows.append(metrics)
        condition_records.append(
            {
                "condition_number": condition_number,
                "episode_name": episode_name,
                "gt_frames": first_tracks["gt_frames"],
                "generated": generated,
                "metrics": metrics,
            }
        )
        for sample in generated:
            sample_rows.append(
                {
                    "condition_number": condition_number,
                    "episode_name": episode_name,
                    "seed": int(sample["row"]["seed"]),
                    "mode": sample["mode"],
                    "mode_from_contact": sample["mode_from_contact"],
                    "mode_supported_by_nearest_train": sample["mode"]
                    in {neighbor["mode"] for neighbor in train_neighbors},
                    "strict_valid": sample["strict_valid"],
                    "task_valid": sample["task_valid"],
                    "pusher_detection_rate": float(
                        sample["row"]["pusher_detection_rate"]
                    ),
                }
            )
        plot_condition(
            output_dir
            / "by_condition"
            / f"condition_{condition_number:02d}_mode_diagnostic.png",
            condition_number,
            episode_name,
            first_tracks["gt_frames"],
            train_neighbors,
            generated,
            metrics,
        )

    plot_global_training_atlas(
        output_dir / "training_and_inference_mode_atlas.png",
        train_episodes,
        metrics_rows,
    )
    plot_inference_path_atlas(
        output_dir / "inference_50step_path_atlas.png",
        condition_records,
    )
    aggregate = {
        "formal_num_inference_steps": 50,
        "conditions": len(metrics_rows),
        "seeds_per_condition": len(sample_rows) // len(metrics_rows),
        "nearest_train_per_condition": args.nearest_train,
        "mean_train_mode_count": float(
            np.mean([row["train_mode_count"] for row in metrics_rows])
        ),
        "mean_task_valid_mode_count": float(
            np.mean([row["task_valid_mode_count"] for row in metrics_rows])
        ),
        "mean_task_valid_mode_coverage": float(
            np.mean([row["task_valid_mode_coverage"] for row in metrics_rows])
        ),
        "mean_strict_valid_rate": float(
            np.mean([row["strict_valid_rate"] for row in metrics_rows])
        ),
        "mean_task_valid_rate": float(
            np.mean([row["task_valid_rate"] for row in metrics_rows])
        ),
        "interpretation": (
            "Mode count is only credited for generated samples that pass the post-hoc task-valid "
            "checks and exhibit an observed contact; strict validity remains the primary criterion."
        ),
    }
    (output_dir / "mode_metrics.json").write_text(
        json.dumps({"aggregate": aggregate, "conditions": metrics_rows}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sample_modes.json").write_text(
        json.dumps(sample_rows, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "mode_metrics.csv", metrics_rows)
    write_csv(output_dir / "sample_modes.csv", sample_rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
