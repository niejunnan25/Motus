#!/usr/bin/env python3
"""Audit whether the prepared PushT demonstrations contain local path modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


SECTOR_NAMES = ("right", "down", "left", "up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--nearest_neighbors", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_contact_index(n_contacts: np.ndarray, state: np.ndarray) -> int:
    hits = np.flatnonzero(n_contacts.reshape(-1) > 0.5)
    if hits.size:
        return int(hits[0])
    distances = np.linalg.norm(state[:, :2] - state[:, 2:4], axis=1)
    return int(np.argmin(distances))


def approach_sector(state: np.ndarray, n_contacts: np.ndarray) -> tuple[int, float]:
    index = first_contact_index(n_contacts, state)
    delta = state[index, :2] - state[index, 2:4]
    angle = float(np.arctan2(delta[1], delta[0]))
    sector = int(np.floor((angle + np.pi / 4.0) / (np.pi / 2.0))) % 4
    return sector, angle


def resample_path(path: np.ndarray, count: int = 64) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(path))
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(target, source, path[:, dim]) for dim in range(path.shape[1])])


def normalized_nearest(descriptors: np.ndarray, neighbor_count: int) -> np.ndarray:
    scale = descriptors.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (descriptors - descriptors.mean(axis=0)) / scale
    distances = cdist(normalized, normalized)
    np.fill_diagonal(distances, np.inf)
    return np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]


def normalized_cross_nearest(
    query_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    mean = reference_descriptors.mean(axis=0)
    scale = reference_descriptors.std(axis=0)
    scale[scale < 1e-6] = 1.0
    query = (query_descriptors - mean) / scale
    reference = (reference_descriptors - mean) / scale
    distances = cdist(query, reference)
    return np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]


def pose_descriptor(state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[:4], [np.sin(state[4]), np.cos(state[4])]])


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(dataset_dir / "train" / "manifest.jsonl") + load_manifest(
        dataset_dir / "test" / "manifest.jsonl"
    )

    descriptors = []
    endpoint_descriptors = []
    sectors = []
    angles = []
    canonical_paths = []
    splits = []
    lengths = []
    for row in rows:
        payload = np.load(dataset_dir / row["metadata_path"])
        state = payload["state"].astype(np.float64)
        n_contacts = payload["n_contacts"].astype(np.float64)
        sector, angle = approach_sector(state, n_contacts)
        relative_path = state[:, :2] - state[0, 2:4]
        descriptors.append(payload["start_descriptor"].astype(np.float64))
        endpoint_descriptors.append(np.concatenate([pose_descriptor(state[0]), pose_descriptor(state[-1])]))
        sectors.append(sector)
        angles.append(angle)
        canonical_paths.append(resample_path(relative_path))
        splits.append(row["split"])
        lengths.append(len(state))

    neighbor_count = min(args.nearest_neighbors, len(rows) - 1)
    descriptors_array = np.stack(descriptors)
    endpoint_array = np.stack(endpoint_descriptors)
    sectors_array = np.asarray(sectors)
    splits_array = np.asarray(splits)
    nearest = normalized_nearest(descriptors_array, neighbor_count)
    endpoint_nearest = normalized_nearest(endpoint_array, neighbor_count)
    local_mode_counts = np.array(
        [len(set(sectors_array[indices].tolist())) for indices in nearest], dtype=np.int64
    )
    endpoint_mode_counts = np.array(
        [len(set(sectors_array[indices].tolist())) for indices in endpoint_nearest], dtype=np.int64
    )

    train_mask = splits_array == "train"
    test_mask = splits_array == "test"
    train_neighbor_count = min(neighbor_count, int(train_mask.sum()) - 1)
    train_endpoint_nearest = normalized_nearest(
        endpoint_array[train_mask], train_neighbor_count
    )
    train_sectors = sectors_array[train_mask]
    train_endpoint_mode_counts = np.array(
        [len(set(train_sectors[indices].tolist())) for indices in train_endpoint_nearest],
        dtype=np.int64,
    )
    test_neighbor_count = min(neighbor_count, int(train_mask.sum()))
    test_to_train_nearest = normalized_cross_nearest(
        endpoint_array[test_mask], endpoint_array[train_mask], test_neighbor_count
    )
    test_to_train_mode_counts = np.array(
        [len(set(train_sectors[indices].tolist())) for indices in test_to_train_nearest],
        dtype=np.int64,
    )

    sector_counts = {
        name: int(np.sum(np.asarray(sectors) == index))
        for index, name in enumerate(SECTOR_NAMES)
    }
    summary = {
        "num_episodes": len(rows),
        "split_counts": {
            split: int(np.sum(np.asarray(splits) == split)) for split in sorted(set(splits))
        },
        "sector_counts": sector_counts,
        "nearest_neighbors": neighbor_count,
        "episodes_with_at_least_two_local_modes": int(np.sum(local_mode_counts >= 2)),
        "episodes_with_at_least_three_local_modes": int(np.sum(local_mode_counts >= 3)),
        "local_mode_count_mean": float(local_mode_counts.mean()),
        "endpoint_nearest_episodes_with_at_least_two_modes": int(np.sum(endpoint_mode_counts >= 2)),
        "endpoint_nearest_episodes_with_at_least_three_modes": int(np.sum(endpoint_mode_counts >= 3)),
        "endpoint_nearest_mode_count_mean": float(endpoint_mode_counts.mean()),
        "train_endpoint_nearest_neighbors": train_neighbor_count,
        "train_endpoint_nearest_episodes_with_at_least_two_modes": int(
            np.sum(train_endpoint_mode_counts >= 2)
        ),
        "train_endpoint_nearest_episodes_with_at_least_three_modes": int(
            np.sum(train_endpoint_mode_counts >= 3)
        ),
        "train_endpoint_nearest_mode_count_mean": float(train_endpoint_mode_counts.mean()),
        "test_to_train_endpoint_nearest_neighbors": test_neighbor_count,
        "test_endpoint_queries_with_at_least_two_train_modes": int(
            np.sum(test_to_train_mode_counts >= 2)
        ),
        "test_endpoint_queries_with_at_least_three_train_modes": int(
            np.sum(test_to_train_mode_counts >= 3)
        ),
        "test_to_train_endpoint_mode_count_mean": float(test_to_train_mode_counts.mean()),
        "length_percentiles": {
            str(percentile): float(np.percentile(lengths, percentile))
            for percentile in (0, 10, 25, 50, 75, 90, 100)
        },
    }
    (output_dir / "dataset_mode_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 4, figsize=(23, 5.5))
    colors = ("#d1495b", "#2d6a9f", "#4f8a4b", "#8b5fbf")
    for path, sector in zip(canonical_paths, sectors):
        axes[0].plot(path[:, 0], path[:, 1], color=colors[sector], alpha=0.22, linewidth=1.0)
    axes[0].scatter([0], [0], color="black", marker="x", s=80, label="initial T center")
    axes[0].set_title("Pusher paths relative to initial T center")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].hist(angles, bins=24, color="#3c6e71", edgecolor="white")
    axes[1].set_title("First-contact approach angle")
    axes[1].set_xlabel("angle (radians)")
    axes[1].set_ylabel("episodes")

    bins = np.arange(0.5, 5.5, 1.0)
    axes[2].hist(local_mode_counts, bins=bins, color="#d4a017", edgecolor="white")
    axes[2].set_xticks((1, 2, 3, 4))
    axes[2].set_title(f"Modes among {neighbor_count} nearest start conditions")
    axes[2].set_xlabel("distinct approach sectors")
    axes[2].set_ylabel("episodes")

    axes[3].hist(endpoint_mode_counts, bins=bins, color="#b56576", edgecolor="white")
    axes[3].set_xticks((1, 2, 3, 4))
    axes[3].set_title(f"Modes among {neighbor_count} nearest endpoint pairs")
    axes[3].set_xlabel("distinct approach sectors")
    axes[3].set_ylabel("episodes")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_path_modes.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
