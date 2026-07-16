#!/usr/bin/env python3
"""Shared PushT path-mode and trajectory-distance utilities."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np


SECTOR_NAMES = ("right", "down", "left", "up")


def sector_index_from_offset(offset_xy: np.ndarray) -> int:
    """Quantize a pusher-minus-block offset into a cardinal contact side."""
    offset = np.asarray(offset_xy, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError(f"expected a finite 2D offset, got {offset_xy}")
    angle = float(np.arctan2(offset[1], offset[0]))
    return int(np.floor((angle + np.pi / 4.0) / (np.pi / 2.0))) % 4


def first_contact_index(n_contacts: np.ndarray, state: np.ndarray) -> int:
    hits = np.flatnonzero(np.asarray(n_contacts).reshape(-1) > 0.5)
    if hits.size:
        return int(hits[0])
    distances = np.linalg.norm(state[:, :2] - state[:, 2:4], axis=1)
    return int(np.argmin(distances))


def state_approach_mode(state: np.ndarray, n_contacts: np.ndarray) -> dict[str, Any]:
    index = first_contact_index(n_contacts, state)
    offset = np.asarray(state[index, :2] - state[index, 2:4], dtype=np.float64)
    sector_index = sector_index_from_offset(offset)
    return {
        "index": index,
        "sector_index": sector_index,
        "sector": SECTOR_NAMES[sector_index],
        "angle": float(np.arctan2(offset[1], offset[0])),
        "from_contact": bool(np.asarray(n_contacts).reshape(-1)[index] > 0.5),
    }


def generated_approach_mode(
    pusher_xy: np.ndarray,
    block_xy: np.ndarray,
    surface_distance: np.ndarray,
    contact_distance_max: float = 0.025,
) -> dict[str, Any]:
    """Infer the cardinal contact mode from rendered generated tracks.

    A color-mask surface contact is preferred. If no generated frame reaches
    the calibrated contact distance, the closest observable frame is returned
    as a diagnostic fallback and ``from_contact`` is false.
    """
    pusher = np.asarray(pusher_xy, dtype=np.float64)
    block = np.asarray(block_xy, dtype=np.float64)
    surface = np.asarray(surface_distance, dtype=np.float64).reshape(-1)
    finite = np.isfinite(pusher).all(axis=-1) & np.isfinite(block).all(axis=-1)
    contact = finite & np.isfinite(surface) & (surface <= contact_distance_max)
    if contact.any():
        index = int(np.flatnonzero(contact)[0])
        from_contact = True
    else:
        candidates = np.flatnonzero(finite & np.isfinite(surface))
        if candidates.size:
            index = int(candidates[np.argmin(surface[candidates])])
        else:
            candidates = np.flatnonzero(finite)
            if not candidates.size:
                return {
                    "index": -1,
                    "sector_index": -1,
                    "sector": "missing",
                    "angle": float("nan"),
                    "from_contact": False,
                }
            center_distance = np.linalg.norm(
                pusher[candidates] - block[candidates], axis=-1
            )
            index = int(candidates[np.argmin(center_distance)])
        from_contact = False
    offset = pusher[index] - block[index]
    sector_index = sector_index_from_offset(offset)
    return {
        "index": index,
        "sector_index": sector_index,
        "sector": SECTOR_NAMES[sector_index],
        "angle": float(np.arctan2(offset[1], offset[0])),
        "from_contact": from_contact,
    }


def pose_descriptor(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return np.concatenate([state[:4], [np.sin(state[4]), np.cos(state[4])]])


def endpoint_descriptor(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return np.concatenate([pose_descriptor(state[0]), pose_descriptor(state[-1])])


def resample_path(path: np.ndarray, count: int = 17) -> np.ndarray | None:
    """Interpolate a 2D path over normalized time while tolerating missing frames."""
    path = np.asarray(path, dtype=np.float64)
    finite = np.isfinite(path).all(axis=-1)
    if finite.sum() < 2:
        return None
    source = np.linspace(0.0, 1.0, len(path))[finite]
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack(
        [np.interp(target, source, path[finite, dim]) for dim in range(path.shape[1])]
    )


def path_rms(left: np.ndarray, right: np.ndarray, count: int = 17) -> float:
    left_resampled = resample_path(left, count=count)
    right_resampled = resample_path(right, count=count)
    if left_resampled is None or right_resampled is None:
        return float("nan")
    return float(np.sqrt(np.mean((left_resampled - right_resampled) ** 2)))


def pairwise_path_rms(paths: Sequence[np.ndarray], count: int = 17) -> float:
    distances = []
    for left_index in range(len(paths)):
        for right_index in range(left_index + 1, len(paths)):
            distance = path_rms(paths[left_index], paths[right_index], count=count)
            if np.isfinite(distance):
                distances.append(distance)
    return float(np.mean(distances)) if distances else float("nan")


def nearest_path_rms(
    queries: Sequence[np.ndarray], references: Sequence[np.ndarray], count: int = 17
) -> float:
    minima = []
    for query in queries:
        distances = [
            path_rms(query, reference, count=count) for reference in references
        ]
        distances = [distance for distance in distances if np.isfinite(distance)]
        if distances:
            minima.append(min(distances))
    return float(np.mean(minima)) if minima else float("nan")


def label_entropy(labels: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    if not len(counts):
        return 0.0
    probability = counts / counts.sum()
    return float(-(probability * np.log(probability)).sum())


def effective_mode_count(labels: Iterable[str]) -> float:
    labels = list(labels)
    if not labels:
        return 0.0
    return float(math.exp(label_entropy(labels)))


def categorical_distribution(labels: Iterable[str]) -> np.ndarray:
    counter = Counter(labels)
    counts = np.asarray([counter[name] for name in SECTOR_NAMES], dtype=np.float64)
    if counts.sum() == 0:
        return np.zeros(len(SECTOR_NAMES), dtype=np.float64)
    return counts / counts.sum()


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.sum() <= 0 or right.sum() <= 0:
        return float("nan")
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def kl_divergence(value: np.ndarray, reference: np.ndarray) -> float:
        nonzero = value > 0
        return float(
            np.sum(value[nonzero] * np.log(value[nonzero] / reference[nonzero]))
        )

    return 0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle)
