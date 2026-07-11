#!/usr/bin/env python3
"""Shared view-aware role helpers for MaskWAM LIBERO annotations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


BASE_TRAIN_ROLES: Tuple[str, ...] = ("object", "target", "robot")
OPTIONAL_BASE_ROLES: Tuple[str, ...] = ("context",)
VIEW_NAMES: Tuple[str, ...] = ("main", "wrist")
VIEW_TRAIN_ROLES: Tuple[str, ...] = tuple(
    f"{view}_{role}" for view in VIEW_NAMES for role in BASE_TRAIN_ROLES
)
VIEW_ROLE_ORDER: Dict[str, int] = {role: idx for idx, role in enumerate(VIEW_TRAIN_ROLES)}
VALID_CORRECTION_ROLES = set(BASE_TRAIN_ROLES) | set(OPTIONAL_BASE_ROLES) | set(VIEW_TRAIN_ROLES)
FINAL_ROLE = "final_train_mask"


def is_view_role(role: str | None) -> bool:
    return isinstance(role, str) and any(role.startswith(f"{view}_") for view in VIEW_NAMES)


def view_name(role: str | None) -> str | None:
    if not is_view_role(role):
        return None
    return str(role).split("_", 1)[0]


def base_role(role: str | None) -> str | None:
    if not isinstance(role, str):
        return None
    if is_view_role(role):
        return role.split("_", 1)[1]
    return role


def view_role(view: str, role: str) -> str:
    return f"{view}_{role}"


def view_roles_for_base(role: str) -> List[str]:
    if role not in BASE_TRAIN_ROLES:
        return []
    return [view_role(view, role) for view in VIEW_NAMES]


def configured_base_train_roles(sample: Dict[str, Any]) -> List[str]:
    roles = {
        str(prompt.get("role"))
        for prompt in sample.get("prompts", [])
        if prompt.get("role") in BASE_TRAIN_ROLES
    }
    return [role for role in BASE_TRAIN_ROLES if role in roles]


def configured_view_train_roles(sample: Dict[str, Any]) -> List[str]:
    roles: List[str] = []
    for role in configured_base_train_roles(sample):
        roles.extend(view_roles_for_base(role))
    return roles


def configured_output_roles(sample: Dict[str, Any], include_context: bool = True) -> List[str]:
    roles = configured_base_train_roles(sample)
    roles.extend(configured_view_train_roles(sample))
    if include_context and any(prompt.get("role") == "context" for prompt in sample.get("prompts", [])):
        roles.append("context")
    roles.append(FINAL_ROLE)
    return roles


def frame_view_slices(shape: Tuple[int, int], view_layout: str) -> Dict[str, Tuple[slice, slice]]:
    height, width = shape
    if view_layout == "vertical":
        split = height // 2
        return {
            "main": (slice(0, split), slice(0, width)),
            "wrist": (slice(split, height), slice(0, width)),
        }
    if view_layout == "horizontal":
        split = width // 2
        return {
            "main": (slice(0, height), slice(0, split)),
            "wrist": (slice(0, height), slice(split, width)),
        }
    return {"main": (slice(0, height), slice(0, width))}


def restrict_mask_to_view(mask: np.ndarray, view: str, view_layout: str) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    view_slices = frame_view_slices(mask.shape, view_layout)
    if view not in view_slices:
        return out
    ys, xs = view_slices[view]
    out[ys, xs] = mask.astype(bool)[ys, xs]
    return out


def split_masks_by_view(
    masks: Sequence[np.ndarray],
    view_layout: str,
) -> Dict[str, List[np.ndarray]]:
    return {
        view: [restrict_mask_to_view(mask, view, view_layout) for mask in masks]
        for view in VIEW_NAMES
    }


def union_view_train_roles(
    role_to_masks: Dict[str, Sequence[np.ndarray]],
    frame_count: int,
    shape: Tuple[int, int],
    roles: Iterable[str] = VIEW_TRAIN_ROLES,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for frame_idx in range(frame_count):
        combined = np.zeros(shape, dtype=bool)
        for role in roles:
            masks = role_to_masks.get(role)
            if masks is not None and frame_idx < len(masks):
                combined |= masks[frame_idx].astype(bool)
        out.append(combined)
    return out


def role_color(role: str) -> Tuple[int, int, int]:
    base = base_role(role)
    if base == "object":
        return (255, 60, 60)
    if base == "target":
        return (45, 175, 255)
    if base == "robot":
        return (185, 100, 255)
    if role == FINAL_ROLE:
        return (235, 40, 40)
    return (120, 150, 160)
