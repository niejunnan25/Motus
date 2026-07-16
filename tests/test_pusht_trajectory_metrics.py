from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pusht_trajectory_metrics import (  # noqa: E402
    effective_mode_count,
    generated_approach_mode,
    resample_path,
    sector_index_from_offset,
)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        ((1.0, 0.0), 0),
        ((0.0, 1.0), 1),
        ((-1.0, 0.0), 2),
        ((0.0, -1.0), 3),
    ],
)
def test_cardinal_sector_mapping(offset: tuple[float, float], expected: int) -> None:
    assert sector_index_from_offset(np.asarray(offset)) == expected


def test_generated_mode_prefers_first_surface_contact() -> None:
    pusher = np.asarray([[0.0, 0.0], [0.5, 0.2], [0.8, 0.5]])
    block = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    surface = np.asarray([0.2, 0.01, 0.0])

    mode = generated_approach_mode(pusher, block, surface)

    assert mode["index"] == 1
    assert mode["sector"] == "up"
    assert mode["from_contact"] is True


def test_resample_path_interpolates_missing_slots() -> None:
    path = np.asarray([[0.0, 0.0], [np.nan, np.nan], [1.0, 2.0]])
    result = resample_path(path, count=3)

    assert result is not None
    np.testing.assert_allclose(result, [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])


def test_empty_effective_mode_count_is_zero() -> None:
    assert effective_mode_count([]) == 0.0
