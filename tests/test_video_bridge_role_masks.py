from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset, video_bridge_collate_fn


def role_mask_dataset(render_mode: str) -> VideoBridgeDataset:
    dataset = object.__new__(VideoBridgeDataset)
    dataset.role_mask_render_mode = render_mode
    dataset.role_mask_foreground_ids = (1, 2, 4)
    dataset.role_mask_palette = {
        0: np.asarray([0, 0, 0], dtype=np.uint8),
        1: np.asarray([255, 0, 0], dtype=np.uint8),
        2: np.asarray([0, 255, 0], dtype=np.uint8),
        4: np.asarray([0, 0, 255], dtype=np.uint8),
    }
    return dataset


def test_binary_role_mask_rendering() -> None:
    role_ids = np.asarray([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8)
    rendered = role_mask_dataset("binary")._render_role_mask(role_ids)
    assert rendered.shape == (1, 7, 3)
    assert rendered[:, :, 0].tolist() == [[0, 255, 255, 0, 255, 0, 0]]
    assert np.array_equal(rendered[:, :, 0], rendered[:, :, 1])
    assert np.array_equal(rendered[:, :, 1], rendered[:, :, 2])


def test_color_role_mask_rendering() -> None:
    role_ids = np.asarray([[0, 1, 2, 4, 3]], dtype=np.uint8)
    rendered = role_mask_dataset("role_color")._render_role_mask(role_ids)
    assert rendered.tolist() == [
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]
    ]


def test_role_mask_disk_cache_roundtrip(tmp_path: Path) -> None:
    dataset = role_mask_dataset("binary")
    dataset.role_mask_columns = ["main", "wrist"]
    frame_indices = np.asarray([0, 3, 8], dtype=np.int64)
    masks = [
        np.arange(12, dtype=np.uint8).reshape(3, 2, 2),
        np.arange(12, 24, dtype=np.uint8).reshape(3, 2, 2),
    ]
    cache_path = tmp_path / "episode_000000.npz"
    dataset._write_role_mask_cache(cache_path, "source-a", frame_indices, masks)
    restored = dataset._read_role_mask_cache(cache_path, "source-a")
    assert restored is not None
    restored_indices, restored_masks = restored
    assert np.array_equal(restored_indices, frame_indices)
    assert all(np.array_equal(actual, expected) for actual, expected in zip(restored_masks, masks))
    assert dataset._read_role_mask_cache(cache_path, "source-b") is None


def test_collate_preserves_role_mask_shapes() -> None:
    sample = {
        "first_frame": torch.zeros(3, 448, 224),
        "video_frames": torch.zeros(52, 3, 448, 224),
        "first_role_mask": torch.ones(3, 448, 224),
        "role_mask_frames": torch.ones(52, 3, 448, 224),
    }
    batch = video_bridge_collate_fn([sample, sample])
    assert batch is not None
    assert batch["first_role_mask"].shape == (2, 3, 448, 224)
    assert batch["role_mask_frames"].shape == (2, 52, 3, 448, 224)
