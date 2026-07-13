from pathlib import Path
import random
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset


def sampling_dataset() -> VideoBridgeDataset:
    dataset = object.__new__(VideoBridgeDataset)
    dataset.num_video_frames = 52
    return dataset


def test_full_episode_jitter_keeps_endpoints_and_strict_order() -> None:
    dataset = sampling_dataset()
    for total_frames in range(53, 401):
        for seed in range(20):
            random.seed(seed)
            indices = dataset._uniform_frame_indices(total_frames, jitter=True)
            assert len(indices) == 53
            assert indices[0] == 0
            assert indices[-1] == total_frames - 1
            assert all(left < right for left, right in zip(indices, indices[1:]))


def test_full_episode_eval_sampling_is_deterministic() -> None:
    dataset = sampling_dataset()
    first = dataset._uniform_frame_indices(230, jitter=False)
    second = dataset._uniform_frame_indices(230, jitter=False)

    assert first == second
    assert first[0] == 0
    assert first[-1] == 229
    assert len(first) == 53
