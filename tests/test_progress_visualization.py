from __future__ import annotations

import numpy as np

from scripts.render_progress_episode_videos import event_summary, render_frame


def normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / values.sum(axis=-1, keepdims=True)


def test_pair_aware_render_frame_contains_nonblank_joint_diagnostic() -> None:
    queries = 7
    slots = 53
    target = np.linspace(0.0, 1.0, queries, dtype=np.float32)
    prediction = np.clip(target + 0.01, 0.0, 1.0)
    alignment = normalized_rows(
        np.exp(
            -0.5
            * (
                np.arange(slots, dtype=np.float32)[None, :]
                - target[:, None] * (slots - 1)
            )
            ** 2
            / 4.0
        )
    )
    joint = alignment[:, :, None] * alignment[:, None, :]
    joint = joint / joint.sum(axis=(1, 2), keepdims=True)
    image = np.full((72, 72, 3), 127, dtype=np.uint8)
    generated = np.stack(
        [np.full_like(image, slot * 4, dtype=np.uint8) for slot in range(slots)]
    )
    frame = render_frame(
        width=1600,
        height=900,
        model="B5",
        task_text="place the object in the drawer",
        episode_name="robo_dopamine_bench/libero_data/episode_000",
        query_index=3,
        frame_index=12,
        previous_frame_index=8,
        previous_high_frame=image,
        previous_wrist_frame=image,
        high_frame=image,
        wrist_frame=image,
        generated_frames=generated,
        target=target,
        prediction=prediction,
        alignment=alignment,
        heatmap=np.repeat(alignment[:, :, None], 3, axis=2).astype(np.uint8),
        argmax_slots=alignment.argmax(axis=1),
        previous_target=np.roll(target, 1),
        previous_prediction=np.roll(prediction, 1),
        predicted_delta=np.full(queries, 0.1, dtype=np.float32),
        direction_probabilities=np.tile(
            np.asarray([[0.05, 0.10, 0.85]], dtype=np.float32),
            (queries, 1),
        ),
        pair_time_gaps=np.full(queries, 0.5, dtype=np.float32),
        joint_probabilities=joint,
    )
    assert frame.shape == (900, 1600, 3)
    assert frame.dtype == np.uint8
    assert int(frame.max()) > int(frame.min())


def test_event_summary_marks_catastrophic_slot_switch_and_pair_metrics() -> None:
    target = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    prediction = np.asarray([0.0, 0.3, 0.9, 0.2], dtype=np.float32)
    alignment = np.zeros((4, 53), dtype=np.float32)
    alignment[np.arange(4), [0, 16, 50, 8]] = 1.0
    previous_target = np.asarray([0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
    predicted_delta = prediction - np.asarray([0.0, 0.0, 0.3, 0.9], dtype=np.float32)
    direction = np.tile(np.asarray([[0.05, 0.05, 0.90]], dtype=np.float32), (4, 1))
    summary = event_summary(
        "episode_045",
        "close the microwave",
        "B5",
        np.arange(4),
        target,
        prediction,
        alignment,
        previous_target=previous_target,
        previous_prediction=np.asarray([0.0, 0.0, 0.3, 0.9], dtype=np.float32),
        predicted_delta=predicted_delta,
        direction_probabilities=direction,
    )
    assert summary["catastrophic"] is True
    assert summary["max_absolute_slot_jump"] == 42
    assert summary["max_absolute_jump"] > 0.6
    assert "pair_delta_mae" in summary
    assert "pair_direction_accuracy" in summary
