#!/usr/bin/env python3
"""Replay real and generated PushT pusher paths in the official simulator.

The VGM predicts observations rather than actions. This diagnostic first fits a
small inverse-dynamics map from real pusher positions to absolute PushT actions,
calibrates it on held-out real trajectories, and only then replays generated
17-frame pusher tracks. Generated tracks with missing endpoints or long missing
runs are not silently repaired and are reported as unrecoverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORLD_SIZE = 512.0
CONDITIONED_GOAL_POSITION_MAX = 20.0
CONDITIONED_GOAL_ANGLE_MAX = np.deg2rad(15.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--official_repo_root", type=Path, required=True)
    parser.add_argument("--dependency_root", type=Path)
    parser.add_argument("--max_missing_run", type=int, default=2)
    parser.add_argument("--render_size", type=int, default=256)
    return parser.parse_args()


def install_official_imports(
    official_repo_root: Path, dependency_root: Path | None
) -> Any:
    if dependency_root is not None:
        sys.path.insert(0, str(dependency_root.expanduser().resolve()))
    sys.path.insert(0, str(official_repo_root.expanduser().resolve()))
    import gymnasium

    sys.modules.setdefault("gym", gymnasium)
    if "skimage" not in sys.modules:
        skimage = types.ModuleType("skimage")
        transform = types.ModuleType("skimage.transform")
        skimage.transform = transform
        sys.modules["skimage"] = skimage
        sys.modules["skimage.transform"] = transform
    from diffusion_policy.env.pusht.pusht_env import PushTEnv

    return PushTEnv


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def inverse_features(pusher_xy: np.ndarray) -> np.ndarray:
    pusher = np.asarray(pusher_xy, dtype=np.float64)
    previous = np.vstack([pusher[0], pusher[:-1]])
    following = np.vstack([pusher[1:], pusher[-1]])
    return np.stack(
        [
            np.ones_like(pusher),
            previous,
            pusher,
            following,
        ],
        axis=-1,
    )


def fit_inverse_dynamics(
    dataset_dir: Path, train_rows: Sequence[dict[str, Any]]
) -> np.ndarray:
    features = []
    targets = []
    for row in train_rows:
        payload = np.load(dataset_dir / row["metadata_path"])
        pusher = payload["state"].astype(np.float64)[:, :2]
        action = payload["action"].astype(np.float64)
        current_features = inverse_features(pusher)[1:-1]
        features.append(current_features.reshape(-1, 4))
        targets.append(action[1:-1].reshape(-1))
    coefficients, _, _, _ = np.linalg.lstsq(
        np.concatenate(features), np.concatenate(targets), rcond=None
    )
    return coefficients


def predict_actions(pusher_xy: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    actions = inverse_features(pusher_xy) @ coefficients
    return np.clip(actions, 0.0, WORLD_SIZE)


def inverse_prediction_metrics(
    dataset_dir: Path,
    rows: Sequence[dict[str, Any]],
    coefficients: np.ndarray,
) -> dict[str, float]:
    error = []
    for row in rows:
        payload = np.load(dataset_dir / row["metadata_path"])
        predicted = predict_actions(payload["state"][:, :2], coefficients)
        error.append(predicted[1:-1] - payload["action"][1:-1])
    delta = np.concatenate(error)
    norm = np.linalg.norm(delta, axis=-1)
    return {
        "coordinate_rmse_pixels": float(np.sqrt(np.mean(delta**2))),
        "coordinate_mae_pixels": float(np.mean(np.abs(delta))),
        "vector_error_mean_pixels": float(norm.mean()),
        "vector_error_p95_pixels": float(np.quantile(norm, 0.95)),
    }


def angle_error(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(left) - np.asarray(right) + np.pi) % (2 * np.pi) - np.pi)


def max_false_run(valid: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in valid:
        if value:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def recover_waypoints(
    path: np.ndarray, max_missing_run: int
) -> tuple[np.ndarray | None, str]:
    path = np.asarray(path, dtype=np.float64)
    valid = np.isfinite(path).all(axis=-1)
    if not valid[0] or not valid[-1]:
        return None, "missing_endpoint"
    longest_gap = max_false_run(valid)
    if longest_gap > max_missing_run:
        return None, f"missing_run_{longest_gap}_gt_{max_missing_run}"
    source = np.flatnonzero(valid)
    target = np.arange(len(path))
    recovered = np.column_stack(
        [np.interp(target, source, path[valid, dimension]) for dimension in range(2)]
    )
    return recovered, (
        "complete" if valid.all() else f"interpolated_max_gap_{longest_gap}"
    )


def densify_waypoints(
    waypoints: np.ndarray, frame_indices: np.ndarray, length: int
) -> np.ndarray:
    target = np.arange(length, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(target, frame_indices.astype(np.float64), waypoints[:, dimension])
            for dimension in range(2)
        ]
    )


def replay_actions(
    env_class: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
    capture_indices: Sequence[int],
    render_size: int,
) -> dict[str, Any]:
    env = env_class(
        legacy=False,
        reset_to_state=np.asarray(initial_state, dtype=np.float64),
        render_action=False,
        render_size=render_size,
    )
    observation = env.reset()
    _, initial_reward, initial_done, _ = env.step(None)
    states = [np.asarray(observation, dtype=np.float64)]
    rewards = [float(initial_reward)]
    dones = [bool(initial_done)]
    capture = {int(capture_indices[0]): env.render(mode="rgb_array")}
    capture_set = set(int(value) for value in capture_indices)
    for time_index in range(len(actions) - 1):
        observation, reward, done, _ = env.step(
            np.asarray(actions[time_index], dtype=np.float64)
        )
        state_index = time_index + 1
        states.append(np.asarray(observation, dtype=np.float64))
        rewards.append(float(reward))
        dones.append(bool(done))
        if state_index in capture_set:
            capture[state_index] = env.render(mode="rgb_array")
    env.close()
    frames = np.stack([capture[int(index)] for index in capture_indices])
    return {
        "states": np.stack(states),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "dones": np.asarray(dones, dtype=bool),
        "frames": frames,
    }


def replay_metrics(
    replay: dict[str, Any],
    target_state: np.ndarray,
    reference_state: np.ndarray | None = None,
) -> dict[str, Any]:
    states = replay["states"]
    final_position_error = float(np.linalg.norm(states[-1, 2:4] - target_state[2:4]))
    final_angle_error = float(angle_error(states[-1, 4], target_state[4]))
    result = {
        "max_official_reward": float(replay["rewards"].max()),
        "final_official_reward": float(replay["rewards"][-1]),
        "official_success": bool(replay["dones"].any()),
        "conditioned_goal_position_error_pixels": final_position_error,
        "conditioned_goal_angle_error_radians": final_angle_error,
        "conditioned_goal_reached": bool(
            final_position_error <= CONDITIONED_GOAL_POSITION_MAX
            and final_angle_error <= CONDITIONED_GOAL_ANGLE_MAX
        ),
    }
    if reference_state is not None:
        position_delta = states[:, :4] - reference_state[:, :4]
        result["state_position_rmse_pixels"] = float(
            np.sqrt(np.mean(position_delta**2))
        )
        result["state_angle_mae_radians"] = float(
            np.mean(angle_error(states[:, 4], reference_state[:, 4]))
        )
    return result


def generated_consistency_metrics(
    replay: dict[str, Any],
    frame_indices: np.ndarray,
    generated_pusher: np.ndarray,
    generated_block: np.ndarray,
) -> dict[str, float]:
    sampled_state = replay["states"][frame_indices]

    def rms(actual: np.ndarray, expected: np.ndarray) -> float:
        finite = np.isfinite(expected).all(axis=-1)
        if not finite.any():
            return float("nan")
        return float(
            np.sqrt(np.mean((actual[finite] / WORLD_SIZE - expected[finite]) ** 2))
        )

    return {
        "physical_pusher_vs_generated_rms": rms(sampled_state[:, :2], generated_pusher),
        "physical_block_vs_generated_rms": rms(sampled_state[:, 2:4], generated_block),
    }


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def save_replay_sheet(
    path: Path,
    episode_name: str,
    gt_frames: np.ndarray,
    sparse_replay: dict[str, Any],
    generated_rows: Sequence[dict[str, Any]],
    generated_replays: dict[int, dict[str, Any]],
) -> None:
    tile = 96
    frame_count = 17
    row_label = 690
    header = 58
    frame_label = 22
    rows: list[tuple[str, np.ndarray | None]] = [
        ("Recorded ground truth", gt_frames),
        ("Physics replay: real 17-waypoint path", sparse_replay["frames"]),
    ]
    for row in sorted(generated_rows, key=lambda value: int(value["seed"])):
        seed = int(row["seed"])
        replay = generated_replays.get(seed)
        if replay is None:
            label = f"Seed {seed}: unrecoverable ({row['track_status']})"
            rows.append((label, None))
        else:
            label = (
                f"Seed {seed} | reward={row['max_official_reward']:.3f} | "
                f"goal error={row['conditioned_goal_position_error_pixels']:.1f}px | "
                f"success={row['official_success']}"
            )
            rows.append((label, replay["frames"]))
    canvas = Image.new(
        "RGB",
        (row_label + frame_count * tile, header + len(rows) * (tile + frame_label)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"{episode_name}: 17-frame physical replay calibration",
        fill=(16, 24, 32),
        font=load_font(25),
    )
    label_font = load_font(18)
    frame_font = load_font(14)
    for row_index, (label, frames) in enumerate(rows):
        y = header + row_index * (tile + frame_label)
        draw.text((12, y + tile // 2), label, fill=(16, 24, 32), font=label_font)
        for frame_index in range(frame_count):
            x = row_label + frame_index * tile
            if frames is None:
                frame = Image.new("RGB", (tile, tile), (232, 232, 232))
                frame_draw = ImageDraw.Draw(frame)
                frame_draw.line(
                    (12, 12, tile - 12, tile - 12), fill=(150, 50, 50), width=4
                )
                frame_draw.line(
                    (tile - 12, 12, 12, tile - 12), fill=(150, 50, 50), width=4
                )
            else:
                frame = Image.fromarray(frames[frame_index]).resize(
                    (tile, tile), Image.Resampling.LANCZOS
                )
            canvas.paste(frame, (x, y + frame_label))
            draw.text(
                (x + 4, y + 2), f"{frame_index:02d}", fill=(24, 32, 40), font=frame_font
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def plot_replay_summary(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    selected = [row for row in rows if row["scope"] == "selected_condition"]
    conditions = sorted({int(row["condition_number"]) for row in selected})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = {
        "recorded_actions": "#444444",
        "inverse_dense_gt": "#55a868",
        "inverse_sparse17_gt": "#4c72b0",
        "inverse_generated17": "#c44e52",
    }
    legend_methods: set[str] = set()
    for condition in conditions:
        current = [
            row
            for row in selected
            if int(row["condition_number"]) == condition
            and row.get("replay_available", True)
        ]
        for row in current:
            method = str(row["method"])
            x = (
                condition
                + {
                    "recorded_actions": -0.24,
                    "inverse_dense_gt": -0.08,
                    "inverse_sparse17_gt": 0.08,
                    "inverse_generated17": 0.24,
                }[method]
            )
            marker = "x" if method == "inverse_generated17" else "o"
            axes[0].scatter(
                x,
                float(row["max_official_reward"]),
                color=colors[method],
                marker=marker,
                s=55,
                alpha=0.8,
                label=method if method not in legend_methods else None,
            )
            legend_methods.add(method)
            axes[1].scatter(
                x,
                float(row["conditioned_goal_position_error_pixels"]),
                color=colors[method],
                marker=marker,
                s=55,
                alpha=0.8,
            )
    axes[0].set_title("Official PushT maximum reward")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(fontsize=9)
    axes[1].set_title("Final block distance to conditioned goal")
    axes[1].axhline(
        CONDITIONED_GOAL_POSITION_MAX, color="black", linestyle="--", linewidth=1
    )
    axes[1].set_ylabel("pixels")
    for axis in axes:
        axis.set_xticks(conditions, [f"C{condition:02d}" for condition in conditions])
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Official PushT physics replay: real-path controls versus VGM paths",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = read_json(evaluation_dir / "run_summary.json")
    if int(run_summary["num_inference_steps"]) != 50:
        raise ValueError(
            "physical replay is defined on the formal 50-step generation run"
        )
    env_class = install_official_imports(args.official_repo_root, args.dependency_root)

    train_rows = read_manifest(dataset_dir / "train" / "manifest.jsonl")
    test_rows = read_manifest(dataset_dir / "test" / "manifest.jsonl")
    test_by_name = {row["episode_name"]: row for row in test_rows}
    coefficients = fit_inverse_dynamics(dataset_dir, train_rows)
    inverse_metrics = inverse_prediction_metrics(dataset_dir, test_rows, coefficients)

    # Calibrate the full replay chain on all held-out real episodes.
    calibration_rows = []
    for row in test_rows:
        payload = np.load(dataset_dir / row["metadata_path"])
        state = payload["state"].astype(np.float64)
        length = len(state)
        sparse_indices = np.rint(np.linspace(0, length - 1, 17)).astype(np.int64)
        methods = {
            "recorded_actions": payload["action"].astype(np.float64),
            "inverse_dense_gt": predict_actions(state[:, :2], coefficients),
            "inverse_sparse17_gt": predict_actions(
                densify_waypoints(state[sparse_indices, :2], sparse_indices, length),
                coefficients,
            ),
        }
        for method, actions in methods.items():
            replay = replay_actions(
                env_class,
                state[0],
                actions,
                sparse_indices,
                args.render_size,
            )
            calibration_rows.append(
                {
                    "scope": "all_test_calibration",
                    "episode_name": row["episode_name"],
                    "method": method,
                    **replay_metrics(replay, state[-1], reference_state=state),
                }
            )

    windows = read_json(evaluation_dir / "windows.json")
    metric_rows = read_json(evaluation_dir / "sample_metrics.json")
    metrics_by_condition: dict[int, list[dict[str, Any]]] = {}
    for row in metric_rows:
        metrics_by_condition.setdefault(int(row["condition_number"]), []).append(row)

    selected_rows = []
    for window in windows:
        condition = int(window["condition_number"])
        episode_name = str(window["episode_name"])
        payload = np.load(dataset_dir / test_by_name[episode_name]["metadata_path"])
        state = payload["state"].astype(np.float64)
        frame_indices = np.asarray(window["frame_indices"], dtype=np.int64)
        if len(frame_indices) != 17:
            raise ValueError(
                f"expected 17 frame indices for {episode_name}, got {len(frame_indices)}"
            )
        directory = next(evaluation_dir.glob(f"condition_{condition:02d}_*"))
        first_tracks = np.load(
            directory
            / f"seed_{int(metrics_by_condition[condition][0]['seed']):04d}"
            / "trajectory_tracks.npz"
        )

        control_replays: dict[str, dict[str, Any]] = {}
        control_actions = {
            "recorded_actions": payload["action"].astype(np.float64),
            "inverse_dense_gt": predict_actions(state[:, :2], coefficients),
            "inverse_sparse17_gt": predict_actions(
                densify_waypoints(state[frame_indices, :2], frame_indices, len(state)),
                coefficients,
            ),
        }
        for method, actions in control_actions.items():
            replay = replay_actions(
                env_class,
                state[0],
                actions,
                frame_indices,
                args.render_size,
            )
            control_replays[method] = replay
            selected_rows.append(
                {
                    "scope": "selected_condition",
                    "condition_number": condition,
                    "episode_name": episode_name,
                    "seed": "",
                    "method": method,
                    "track_status": "real_control",
                    **replay_metrics(replay, state[-1], reference_state=state),
                }
            )

        generated_replays: dict[int, dict[str, Any]] = {}
        generated_rows = []
        for metric_row in sorted(
            metrics_by_condition[condition], key=lambda value: int(value["seed"])
        ):
            seed = int(metric_row["seed"])
            tracks = np.load(directory / f"seed_{seed:04d}" / "trajectory_tracks.npz")
            recovered, status = recover_waypoints(
                tracks["pred_pusher_xy"], args.max_missing_run
            )
            base = {
                "scope": "selected_condition",
                "condition_number": condition,
                "episode_name": episode_name,
                "seed": seed,
                "method": "inverse_generated17",
                "track_status": status,
                "generated_pusher_detection_rate": float(
                    metric_row["pusher_detection_rate"]
                ),
            }
            if recovered is None:
                unavailable_row = {**base, "replay_available": False}
                generated_rows.append(unavailable_row)
                selected_rows.append(unavailable_row)
                continue
            dense_pusher = densify_waypoints(
                recovered * WORLD_SIZE,
                frame_indices,
                len(state),
            )
            replay = replay_actions(
                env_class,
                state[0],
                predict_actions(dense_pusher, coefficients),
                frame_indices,
                args.render_size,
            )
            generated_replays[seed] = replay
            row = {
                **base,
                "replay_available": True,
                **replay_metrics(replay, state[-1]),
                **generated_consistency_metrics(
                    replay,
                    frame_indices,
                    tracks["pred_pusher_xy"],
                    tracks["pred_block_xy"],
                ),
            }
            generated_rows.append(row)
            selected_rows.append(row)

        save_replay_sheet(
            output_dir
            / "by_condition"
            / f"condition_{condition:02d}_physics_replay_17f.png",
            episode_name,
            first_tracks["gt_frames"],
            control_replays["inverse_sparse17_gt"],
            generated_rows,
            generated_replays,
        )

    rows = calibration_rows + selected_rows
    write_csv(output_dir / "physical_replay_metrics.csv", rows)
    (output_dir / "physical_replay_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    plot_replay_summary(selected_rows, output_dir / "physical_replay_summary.png")

    def method_summary(method: str) -> dict[str, Any]:
        values = [row for row in calibration_rows if row["method"] == method]
        return {
            "episodes": len(values),
            "official_success_rate": float(
                np.mean([row["official_success"] for row in values])
            ),
            "conditioned_goal_reached_rate": float(
                np.mean([row["conditioned_goal_reached"] for row in values])
            ),
            "mean_max_official_reward": float(
                np.mean([row["max_official_reward"] for row in values])
            ),
            "mean_conditioned_goal_position_error_pixels": float(
                np.mean(
                    [row["conditioned_goal_position_error_pixels"] for row in values]
                )
            ),
        }

    generated_available = [
        row
        for row in selected_rows
        if row["method"] == "inverse_generated17" and row.get("replay_available", True)
    ]
    summary = {
        "official_environment": str(args.official_repo_root),
        "formal_num_inference_steps": 50,
        "inverse_dynamics_coefficients": coefficients.tolist(),
        "heldout_inverse_action_metrics": inverse_metrics,
        "all_test_real_replay_calibration": {
            method: method_summary(method)
            for method in (
                "recorded_actions",
                "inverse_dense_gt",
                "inverse_sparse17_gt",
            )
        },
        "generated": {
            "samples": len(metric_rows),
            "strict_complete_track_samples": int(
                sum(float(row["pusher_detection_rate"]) == 1.0 for row in metric_rows)
            ),
            "relaxed_recoverable_samples": len(generated_available),
            "relaxed_recoverable_rate": len(generated_available) / len(metric_rows),
            "official_success_rate_among_recoverable": (
                float(np.mean([row["official_success"] for row in generated_available]))
                if generated_available
                else float("nan")
            ),
            "conditioned_goal_reached_rate_among_recoverable": (
                float(
                    np.mean(
                        [row["conditioned_goal_reached"] for row in generated_available]
                    )
                )
                if generated_available
                else float("nan")
            ),
            "mean_max_official_reward_among_recoverable": (
                float(
                    np.mean([row["max_official_reward"] for row in generated_available])
                )
                if generated_available
                else float("nan")
            ),
        },
        "conditioned_goal_thresholds": {
            "position_pixels": CONDITIONED_GOAL_POSITION_MAX,
            "angle_radians": float(CONDITIONED_GOAL_ANGLE_MAX),
        },
        "generated_recovery_policy": {
            "endpoints_required": True,
            "max_missing_run": args.max_missing_run,
            "strict_primary_interpretation": "No missing pusher detections across all 17 frames.",
            "relaxed_interpretation": "Diagnostic only; short gaps are linearly interpolated.",
        },
    }
    (output_dir / "physical_replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
