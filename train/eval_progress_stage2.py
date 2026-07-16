#!/usr/bin/env python3
"""Evaluate cached single- or two-frame Progress trajectory alignment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover
    safe_load_file = None

sys.path.append(str(Path(__file__).parent.parent))

from data.progress import ProgressEpisodeCacheDataset
from models.progress_stage2 import (
    build_progress_model,
    build_progress_target_distribution,
)

if __package__:
    from .train_progress_stage2 import (
        forward_progress,
        load_frozen_vgm,
        model_config_from_yaml,
    )
else:
    from train_progress_stage2 import (
        forward_progress,
        load_frozen_vgm,
        model_config_from_yaml,
    )


logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "training_state.pt",
    ]
    candidates.extend(sorted(path.glob("pytorch_model_*.bin")))
    result = next((candidate for candidate in candidates if candidate.exists()), None)
    if result is None:
        raise FileNotFoundError(f"No Progress model file found under {path}")
    return result


def load_progress_checkpoint(model: torch.nn.Module, path: Path) -> None:
    file = checkpoint_file(path)
    if file.suffix == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("safetensors is required to load this checkpoint")
        state_dict = safe_load_file(str(file), device="cpu")
    else:
        state = torch.load(file, map_location="cpu")
        state_dict = state.get("model", state) if isinstance(state, dict) else state
    cleaned = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)


def rankdata(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values)
    sorted_ranks = torch.empty(
        values.numel(), device=values.device, dtype=torch.float32
    )
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.numel() < 2:
        return 0.0
    pred_rank = rankdata(prediction)
    target_rank = rankdata(target)
    pred_rank = pred_rank - pred_rank.mean()
    target_rank = target_rank - target_rank.mean()
    denominator = pred_rank.norm() * target_rank.norm()
    if denominator <= 0:
        return 0.0
    return float((pred_rank * target_rank).sum() / denominator)


def pearson(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Linear correlation between predicted and target absolute progress."""
    if prediction.numel() < 2:
        return 0.0
    prediction = prediction.float() - prediction.float().mean()
    target = target.float() - target.float().mean()
    denominator = prediction.norm() * target.norm()
    if denominator <= 0:
        return 0.0
    return float((prediction * target).sum() / denominator)


def ordering_accuracy(
    prediction: torch.Tensor, target: torch.Tensor, minimum_gap: float = 0.05
) -> float:
    target_delta = target[:, None] - target[None, :]
    prediction_delta = prediction[:, None] - prediction[None, :]
    valid = target_delta.abs() >= minimum_gap
    if not valid.any():
        return 0.0
    correct = (target_delta[valid].sign() == prediction_delta[valid].sign()).float()
    return float(correct.mean())


def episode_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alignment: torch.Tensor,
    target_sigma_bins: float = 2.0,
    *,
    previous_target: torch.Tensor | None = None,
    predicted_delta: torch.Tensor | None = None,
    direction_probabilities: torch.Tensor | None = None,
) -> Dict[str, float]:
    error = prediction - target
    monotonic_violations = (prediction[1:] + 1e-4 < prediction[:-1]).float()
    num_bins = alignment.shape[1]
    target_frame = target * float(num_bins - 1)
    matched_frame = alignment.argmax(dim=-1).float()
    target_distribution = build_progress_target_distribution(
        target,
        num_bins=num_bins,
        sigma_bins=target_sigma_bins,
    )
    alignment_cross_entropy = -(
        target_distribution * alignment.clamp_min(1e-12).log()
    ).sum(dim=-1)
    alignment_entropy = -(alignment * alignment.clamp_min(1e-12).log()).sum(dim=-1)
    absolute_error = error.abs()
    metrics = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.pow(2).mean().sqrt()),
        "absolute_error_p95": float(torch.quantile(absolute_error, 0.95)),
        "absolute_error_p99": float(torch.quantile(absolute_error, 0.99)),
        "max_absolute_error": float(absolute_error.max()),
        "voc_pearson": pearson(prediction, target),
        "spearman": spearman(prediction, target),
        "ordering_accuracy": ordering_accuracy(prediction, target),
        "monotonic_violation_rate": (
            float(monotonic_violations.mean()) if monotonic_violations.numel() else 0.0
        ),
        "start_error": float((prediction[0] - target[0]).abs()),
        "end_error": float((prediction[-1] - target[-1]).abs()),
        "expected_frame_mae": float(error.abs().mean() * float(num_bins - 1)),
        "matched_frame_mae": float((matched_frame - target_frame).abs().mean()),
        "alignment_cross_entropy": float(alignment_cross_entropy.mean()),
        "alignment_entropy": float(alignment_entropy.mean()),
        "matched_within_one": float(
            ((matched_frame - target_frame).abs() <= 1.0).float().mean()
        ),
        "matched_within_three": float(
            ((matched_frame - target_frame).abs() <= 3.0).float().mean()
        ),
        "matched_within_five": float(
            ((matched_frame - target_frame).abs() <= 5.0).float().mean()
        ),
    }
    if prediction.numel() > 1:
        predicted_step = prediction[1:] - prediction[:-1]
        target_step = target[1:] - target[:-1]
        step_error = (predicted_step - target_step).abs()
        metrics.update(
            {
                "sequential_delta_mae": float(step_error.mean()),
                "sequential_delta_error_p95": float(torch.quantile(step_error, 0.95)),
                "sequential_delta_error_p99": float(torch.quantile(step_error, 0.99)),
                "max_sequential_delta_error": float(step_error.max()),
                "predicted_jump_p95": float(torch.quantile(predicted_step.abs(), 0.95)),
                "predicted_jump_p99": float(torch.quantile(predicted_step.abs(), 0.99)),
                "max_predicted_jump": float(predicted_step.abs().max()),
                "backward_step_rate": float((predicted_step < -1e-4).float().mean()),
            }
        )
    else:
        metrics.update(
            {
                "sequential_delta_mae": 0.0,
                "sequential_delta_error_p95": 0.0,
                "sequential_delta_error_p99": 0.0,
                "max_sequential_delta_error": 0.0,
                "predicted_jump_p95": 0.0,
                "predicted_jump_p99": 0.0,
                "max_predicted_jump": 0.0,
                "backward_step_rate": 0.0,
            }
        )
    if previous_target is not None:
        if previous_target.shape != target.shape:
            raise ValueError("previous_target must match target shape")
        target_delta = target - previous_target
        metrics["pair_target_delta_mean"] = float(target_delta.mean())
        if predicted_delta is not None:
            if predicted_delta.shape != target.shape:
                raise ValueError("predicted_delta must match target shape")
            delta_error = (predicted_delta - target_delta).abs()
            metrics.update(
                {
                    "pair_delta_mae": float(delta_error.mean()),
                    "pair_delta_error_p95": float(torch.quantile(delta_error, 0.95)),
                    "pair_delta_error_p99": float(torch.quantile(delta_error, 0.99)),
                    "max_pair_delta_error": float(delta_error.max()),
                }
            )
        if direction_probabilities is not None:
            if direction_probabilities.shape != (target.shape[0], 3):
                raise ValueError("direction_probabilities must be [N,3]")
            predicted_direction = direction_probabilities.argmax(dim=-1) - 1
            target_direction = target_delta.sign().long()
            non_stay = target_direction != 0
            metrics["pair_direction_accuracy"] = float(
                (predicted_direction == target_direction).float().mean()
            )
            metrics["pair_wrong_direction_rate"] = (
                float(
                    (predicted_direction[non_stay] * target_direction[non_stay] < 0)
                    .float()
                    .mean()
                )
                if bool(non_stay.any())
                else 0.0
            )
    return metrics


def render_diagnostic(
    output_path: Path,
    episode_name: str,
    target: torch.Tensor,
    prediction: torch.Tensor,
    alignment: torch.Tensor,
    *,
    query_mode: str = "single_frame",
) -> None:
    if plt is None:
        return
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = torch.arange(target.numel()).numpy()
    axes[0].plot(x, target.numpy(), label="GT progress", linewidth=2.5, color="#1f77b4")
    axes[0].plot(
        x,
        prediction.numpy(),
        label="Predicted progress",
        linewidth=2.2,
        color="#d62728",
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Source frame order")
    axes[0].set_ylabel("Progress")
    axes[0].set_title(episode_name)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    image = axes[1].imshow(
        alignment.numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
    )
    target_frame = target.numpy() * float(alignment.shape[1] - 1)
    axes[1].plot(
        target_frame,
        torch.arange(target.numel()).numpy(),
        color="white",
        linewidth=1.5,
        linestyle="--",
        label="Source-time target slot",
    )
    axes[1].set_xlabel(f"Generated trajectory frame (0-{alignment.shape[1] - 1})")
    axes[1].set_ylabel("Current source frame order")
    query_label = "Single-frame" if query_mode == "single_frame" else "Two-frame"
    axes[1].set_title(f"{query_label} current-to-trajectory alignment probability")
    axes[1].legend(loc="upper left")
    figure.colorbar(image, ax=axes[1], label="Probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def _run_progress_queries(
    model: torch.nn.Module,
    model_config: Any,
    current_latents: torch.Tensor,
    trajectory_latent: torch.Tensor,
    trajectory_frame_latents: torch.Tensor,
    video_features: Any,
    query_batch_size: int,
    *,
    previous_latents: torch.Tensor | None = None,
    pair_time_gaps: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    collected: Dict[str, List[torch.Tensor]] = {}
    output_names = (
        "progress",
        "alignment_probabilities",
        "previous_progress",
        "previous_alignment_probabilities",
        "delta_progress",
        "direction_probabilities",
        "joint_alignment_probabilities",
    )
    for start in range(0, current_latents.shape[0], query_batch_size):
        end = min(start + query_batch_size, current_latents.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = forward_progress(
                model,
                model_config.fusion_mode,
                current_latents[start:end],
                trajectory_latent,
                trajectory_frame_latents,
                video_features,
                previous_latent=(
                    previous_latents[start:end]
                    if previous_latents is not None
                    else None
                ),
                pair_time_gap=(
                    pair_time_gaps[start:end] if pair_time_gaps is not None else None
                ),
            )
        for name in output_names:
            if name in outputs:
                collected.setdefault(name, []).append(outputs[name].float().cpu())
    return {name: torch.cat(values) for name, values in collected.items()}


def _balanced_pair_indices(
    count: int,
    evaluation_gap: int,
    maximum_gap: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build deterministic forward/stay/reverse pairs with equal class counts."""
    if count < 1:
        raise ValueError("Cannot evaluate an empty Progress episode")
    if count == 1:
        indices = torch.zeros(1, device=device, dtype=torch.long)
        return {
            "previous_indices": indices,
            "current_indices": indices,
            "pair_time_gaps": torch.zeros(1, device=device),
        }
    gap = min(int(evaluation_gap), count - 1)
    if gap < 1 or maximum_gap < gap:
        raise ValueError("Balanced evaluation gap must be in [1, maximum_gap]")
    later = torch.arange(gap, count, device=device)
    earlier = later - gap
    # The three blocks are forward, stay, and reverse with equal counts.
    previous_indices = torch.cat([earlier, later, later])
    current_indices = torch.cat([later, later, earlier])
    moving_gap = torch.full(
        (later.numel(),),
        float(gap) / float(maximum_gap),
        device=device,
    )
    pair_time_gaps = torch.cat([moving_gap, torch.zeros_like(moving_gap), moving_gap])
    return {
        "previous_indices": previous_indices,
        "current_indices": current_indices,
        "pair_time_gaps": pair_time_gaps,
    }


def _permute_previous_within_gap(
    previous_latents: torch.Tensor,
    pair_time_gaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically replace previous frames while preserving each gap group."""
    if pair_time_gaps.ndim != 1 or previous_latents.shape[0] != pair_time_gaps.shape[0]:
        raise ValueError(
            "previous_latents and pair_time_gaps must share batch dimension"
        )
    if not torch.isfinite(pair_time_gaps).all():
        raise ValueError("pair_time_gaps must be finite")

    shuffled = previous_latents.clone()
    valid = torch.zeros(
        pair_time_gaps.shape[0],
        device=pair_time_gaps.device,
        dtype=torch.bool,
    )
    for gap in torch.unique(pair_time_gaps):
        indices = torch.nonzero(pair_time_gaps == gap, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        # A half-group cyclic shift is deterministic, has no fixed points, and
        # replaces adjacent source frames with substantially more distant ones.
        shift = max(1, int(indices.numel()) // 2)
        donor_indices = indices.roll(shifts=shift)
        shuffled[indices] = previous_latents.index_select(0, donor_indices)
        valid[indices] = True
    return shuffled, valid


def _balanced_pair_metrics(
    outputs: Dict[str, torch.Tensor],
    previous_target: torch.Tensor,
    current_target: torch.Tensor,
) -> Dict[str, float]:
    prediction = outputs["progress"]
    target_delta = current_target - previous_target
    target_direction = target_delta.sign().long()
    metrics = {
        "balanced_pair_count": float(current_target.numel()),
        "balanced_pair_current_mae": float((prediction - current_target).abs().mean()),
    }
    for direction, name in ((-1, "reverse"), (0, "stay"), (1, "forward")):
        mask = target_direction == direction
        if bool(mask.any()):
            metrics[f"balanced_{name}_current_mae"] = float(
                (prediction[mask] - current_target[mask]).abs().mean()
            )
    if "previous_progress" in outputs:
        previous_prediction = outputs["previous_progress"]
        predicted_delta = outputs["delta_progress"]
        predicted_direction = outputs["direction_probabilities"].argmax(dim=-1) - 1
        metrics.update(
            {
                "balanced_pair_previous_mae": float(
                    (previous_prediction - previous_target).abs().mean()
                ),
                "balanced_pair_delta_mae": float(
                    (predicted_delta - target_delta).abs().mean()
                ),
                "balanced_pair_direction_accuracy": float(
                    (predicted_direction == target_direction).float().mean()
                ),
            }
        )
        non_stay = target_direction != 0
        metrics["balanced_pair_wrong_direction_rate"] = (
            float(
                (predicted_direction[non_stay] * target_direction[non_stay] < 0)
                .float()
                .mean()
            )
            if bool(non_stay.any())
            else 0.0
        )
        for direction, name in ((-1, "reverse"), (0, "stay"), (1, "forward")):
            mask = target_direction == direction
            if bool(mask.any()):
                metrics[f"balanced_{name}_direction_accuracy"] = float(
                    (predicted_direction[mask] == direction).float().mean()
                )
    return metrics


@torch.no_grad()
def evaluate_episode(
    model: torch.nn.Module,
    frozen_vgm: Any,
    model_config: Any,
    config: Any,
    episode: Dict[str, Any],
    query_batch_size: int,
    device: torch.device,
    *,
    run_pair_diagnostics: bool = True,
) -> Dict[str, Any]:
    trajectory_latent = episode["trajectory_latent"].unsqueeze(0).to(device)
    trajectory_frame_latents = (
        episode["trajectory_frame_latents"].unsqueeze(0).to(device)
    )
    current_latents = episode["current_latents"].to(device)
    target = episode["progress"].float().to(device)
    previous_latents = None
    previous_target = None
    previous_indices = None
    pair_time_gaps = None
    if model_config.query_mode != "single_frame":
        pair_config = config.get("pair_sampling", {})
        evaluation_gap = int(pair_config.get("evaluation_query_gap", 4))
        maximum_gap = int(pair_config.get("maximum_query_gap", 8))
        if evaluation_gap < 1 or maximum_gap < evaluation_gap:
            raise ValueError(
                "Evaluation pair gap must satisfy 1 <= evaluation <= maximum"
            )
        current_indices = torch.arange(current_latents.shape[0], device=device)
        previous_indices = (current_indices - evaluation_gap).clamp_min(0)
        previous_latents = current_latents.index_select(0, previous_indices)
        previous_target = target.index_select(0, previous_indices)
        pair_time_gaps = (current_indices - previous_indices).float()
        pair_time_gaps = (pair_time_gaps / float(maximum_gap)).clamp(0.0, 1.0)
    video_features = None
    if frozen_vgm is not None:
        video_features = frozen_vgm.extract_bridge_hidden_states(
            trajectory_latent=trajectory_latent,
            first_frame=episode["first_frame"]
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device),
            last_frame=episode["last_frame"].float().div(255.0).unsqueeze(0).to(device),
            language_embeddings=episode["language_embedding"].unsqueeze(0).to(device),
            feature_timestep=float(config.progress_model.get("feature_timestep", 0.0)),
        )

    outputs = _run_progress_queries(
        model,
        model_config,
        current_latents,
        trajectory_latent,
        trajectory_frame_latents,
        video_features,
        query_batch_size,
        previous_latents=previous_latents,
        pair_time_gaps=pair_time_gaps,
    )
    result = {
        "prediction": outputs["progress"],
        "alignment_probabilities": outputs["alignment_probabilities"],
    }
    if previous_target is not None:
        result["previous_target"] = previous_target.float().cpu()
        result["previous_frame_indices"] = (
            episode["frame_indices"].index_select(0, previous_indices.cpu()).long()
        )
        result["pair_time_gaps"] = pair_time_gaps.float().cpu()
    if "previous_progress" in outputs:
        result["previous_prediction"] = outputs["previous_progress"]
        result["predicted_delta"] = outputs["delta_progress"]
        result["direction_probabilities"] = outputs["direction_probabilities"]
    for name in (
        "previous_alignment_probabilities",
        "joint_alignment_probabilities",
    ):
        if name in outputs:
            result[name] = outputs[name]

    if model_config.query_mode != "single_frame" and run_pair_diagnostics:
        pair_config = config.get("pair_sampling", {})
        evaluation_gap = int(pair_config.get("evaluation_query_gap", 4))
        maximum_gap = int(pair_config.get("maximum_query_gap", 8))
        balanced = _balanced_pair_indices(
            current_latents.shape[0],
            evaluation_gap,
            maximum_gap,
            device,
        )
        balanced_previous = current_latents.index_select(
            0, balanced["previous_indices"]
        )
        balanced_current = current_latents.index_select(0, balanced["current_indices"])
        balanced_outputs = _run_progress_queries(
            model,
            model_config,
            balanced_current,
            trajectory_latent,
            trajectory_frame_latents,
            video_features,
            query_batch_size,
            previous_latents=balanced_previous,
            pair_time_gaps=balanced["pair_time_gaps"],
        )
        balanced_previous_target = target.index_select(
            0, balanced["previous_indices"]
        ).cpu()
        balanced_current_target = target.index_select(
            0, balanced["current_indices"]
        ).cpu()
        result["diagnostic_metrics"] = _balanced_pair_metrics(
            balanced_outputs,
            balanced_previous_target,
            balanced_current_target,
        )

        if previous_latents is not None and previous_latents.shape[0] > 1:
            shuffled_previous, shuffle_mask = _permute_previous_within_gap(
                previous_latents,
                pair_time_gaps,
            )
        else:
            shuffle_mask = None
        if shuffle_mask is not None and bool(shuffle_mask.any()):
            shuffled_outputs = _run_progress_queries(
                model,
                model_config,
                current_latents,
                trajectory_latent,
                trajectory_frame_latents,
                video_features,
                query_batch_size,
                previous_latents=shuffled_previous,
                pair_time_gaps=pair_time_gaps,
            )
            shuffle_mask_cpu = shuffle_mask.cpu()
            progress_change = (
                shuffled_outputs["progress"] - outputs["progress"]
            ).abs()[shuffle_mask_cpu]
            alignment_tv = (
                0.5
                * (
                    shuffled_outputs["alignment_probabilities"]
                    - outputs["alignment_probabilities"]
                )
                .abs()
                .sum(dim=-1)[shuffle_mask_cpu]
            )
            result["diagnostic_metrics"].update(
                {
                    "previous_shuffle_count": float(shuffle_mask.sum().item()),
                    "previous_shuffle_progress_change": float(progress_change.mean()),
                    "previous_shuffle_alignment_tv": float(alignment_tv.mean()),
                    "previous_shuffle_effect_rate_001": float(
                        (progress_change > 0.01).float().mean()
                    ),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, help="Override cache.cache_dir")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument(
        "--skip_pair_diagnostics",
        action="store_true",
        help="Skip balanced direction and previous-frame shuffle diagnostics",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    if not torch.cuda.is_available():
        raise RuntimeError("Progress evaluation requires CUDA")
    device = torch.device("cuda")

    config = OmegaConf.load(args.config)
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    model_config = model_config_from_yaml(config)
    model = build_progress_model(model_config).to(device)
    load_progress_checkpoint(model, Path(args.checkpoint))
    model.eval()
    frozen_vgm = load_frozen_vgm(config, device)
    dataset = ProgressEpisodeCacheDataset(
        config.cache.cache_dir,
        split=args.split,
        load_language_embedding=model_config.fusion_mode == "layerwise_wvm",
        expected_num_progress_bins=model_config.num_progress_bins,
        max_episodes=args.max_episodes,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    episode_outputs: List[Dict[str, Any]] = []
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    query_batch_size = int(config.training.query_batch_size)
    target_sigma_bins = float(config.loss.get("target_sigma_bins", 2.0))
    for index in range(len(dataset)):
        episode = dataset[index]
        evaluated = evaluate_episode(
            model,
            frozen_vgm,
            model_config,
            config,
            episode,
            query_batch_size,
            device,
            run_pair_diagnostics=not args.skip_pair_diagnostics,
        )
        prediction = evaluated["prediction"]
        alignment = evaluated["alignment_probabilities"]
        target = episode["progress"].cpu()
        metrics = episode_metrics(
            prediction,
            target,
            alignment,
            target_sigma_bins=target_sigma_bins,
            previous_target=evaluated.get("previous_target"),
            predicted_delta=evaluated.get("predicted_delta"),
            direction_probabilities=evaluated.get("direction_probabilities"),
        )
        metrics.update(evaluated.get("diagnostic_metrics", {}))
        row = {
            "episode_name": episode["episode_name"],
            "task_index": episode["task_index"],
            "num_queries": target.numel(),
            **metrics,
        }
        rows.append(row)
        output_payload = {
            "episode_name": str(episode["episode_name"]),
            "task_index": episode["task_index"],
            "total_frames": int(episode["total_frames"]),
            "frame_indices": episode["frame_indices"].cpu().long().contiguous(),
            "target": target.float().contiguous(),
            "prediction": prediction.float().contiguous(),
            "alignment_probabilities": alignment.float().contiguous(),
        }
        for name in (
            "previous_target",
            "previous_prediction",
            "predicted_delta",
            "direction_probabilities",
            "pair_time_gaps",
        ):
            if name in evaluated:
                output_payload[name] = evaluated[name].float().contiguous()
        if "previous_frame_indices" in evaluated:
            output_payload["previous_frame_indices"] = (
                evaluated["previous_frame_indices"].long().contiguous()
            )
        for name in (
            "previous_alignment_probabilities",
            "joint_alignment_probabilities",
        ):
            if name in evaluated:
                # Full 53x53 joint maps are retained for visual diagnosis. FP16
                # keeps a 100-episode benchmark practical without changing any
                # metric, which is computed above in float32.
                output_payload[name] = evaluated[name].half().contiguous()
        episode_outputs.append(output_payload)
        all_predictions.append(prediction)
        all_targets.append(target)
        if index < args.num_visualizations:
            safe_name = str(episode["episode_name"]).replace("/", "__")
            render_diagnostic(
                output_dir / "diagnostics" / f"{index:04d}_{safe_name}.png",
                str(episode["episode_name"]),
                target,
                prediction,
                alignment,
                query_mode=model_config.query_mode,
            )
        logger.info(
            "Evaluated %s: MAE=%.4f RMSE=%.4f ordering=%.4f",
            episode["episode_name"],
            metrics["mae"],
            metrics["rmse"],
            metrics["ordering_accuracy"],
        )

    if not rows:
        raise RuntimeError("Progress evaluation dataset is empty")
    prediction = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    error = prediction - target

    metadata_keys = {"episode_name", "task_index", "num_queries"}

    def mean_metrics(metric_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = sorted(
            {
                key
                for metric_row in metric_rows
                for key in metric_row
                if key not in metadata_keys
            }
        )
        return {
            key: sum(float(row[key]) for row in metric_rows if key in row)
            / sum(1 for row in metric_rows if key in row)
            for key in keys
        }

    episode_mean = mean_metrics(rows)
    task_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        task_groups.setdefault(str(row["task_index"]), []).append(row)
    task_metrics = {
        task_index: mean_metrics(task_rows)
        for task_index, task_rows in task_groups.items()
    }
    task_metric_rows = [dict(metrics) for metrics in task_metrics.values()]
    task_macro = mean_metrics(task_metric_rows)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "query_mode": model_config.query_mode,
        "pair_diagnostics": not args.skip_pair_diagnostics,
        "episodes": len(rows),
        "queries": target.numel(),
        "tasks": len(task_metrics),
        "task_indices": sorted(task_metrics, key=lambda value: int(value)),
        "query_weighted": {
            "mae": float(error.abs().mean()),
            "rmse": float(error.pow(2).mean().sqrt()),
        },
        # Sequence metrics must be computed within each episode. Concatenating
        # episodes creates artificial end-to-start transitions.
        "episode_mean": episode_mean,
        "task_macro": task_macro,
        "per_task": task_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        metric_fields = sorted(
            {key for row in rows for key in row if key not in metadata_keys}
        )
        writer = csv.DictWriter(
            file,
            fieldnames=["episode_name", "task_index", "num_queries", *metric_fields],
        )
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "schema_version": 1,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "episodes": episode_outputs,
        },
        output_dir / "episode_outputs.pt",
    )
    logger.info("Progress evaluation complete: %s", output_dir)


if __name__ == "__main__":
    main()
