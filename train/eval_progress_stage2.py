#!/usr/bin/env python3
"""Evaluate cached single-frame Progress and render trajectory-alignment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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
    """Value-Order Correlation used by robot progress-model evaluations."""
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
    alignment_entropy = -(
        alignment * alignment.clamp_min(1e-12).log()
    ).sum(dim=-1)
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.pow(2).mean().sqrt()),
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


def render_diagnostic(
    output_path: Path,
    episode_name: str,
    target: torch.Tensor,
    prediction: torch.Tensor,
    alignment: torch.Tensor,
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
        label="GT matched frame",
    )
    axes[1].set_xlabel(f"Generated trajectory frame (0-{alignment.shape[1] - 1})")
    axes[1].set_ylabel("Current source frame order")
    axes[1].set_title("Single-frame to trajectory alignment probability")
    axes[1].legend(loc="upper left")
    figure.colorbar(image, ax=axes[1], label="Probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def evaluate_episode(
    model: torch.nn.Module,
    frozen_vgm: Any,
    model_config: Any,
    config: Any,
    episode: Dict[str, Any],
    query_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory_latent = episode["trajectory_latent"].unsqueeze(0).to(device)
    trajectory_frame_latents = (
        episode["trajectory_frame_latents"].unsqueeze(0).to(device)
    )
    current_latents = episode["current_latents"].to(device)
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

    predictions: List[torch.Tensor] = []
    alignments: List[torch.Tensor] = []
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
            )
        predictions.append(outputs["progress"].float().cpu())
        alignments.append(outputs["alignment_probabilities"].float().cpu())
    return torch.cat(predictions), torch.cat(alignments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, help="Override cache.cache_dir")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_visualizations", type=int, default=16)
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
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    query_batch_size = int(config.training.query_batch_size)
    target_sigma_bins = float(config.loss.get("target_sigma_bins", 2.0))
    for index in range(len(dataset)):
        episode = dataset[index]
        prediction, alignment = evaluate_episode(
            model,
            frozen_vgm,
            model_config,
            config,
            episode,
            query_batch_size,
            device,
        )
        target = episode["progress"].cpu()
        metrics = episode_metrics(
            prediction,
            target,
            alignment,
            target_sigma_bins=target_sigma_bins,
        )
        row = {
            "episode_name": episode["episode_name"],
            "task_index": episode["task_index"],
            "num_queries": target.numel(),
            **metrics,
        }
        rows.append(row)
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
            )
        logger.info(
            "Evaluated %s: MAE=%.4f RMSE=%.4f ordering=%.4f",
            episode["episode_name"],
            metrics["mae"],
            metrics["rmse"],
            metrics["ordering_accuracy"],
        )

    prediction = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    error = prediction - target
    episode_mean = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
        if key not in {"episode_name", "task_index", "num_queries"}
    }
    task_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        task_groups.setdefault(str(row["task_index"]), []).append(row)
    task_metrics = {
        task_index: {
            key: sum(float(row[key]) for row in task_rows) / len(task_rows)
            for key in task_rows[0]
            if key not in {"episode_name", "task_index", "num_queries"}
        }
        for task_index, task_rows in task_groups.items()
    }
    task_macro = {
        key: sum(metrics[key] for metrics in task_metrics.values()) / len(task_metrics)
        for key in next(iter(task_metrics.values()))
    }
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "episodes": len(rows),
        "queries": target.numel(),
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
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Progress evaluation complete: %s", output_dir)


if __name__ == "__main__":
    main()
