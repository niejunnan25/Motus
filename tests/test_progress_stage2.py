import json

import pytest
import torch

from data.progress.progress_cache_dataset import (
    SCHEMA_VERSION,
    ProgressEpisodeCacheDataset,
    progress_episode_collate_fn,
)
from models.progress_stage2 import (
    ProgressStage2Config,
    build_progress_target_distribution,
    build_progress_model,
    compute_progress_loss,
)


def small_config(fusion_mode: str, num_layers: int) -> ProgressStage2Config:
    return ProgressStage2Config(
        fusion_mode=fusion_mode,
        num_progress_bins=53,
        latent_channels=4,
        hidden_dim=32,
        num_heads=4,
        num_layers=num_layers,
        ffn_multiplier=2,
        patch_size=(1, 2, 2),
        frame_num_tokens=4,
        video_hidden_dim=48,
        dropout=0.0,
    )


def test_serial_progress_forward_and_loss_backward():
    model = build_progress_model(small_config("serial_latent", num_layers=2))
    current = torch.randn(5, 4, 1, 8, 8)
    trajectory_frames = torch.randn(1, 53, 4, 1, 8, 8)
    target = torch.linspace(0.0, 1.0, 5)

    outputs = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
    )
    assert outputs["progress"].shape == (5,)
    assert outputs["expected_frame"].shape == (5,)
    assert outputs["matched_frame"].shape == (5,)
    assert outputs["alignment_probabilities"].shape == (5, 53)
    torch.testing.assert_close(
        outputs["alignment_probabilities"].sum(dim=-1),
        torch.ones(5),
    )
    losses = compute_progress_loss(outputs, target)
    losses["total_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_layerwise_progress_is_asymmetric_and_uses_every_layer():
    model = build_progress_model(small_config("layerwise_wvm", num_layers=3))
    current = torch.randn(4, 4, 1, 8, 8)
    trajectory_frames = torch.randn(1, 53, 4, 1, 8, 8)
    video_hidden = [torch.randn(1, 12, 48) for _ in range(3)]
    originals = [item.clone() for item in video_hidden]

    outputs = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        video_hidden_states=video_hidden,
        video_grid_size=(3, 2, 2),
    )
    assert outputs["progress"].shape == (4,)
    assert outputs["alignment_probabilities"].shape == (4, 53)
    compute_progress_loss(outputs, torch.linspace(0.0, 1.0, 4))["total_loss"].backward()

    for before, after in zip(originals, video_hidden):
        torch.testing.assert_close(before, after)
        assert after.grad is None
    assert all(block.video_projection.weight.grad is not None for block in model.blocks)
    assert model.frame_encoder.tokenizer.proj.weight.grad is not None


def test_progress_cache_dataset_preserves_whole_episode(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    language_dir = tmp_path / "language"
    language_dir.mkdir()
    torch.save(torch.randn(5, 16), language_dir / "task_000001.pt")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_name": "task/episode_000001",
        "task_index": 1,
        "total_frames": 7,
        "num_progress_bins": 53,
        "trajectory_latent": torch.randn(4, 3, 8, 8),
        "trajectory_frame_latents": torch.randn(53, 4, 1, 8, 8),
        "current_latents": torch.randn(7, 4, 1, 8, 8),
        "progress": torch.linspace(0.0, 1.0, 7),
        "frame_indices": torch.arange(7),
        "first_frame": torch.zeros(3, 16, 16, dtype=torch.uint8),
        "last_frame": torch.zeros(3, 16, 16, dtype=torch.uint8),
        "language_file": "language/task_000001.pt",
    }
    torch.save(payload, episode_dir / "episode.pt")
    manifest_entry = {
        "schema_version": SCHEMA_VERSION,
        "episode_name": payload["episode_name"],
        "task_index": 1,
        "split": "train",
        "cache_file": "episodes/episode.pt",
        "language_file": payload["language_file"],
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest_entry) + "\n")

    dataset = ProgressEpisodeCacheDataset(
        tmp_path,
        split="train",
        load_language_embedding=True,
        expected_num_progress_bins=53,
    )
    sample = dataset[0]
    assert sample["trajectory_frame_latents"].shape == (53, 4, 1, 8, 8)
    assert sample["num_progress_bins"] == 53
    assert sample["current_latents"].shape[0] == 7
    assert sample["progress"].tolist() == torch.linspace(0.0, 1.0, 7).tolist()
    assert progress_episode_collate_fn([sample]) is sample


def test_progress_soft_targets_use_explicit_53_frame_bins():
    target = torch.tensor([0.0, 0.5, 1.0])
    distribution = build_progress_target_distribution(
        target, num_bins=53, sigma_bins=2.0
    )
    assert distribution.shape == (3, 53)
    torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(3))
    assert distribution.argmax(dim=-1).tolist() == [0, 26, 52]


def test_progress_model_rejects_non_53_frame_memory():
    model = build_progress_model(small_config("serial_latent", num_layers=1))
    with pytest.raises(ValueError, match="exactly 53"):
        model(
            current_latent=torch.randn(2, 4, 1, 8, 8),
            trajectory_frame_latents=torch.randn(1, 14, 4, 1, 8, 8),
        )


def test_progress_dataset_rejects_legacy_14_slice_schema(tmp_path):
    manifest_entry = {
        "schema_version": 1,
        "episode_name": "legacy/episode",
        "split": "train",
        "cache_file": "episodes/legacy.pt",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest_entry) + "\n")
    with pytest.raises(ValueError, match="Unsupported Progress cache schema"):
        ProgressEpisodeCacheDataset(tmp_path, split="train")
