import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from data.video_bridge.video_bridge_dataset import VideoBridgeDataset, video_bridge_collate_fn
from models.vgm_bridge_stage1 import EndpointTokenAdapter, VGMBridgeStage1
from models.vgm_multiview import (
    BottleneckResidual,
    CrossViewAdapter,
    CrossViewConsistencyHead,
    MosaicViewEmbedding,
    MultiViewWanRunner,
    SEPARATE_VIEW_MODES,
    matched_control_bottleneck_dim,
)
from scripts.cache_vgm_progress_latents import sampled_trajectory_frames
from train.eval_vgm_bridge_stage1 import (
    cross_view_motion_sync_metrics,
    multiview_pixel_metrics,
    pixel_metrics,
    role_region_pixel_metrics,
    save_contact_sheet,
    split_rgb_views,
    trajectory_slot_diagnostics,
    window_fingerprint,
)
from utils.config_utils import load_config_with_base
from wan.utils.fm import FlowMatchScheduler


ROOT = Path(__file__).resolve().parents[1]


def bare_bridge(mode: str, layout: str = "vertical") -> VGMBridgeStage1:
    model = VGMBridgeStage1.__new__(VGMBridgeStage1)
    nn.Module.__init__(model)
    model.multiview_mode = mode
    model.num_views = 2
    model.latent_channels = 4
    model.config = type("Config", (), {"multiview_layout": layout})()
    return model


def test_multiview_config_matrix_resolves_complete_configs() -> None:
    expected_modes = {
        "legacy",
        "rgb_mosaic_view_embedding",
        "rgb_mosaic_adapter_control",
        "latent_spatial",
        "latent_channel",
        "independent",
        "independent_consistency",
        "independent_adapter_control",
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "autoregressive_high_to_wrist",
    }
    configs = sorted((ROOT / "configs").glob("vgm_bridge_mv*_53f.yaml"))
    assert len(configs) == 18
    resolved_modes = set()
    for path in configs:
        config = load_config_with_base(path)
        resolved_modes.add(str(config.common.multiview_mode))
        assert int(config.common.num_video_frames) == 52
        assert int(config.training.max_steps) == 50004
        assert len(config.dataset.image_columns) == 2
        assert config.model.wan.checkpoint_path
        assert int(config.training.batch_size) * int(
            config.training.gradient_accumulation_steps
        ) * 2 == int(config.training.expected_global_batch_size) == 8
        assert int(config.training.seed) == 0
        assert bool(config.dataset.index_seeded_sampling)
        assert bool(config.dataset.return_separate_views) == (
            str(config.common.multiview_mode) in SEPARATE_VIEW_MODES
        )
        assert str(config.dataset.view_layout) == str(config.common.multiview_output_layout)
    assert resolved_modes == expected_modes


def test_dataset_collate_preserves_camera_dimension() -> None:
    sample = {
        "first_frame": torch.zeros(3, 8, 4),
        "video_frames": torch.zeros(2, 3, 8, 4),
        "first_view_frames": torch.zeros(2, 3, 4, 4),
        "view_video_frames": torch.zeros(2, 2, 3, 4, 4),
        "first_role_mask": None,
        "role_mask_frames": None,
        "language_embedding": None,
        "first_state": None,
        "video_states": None,
        "last_state": None,
    }
    batch = video_bridge_collate_fn([sample, sample])
    assert batch["first_view_frames"].shape == (2, 2, 3, 4, 4)
    assert batch["view_video_frames"].shape == (2, 2, 2, 3, 4, 4)


def test_index_seeded_sampling_uses_sampler_index_without_global_rng() -> None:
    dataset = VideoBridgeDataset.__new__(VideoBridgeDataset)
    dataset.episodes = [{"episode_name": "episode_0"}, {"episode_name": "episode_1"}]
    dataset.index_seeded_sampling = True
    dataset.sampling_seed = 17
    episode_a, rng_a = dataset._episode_and_rng_for_index(3, 0)
    episode_b, rng_b = dataset._episode_and_rng_for_index(3, 0)
    episode_c, rng_c = dataset._episode_and_rng_for_index(4, 0)
    assert episode_a["episode_name"] == episode_b["episode_name"] == "episode_1"
    assert episode_c["episode_name"] == "episode_0"
    assert rng_a is not None and rng_b is not None and rng_c is not None
    assert rng_a.getstate() == rng_b.getstate()
    assert rng_a.getstate() != rng_c.getstate()


@pytest.mark.parametrize(
    ("layout", "expected_shape"),
    [("vertical", (3, 3, 8, 4)), ("horizontal", (3, 3, 4, 8))],
)
def test_dataset_composes_view_tensor_without_changing_view_order(layout, expected_shape) -> None:
    dataset = VideoBridgeDataset.__new__(VideoBridgeDataset)
    dataset.view_layout = layout
    dataset.video_size = expected_shape[-2:]
    high = torch.ones(3, 3, 4, 4)
    wrist = torch.full((3, 3, 4, 4), 2.0)
    composed = dataset._compose_view_tensor(torch.stack([high, wrist], dim=1))
    assert composed.shape == expected_shape
    if layout == "vertical":
        assert torch.equal(composed[..., :4, :], high)
        assert torch.equal(composed[..., 4:, :], wrist)
    else:
        assert torch.equal(composed[..., :, :4], high)
        assert torch.equal(composed[..., :, 4:], wrist)


def test_multiview_metrics_keep_high_and_wrist_results_separate() -> None:
    gt = torch.zeros(1, 5, 3, 8, 4)
    pred = gt.clone()
    pred[..., :4, :] = 1.0
    metrics = multiview_pixel_metrics(
        gt,
        pred,
        tail_condition_frames=1,
        view_layout="vertical",
        view_names=["high", "wrist"],
    )
    assert metrics["view_high_generated_mse"] == pytest.approx(1.0)
    assert metrics["view_wrist_generated_mse"] == pytest.approx(0.0)
    assert metrics["view_macro_generated_mse"] == pytest.approx(0.5)
    split = split_rgb_views(pred[0], "vertical", ["high", "wrist"])
    assert [name for name, _ in split] == ["high", "wrist"]
    assert all(view.shape == (5, 3, 4, 4) for _, view in split)


def test_v1_endpoint_is_counted_as_generated_not_hard_clamped() -> None:
    gt = torch.zeros(1, 5, 3, 4, 4)
    pred = gt.clone()
    pred[:, -1] = 1.0
    legacy = pixel_metrics(gt, pred, tail_condition_frames=1)
    v1_proper = pixel_metrics(
        gt,
        pred,
        tail_condition_frames=1,
        hard_clamped_tail_frames=0,
    )
    assert legacy["generated_mse"] == pytest.approx(0.0)
    assert v1_proper["generated_mse"] > 0.0
    assert v1_proper["endpoint_mse"] == pytest.approx(1.0)


def test_v1_endpoint_is_included_in_role_region_rgb_metrics() -> None:
    gt = torch.zeros(1, 5, 3, 4, 4)
    pred = gt.clone()
    pred[:, -1] = 1.0
    role = torch.ones_like(gt)
    legacy = role_region_pixel_metrics(
        gt,
        pred,
        role,
        "binary",
        tail_condition_frames=1,
    )
    v1_proper = role_region_pixel_metrics(
        gt,
        pred,
        role,
        "binary",
        tail_condition_frames=1,
        hard_clamped_tail_frames=0,
    )
    assert legacy["rgb_role_foreground_mse"] == pytest.approx(0.0)
    assert v1_proper["rgb_role_foreground_mse"] > 0.0


def test_window_fingerprint_is_independent_of_machine_paths() -> None:
    common = {
        "episode_name": "episode_000045",
        "task_index": 3,
        "frame_indices": [0, 2, 5, 9],
        "total_frames": 10,
    }
    local = [{**common, "video_path": "/Users/n/data/episode.mp4"}]
    remote = [{**common, "video_path": "/mnt/workspace/data/episode.mp4"}]
    assert window_fingerprint(local) == window_fingerprint(remote)


def test_contact_sheet_preserves_all_53_frames(tmp_path: Path) -> None:
    frame_count = 53
    gt = torch.stack(
        [torch.full((3, 4, 4), index / (frame_count - 1)) for index in range(frame_count)]
    )
    pred = 1.0 - gt
    output = tmp_path / "sheet.png"
    save_contact_sheet(gt, pred, output)
    with Image.open(output) as sheet:
        assert sheet.width == 84 + frame_count * 4
        assert sheet.height == 34 + 2 * 4


def test_cross_view_motion_sync_metric_detects_shifted_camera_timing() -> None:
    gt = torch.zeros(1, 5, 3, 8, 4)
    gt[:, 2:, :, :4] = 1.0
    gt[:, 2:, :, 4:] = 1.0
    aligned = cross_view_motion_sync_metrics(
        gt,
        gt,
        view_layout="vertical",
        view_names=["high", "wrist"],
    )
    shifted = gt.clone()
    shifted[:, :, :, 4:] = 0.0
    shifted[:, 4:, :, 4:] = 1.0
    shifted_metrics = cross_view_motion_sync_metrics(
        gt,
        shifted,
        view_layout="vertical",
        view_names=["high", "wrist"],
    )
    assert aligned["xview_motion_profile_excess_l1"] == pytest.approx(0.0)
    assert shifted_metrics["xview_motion_profile_excess_l1"] > 0.0
    assert shifted_metrics["xview_motion_peak_offset_error"] > 0.0


@pytest.mark.parametrize("mode", ["latent_spatial", "latent_channel"])
def test_pack_unpack_view_latents_is_lossless(mode: str) -> None:
    model = bare_bridge(mode)
    latent = torch.randn(2, 2, 4, 3, 5, 7)
    packed = model._pack_view_latents(latent)
    restored = model._unpack_view_latents(packed)
    assert torch.equal(restored, latent)


def test_mosaic_view_embedding_assigns_explicit_bands() -> None:
    adapter = MosaicViewEmbedding(num_views=2, hidden_dim=1)
    with torch.no_grad():
        adapter.embedding.copy_(torch.tensor([[1.0], [2.0]]))
    residual = adapter(
        grid_sizes=torch.tensor([[1, 4, 2]]),
        seq_len=8,
        layout="vertical",
    )
    assert residual.shape == (1, 8, 1)
    assert torch.equal(residual[0, :4, 0], torch.ones(4))
    assert torch.equal(residual[0, 4:, 0], torch.full((4,), 2.0))


def test_cross_view_adapter_is_identity_at_initialization() -> None:
    adapter = CrossViewAdapter(hidden_dim=16, adapter_dim=8, num_heads=2)
    tokens = torch.randn(2, 2, 8, 16)
    output = adapter(tokens, (2, 2, 2))
    assert torch.equal(output, tokens)


def test_adapter_control_is_parameter_matched_to_cross_view_adapter() -> None:
    common = dict(
        num_views=2,
        hidden_dim=3072,
        adapter_dim=256,
        adapter_heads=8,
        layer_indices=[0],
        num_scene_tokens=4,
        capacity_mode="shared",
    )
    control = MultiViewWanRunner(mode="independent_adapter_control", **common)
    cross_view = MultiViewWanRunner(mode="cross_view_attention", **common)
    control_count = sum(parameter.numel() for parameter in control.parameters())
    cross_count = sum(parameter.numel() for parameter in cross_view.parameters())
    assert abs(control_count - cross_count) / cross_count < 0.01


def test_mosaic_control_is_parameter_matched_to_cross_view_adapter() -> None:
    hidden_dim = 3072
    adapter_dim = 256
    layer_indices = [2, 6, 10, 14, 18, 22, 26, 29]
    cross_view = MultiViewWanRunner(
        mode="cross_view_attention",
        num_views=2,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
        adapter_heads=8,
        layer_indices=layer_indices,
        num_scene_tokens=4,
        capacity_mode="shared",
    )
    control_dim = matched_control_bottleneck_dim(hidden_dim, adapter_dim)
    mosaic_embedding = MosaicViewEmbedding(2, hidden_dim)
    mosaic_adapters = nn.ModuleList(
        [BottleneckResidual(hidden_dim, control_dim) for _ in layer_indices]
    )
    control_count = sum(
        parameter.numel()
        for module in (mosaic_embedding, mosaic_adapters)
        for parameter in module.parameters()
    )
    cross_count = sum(parameter.numel() for parameter in cross_view.parameters())
    assert abs(control_count - cross_count) / cross_count < 0.01


def test_endpoint_context_uses_only_independently_encoded_boundaries() -> None:
    model = bare_bridge("endpoint_scene_context")
    condition = torch.randn(1, 2, 4, 4, 3, 3)
    endpoint = torch.randn(1, 2, 4, 1, 3, 3)
    context = model._endpoint_context_view_latent(condition, endpoint)
    assert torch.equal(context[:, :, :, 0:1], condition[:, :, :, 0:1])
    assert torch.equal(context[:, :, :, -1:], endpoint)
    assert torch.count_nonzero(context[:, :, :, 1:-1]) == 0


def test_consistency_loss_rewards_same_stage_pairing() -> None:
    head = CrossViewConsistencyHead(
        latent_channels=2,
        num_views=2,
        projection_dim=2,
        temperature=0.05,
    )
    with torch.no_grad():
        for projector in head.projectors:
            projector.weight.copy_(torch.eye(2))
            projector.bias.zero_()
    stages = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    prediction = stages.transpose(0, 1).view(1, 1, 2, 3, 1, 1).repeat(1, 2, 1, 1, 1, 1)
    unknown = torch.ones(1, 1, 3, 1, 1)
    aligned = head(prediction, unknown)
    shuffled = prediction.clone()
    shuffled[:, 1] = shuffled[:, 1, :, torch.tensor([1, 2, 0])]
    assert aligned < head(shuffled, unknown)


def test_consistency_loss_does_not_use_other_episodes_as_negatives() -> None:
    head = CrossViewConsistencyHead(
        latent_channels=2,
        num_views=2,
        projection_dim=2,
        temperature=0.05,
    )
    with torch.no_grad():
        for projector in head.projectors:
            projector.weight.copy_(torch.eye(2))
            projector.bias.zero_()
    stages = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    one_episode = stages.transpose(0, 1).view(1, 1, 2, 3, 1, 1).repeat(1, 2, 1, 1, 1, 1)
    two_identical_episodes = one_episode.repeat(2, 1, 1, 1, 1, 1)
    single_loss = head(one_episode, torch.ones(1, 1, 3, 1, 1)).item()
    duplicate_loss = head(
        two_identical_episodes, torch.ones(2, 1, 3, 1, 1)
    ).item()
    assert single_loss == pytest.approx(duplicate_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 projection test requires CUDA")
def test_consistency_loss_accepts_bfloat16_model_weights() -> None:
    device = torch.device("cuda")
    head = CrossViewConsistencyHead(4, 2, 8, 0.1).to(device=device, dtype=torch.bfloat16)
    prediction = torch.randn(2, 2, 4, 5, 2, 2, device=device, dtype=torch.bfloat16)
    loss = head(prediction, torch.ones(2, 1, 5, 1, 1, device=device))
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_slot_diagnostics_detects_temporal_shift_and_collision() -> None:
    gt = torch.zeros(8, 3, 16, 16)
    for index in range(8):
        gt[index, :, :, : index + 1] = 1.0
    aligned_metrics, _, aligned_slots = trajectory_slot_diagnostics(gt, gt)
    collapsed = gt[[0, 0, 0, 3, 3, 3, 7, 7]]
    collapsed_metrics, _, collapsed_slots = trajectory_slot_diagnostics(gt, collapsed)
    assert torch.equal(aligned_slots, torch.arange(8))
    assert aligned_metrics["slot_alignment_mae"] == pytest.approx(0.0)
    assert collapsed_metrics["slot_collision_rate"] > aligned_metrics["slot_collision_rate"]
    assert collapsed_metrics["slot_alignment_mae"] > aligned_metrics["slot_alignment_mae"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="WAN attention smoke test requires CUDA")
@pytest.mark.parametrize(
    "mode",
    [
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "independent_adapter_control",
    ],
)
def test_multiview_wan_runner_preserves_per_view_output_shape(mode: str) -> None:
    from wan.modules.model import WanModel

    device = torch.device("cuda")
    wan = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        cross_attn_norm=True,
    ).to(device)
    wan.freqs = wan.freqs.to(device)
    runner = MultiViewWanRunner(
        mode=mode,
        num_views=2,
        hidden_dim=32,
        adapter_dim=16,
        adapter_heads=4,
        layer_indices=[0],
        num_scene_tokens=2,
        capacity_mode="shared",
    ).to(device)
    inputs = torch.randn(1, 2, 4, 2, 4, 4, device=device)
    timestep = torch.full((1, 2, 8), 100.0, device=device)
    context = [torch.randn(6, 16, device=device)]
    residual = torch.zeros(1, 2, 8, 32, device=device)
    if mode == "joint_attention":
        output = runner.forward_joint(
            wan_model=wan,
            inputs=inputs,
            timestep_tokens=timestep,
            context=context,
            token_residual=residual,
        )
    else:
        output = runner.forward_factorized(
            wan_model=wan,
            inputs=inputs,
            timestep_tokens=timestep,
            context=context,
            token_residual=residual,
            context_inputs=inputs if mode == "endpoint_scene_context" else None,
        )
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


class TinyVideoModel(nn.Module):
    def __init__(self, wan_model: nn.Module) -> None:
        super().__init__()
        self.wan_model = wan_model
        self.precision = torch.float32

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        temporal = 1 + (video.shape[2] - 1) // 4
        value = video.mean(dim=1, keepdim=True)
        value = torch.nn.functional.adaptive_avg_pool3d(value, (temporal, 4, 4))
        return value.repeat(1, 4, 1, 1, 1)

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        frames = 1 + (latent.shape[2] - 1) * 4
        value = torch.nn.functional.interpolate(
            latent[:, :3],
            size=(frames, 8, 8),
            mode="trilinear",
            align_corners=False,
        )
        return value.clamp(-1.0, 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mosaic adapter smoke test requires CUDA")
def test_mosaic_adapter_control_uses_view_embedding_and_plain_adapters() -> None:
    from wan.modules.model import WanModel

    device = torch.device("cuda")
    wan = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        cross_attn_norm=True,
    ).to(device)
    wan.freqs = wan.freqs.to(device)
    model = VGMBridgeStage1.__new__(VGMBridgeStage1)
    nn.Module.__init__(model)
    model.video_model = TinyVideoModel(wan)
    model.config = SimpleNamespace(multiview_layout="vertical")
    model.mosaic_view_embedding = MosaicViewEmbedding(2, 32).to(device)
    model.mosaic_control_adapters = nn.ModuleDict(
        {"0": BottleneckResidual(32, matched_control_bottleneck_dim(32, 16))}
    ).to(device)
    latent = torch.randn(1, 4, 2, 8, 4, device=device)
    seq_len = model._wan_seq_len(latent)
    residual = model._mosaic_view_token_residual(latent, seq_len)
    prediction = model._forward_wan(
        latent_list=[latent[0]],
        timestep_tokens=torch.full((1, seq_len), 100.0, device=device),
        context=[torch.randn(6, 16, device=device)],
        seq_len=seq_len,
        token_residual=residual,
    )
    prediction.square().mean().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"Unused MV0 adapter-control parameters: {missing}"


def tiny_multiview_bridge(
    mode: str,
    device: torch.device,
    capacity_mode: str = "shared",
) -> VGMBridgeStage1:
    from wan.modules.model import WanModel

    channel_multiplier = 2 if mode == "latent_channel" else 1
    latent_channels = 4
    model_channels = latent_channels * channel_multiplier
    wan = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=model_channels * 2 + 4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=model_channels,
        num_heads=4,
        num_layers=2,
        cross_attn_norm=True,
    ).to(device)
    wan.freqs = wan.freqs.to(device)

    model = VGMBridgeStage1.__new__(VGMBridgeStage1)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_video_frames=4,
        video_height=16,
        video_width=8,
        tail_condition_frames=1,
        conditioning_mode="v1_mask_endpoint",
        mask_channels=4,
        multiview_layout="vertical",
        multiview_output_layout="vertical",
        view_capacity_mode=capacity_mode,
        multiview_consistency_weight=0.1,
    )
    model.video_model = TinyVideoModel(wan)
    model.device = device
    model.dtype = torch.float32
    model.latent_channels = latent_channels
    model.model_latent_channels = model_channels
    model.multiview_mode = mode
    model.num_views = 2
    model.uses_separate_views = True
    model.condition_on_last_frame = True
    model.use_mask_condition = True
    model.use_endpoint_adapter = True
    model.endpoint_adapter = EndpointTokenAdapter(model_channels, 32, (1, 2, 2)).to(device)
    model.state_adapter = None
    model.mosaic_view_embedding = None
    model.mosaic_control_adapters = nn.ModuleDict()
    model.cross_view_consistency_head = None
    if mode == "independent_consistency":
        model.cross_view_consistency_head = CrossViewConsistencyHead(
            latent_channels=latent_channels,
            num_views=2,
            projection_dim=8,
            temperature=0.1,
        ).to(device)
    model.multiview_runner = None
    if mode in {
        "independent_adapter_control",
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "autoregressive_high_to_wrist",
    }:
        model.multiview_runner = MultiViewWanRunner(
            mode=mode,
            num_views=2,
            hidden_dim=32,
            adapter_dim=16,
            adapter_heads=4,
            layer_indices=[0],
            num_scene_tokens=2,
            capacity_mode=capacity_mode,
        ).to(device)
    model.additional_view_models = nn.ModuleList()
    if capacity_mode == "separate":
        model.additional_view_models.append(copy.deepcopy(wan))
    model.fm_train_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    model.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
    return model


@pytest.mark.parametrize(
    ("mode", "capacity_mode"),
    [
        ("independent", "separate"),
        ("cross_view_attention", "separate"),
        ("joint_attention", "view_adapter"),
    ],
)
def test_multiview_checkpoint_state_dict_roundtrip_is_strict(
    mode: str,
    capacity_mode: str,
) -> None:
    source = tiny_multiview_bridge(mode, torch.device("cpu"), capacity_mode=capacity_mode)
    restored = tiny_multiview_bridge(mode, torch.device("cpu"), capacity_mode=capacity_mode)
    incompatible = restored.load_state_dict(source.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGM integration smoke test requires CUDA")
@pytest.mark.parametrize(
    "mode",
    [
        "latent_spatial",
        "latent_channel",
        "independent",
        "independent_consistency",
        "independent_adapter_control",
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "autoregressive_high_to_wrist",
    ],
)
def test_multiview_training_step_runs_end_to_end_with_tiny_wan(mode: str) -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge(mode, device)
    first = torch.rand(1, 2, 3, 8, 8, device=device)
    future = torch.rand(1, 4, 2, 3, 8, 8, device=device)
    losses = model._multiview_training_step(
        first_view_frames=first,
        view_video_frames=future,
        language_embeddings=None,
        return_dict=True,
    )
    assert losses["total_loss"].ndim == 0
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["view_0_loss"])
    assert torch.isfinite(losses["view_1_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGM backward smoke test requires CUDA")
@pytest.mark.parametrize(
    "mode",
    [
        "latent_spatial",
        "latent_channel",
        "independent",
        "independent_consistency",
        "independent_adapter_control",
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "autoregressive_high_to_wrist",
    ],
)
def test_multiview_backward_uses_every_trainable_parameter(mode: str) -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge(mode, device)
    losses = model._multiview_training_step(
        first_view_frames=torch.rand(1, 2, 3, 8, 8, device=device),
        view_video_frames=torch.rand(1, 4, 2, 3, 8, 8, device=device),
        language_embeddings=None,
        return_dict=True,
    )
    losses["total_loss"].backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert not missing, f"Trainable parameters were unused in {mode}: {missing}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGM capacity smoke test requires CUDA")
@pytest.mark.parametrize(
    ("mode", "capacity_mode"),
    [
        ("independent", "separate"),
        ("joint_attention", "view_adapter"),
        ("cross_view_attention", "view_adapter"),
        ("cross_view_attention", "separate"),
    ],
)
def test_multiview_capacity_variants_use_every_parameter(mode: str, capacity_mode: str) -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge(mode, device, capacity_mode=capacity_mode)
    losses = model._multiview_training_step(
        first_view_frames=torch.rand(1, 2, 3, 8, 8, device=device),
        view_video_frames=torch.rand(1, 4, 2, 3, 8, 8, device=device),
        language_embeddings=None,
        return_dict=True,
    )
    losses["total_loss"].backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"Unused parameters in {mode}/{capacity_mode}: {missing}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGM sampling smoke test requires CUDA")
@pytest.mark.parametrize(
    "mode",
    [
        "latent_spatial",
        "latent_channel",
        "independent",
        "independent_consistency",
        "independent_adapter_control",
        "joint_attention",
        "cross_view_attention",
        "scene_tokens",
        "endpoint_scene_context",
        "autoregressive_high_to_wrist",
    ],
)
def test_multiview_sampling_returns_full_per_view_video(mode: str) -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge(mode, device)
    first = torch.rand(1, 2, 3, 8, 8, device=device)
    last = torch.rand(1, 2, 3, 8, 8, device=device)
    output = model._sample_multiview(
        first_view_frames=first,
        last_view_frames=last,
        tail_view_frames=None,
        language_embeddings=None,
        num_inference_steps=2,
        generator=torch.Generator(device=device).manual_seed(7),
        return_latent=False,
        return_components=True,
    )
    assert output["view_video"].shape == (1, 2, 5, 3, 8, 8)
    assert output["rgb_video"].shape == (1, 5, 3, 16, 8)
    assert torch.isfinite(output["rgb_video"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VGM sampling smoke test requires CUDA")
def test_multiview_sampling_exports_canonical_stage2_latent() -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge("independent", device)
    first = torch.rand(1, 2, 3, 8, 8, device=device)
    last = torch.rand(1, 2, 3, 8, 8, device=device)
    output = model._sample_multiview(
        first_view_frames=first,
        last_view_frames=last,
        tail_view_frames=None,
        language_embeddings=None,
        num_inference_steps=1,
        generator=torch.Generator(device=device).manual_seed(11),
        return_latent=True,
        return_components=True,
    )
    assert output["view_latent"].shape == (1, 2, 4, 2, 4, 4)
    assert output["rgb_latent"].shape == (1, 4, 2, 4, 4)
    assert output["view_video"].shape == (1, 2, 5, 3, 8, 8)
    assert output["rgb_video"].shape == (1, 5, 3, 16, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Separate-WAN sampling requires CUDA")
def test_separate_wan_mv4_sampling_runs_both_backbones() -> None:
    device = torch.device("cuda")
    model = tiny_multiview_bridge("independent", device, capacity_mode="separate")
    output = model._sample_multiview(
        first_view_frames=torch.rand(1, 2, 3, 8, 8, device=device),
        last_view_frames=torch.rand(1, 2, 3, 8, 8, device=device),
        tail_view_frames=None,
        language_embeddings=None,
        num_inference_steps=1,
        generator=torch.Generator(device=device).manual_seed(19),
        return_latent=False,
        return_components=True,
    )
    assert output["view_video"].shape == (1, 2, 5, 3, 8, 8)


def test_progress_cache_prefers_direct_multiview_decode() -> None:
    class DecodeMustNotRun:
        class VideoModel:
            @staticmethod
            def decode_video(_latent):
                raise AssertionError("direct native-view RGB should avoid a second VAE decode")

        video_model = VideoModel()
        dtype = torch.float32

    direct = torch.rand(1, 5, 3, 16, 8)
    frames = sampled_trajectory_frames(
        DecodeMustNotRun(),
        {"rgb_video": direct},
        torch.randn(1, 4, 2, 4, 4),
        expected_frames=5,
    )
    assert torch.equal(frames, direct[0])
