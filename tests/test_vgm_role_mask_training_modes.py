from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vgm_bridge_stage1 import VGMBridgeStage1, VGMBridgeStage1Config
from scripts.cache_vgm_progress_latents import supports_rgb_only_progress_cache


def bare_model(**config_overrides: object) -> VGMBridgeStage1:
    model = object.__new__(VGMBridgeStage1)
    torch.nn.Module.__init__(model)
    model.config = VGMBridgeStage1Config(
        num_video_frames=8,
        mask_channels=4,
        **config_overrides,
    )
    return model


def test_role_rgb_weight_is_normalized_over_generated_latents() -> None:
    model = bare_model(role_rgb_max_weight=8.0, role_rgb_weight_warmup_steps=0)
    model.use_role_rgb_weight = True
    full_role = torch.full((1, 3, 9, 4, 4), -1.0)
    full_role[:, 0, 1:, :2, :2] = 1.0
    rgb_latent = torch.zeros(1, 48, 3, 2, 2)
    known_mask = torch.zeros(1, 1, 3, 1, 1)
    known_mask[:, :, 0] = 1

    weight = model._role_rgb_latent_weight(full_role, rgb_latent, known_mask, global_step=10)

    assert weight is not None
    active = (1 - known_mask).expand_as(weight).bool()
    assert float(weight[active].mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(weight.max()) > 1.0


def test_no_prompt_role_condition_is_exactly_zero() -> None:
    model = bare_model()
    model.role_mask_condition_mode = "none"
    template = torch.randn(2, 48, 3, 2, 2)
    condition_video = torch.randn(2, 3, 9, 4, 4)

    latent, keep_rate = model._encode_role_condition_latent(
        first_role_mask=None,
        condition_role=condition_video,
        latent_template=template,
        apply_prompt_dropout=False,
    )

    assert torch.count_nonzero(latent) == 0
    assert float(keep_rate) == 0.0


def test_dropped_first_prompt_encodes_an_empty_black_mask(monkeypatch: object) -> None:
    model = bare_model(role_mask_prompt_dropout=1.0)
    model.role_mask_condition_mode = "first_prompt_dropout"
    captured: dict[str, torch.Tensor] = {}

    def fake_encode_condition_latent(
        first_frame_norm: torch.Tensor,
        condition_video: torch.Tensor,
        latent_template: torch.Tensor,
    ) -> torch.Tensor:
        captured["first_frame_norm"] = first_frame_norm
        captured["condition_video"] = condition_video
        return torch.zeros_like(latent_template)

    monkeypatch.setattr(model, "_encode_condition_latent", fake_encode_condition_latent)
    first_role_mask = torch.ones(2, 3, 4, 4)
    condition_video = torch.zeros(2, 3, 9, 4, 4)
    template = torch.zeros(2, 48, 3, 2, 2)

    _, keep_rate = model._encode_role_condition_latent(
        first_role_mask=first_role_mask,
        condition_role=condition_video,
        latent_template=template,
        apply_prompt_dropout=True,
    )

    assert float(keep_rate) == 0.0
    assert torch.all(captured["first_frame_norm"] == -1)
    assert torch.all(captured["condition_video"] == -1)


def test_rgb_only_endpoint_adapter_never_encodes_goal_role_mask() -> None:
    model = bare_model()
    model.use_endpoint_adapter = True
    model.role_mask_fusion_mode = "latent_channel"
    model.role_mask_condition_mode = "none"
    model.dtype = torch.float32
    encode_calls: list[torch.Tensor] = []
    adapter_inputs: list[torch.Tensor] = []

    class FakeVideoModel:
        @staticmethod
        def encode_video(value: torch.Tensor) -> torch.Tensor:
            encode_calls.append(value.clone())
            return torch.ones(value.shape[0], 4, 1, 2, 2)

    def endpoint_adapter(
        endpoint_latent: torch.Tensor,
        target_latent: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        adapter_inputs.append(endpoint_latent.clone())
        return target_latent.new_zeros(target_latent.shape[0], seq_len, 1)

    model.video_model = FakeVideoModel()
    model.endpoint_adapter = endpoint_adapter
    last_frame = torch.rand(1, 3, 4, 4)
    last_role_mask = torch.ones_like(last_frame)
    target_latent = torch.zeros(1, 8, 3, 2, 2)

    model._endpoint_token_residual(
        last_frame=last_frame,
        last_role_mask=last_role_mask,
        target_latent=target_latent,
        seq_len=3,
    )

    assert len(encode_calls) == 1
    assert len(adapter_inputs) == 1
    assert torch.all(adapter_inputs[0][:, :4] == 1)
    assert torch.count_nonzero(adapter_inputs[0][:, 4:]) == 0


@pytest.mark.parametrize(
    ("common", "expected"),
    [
        ({"role_mask_fusion_mode": "none"}, True),
        (
            {
                "role_mask_fusion_mode": "latent_channel",
                "role_mask_training_mode": "joint_flow",
                "role_mask_condition_mode": "none",
            },
            True,
        ),
        (
            {
                "role_mask_fusion_mode": "latent_channel",
                "role_mask_training_mode": "joint_flow_rgb_weight",
                "role_mask_condition_mode": "first_prompt_dropout",
            },
            True,
        ),
        (
            {
                "role_mask_fusion_mode": "latent_channel",
                "role_mask_training_mode": "legacy",
                "role_mask_condition_mode": "legacy",
            },
            False,
        ),
    ],
)
def test_progress_cache_accepts_only_rgb_only_role_models(common: dict[str, object], expected: bool) -> None:
    config = SimpleNamespace(common=common)
    assert supports_rgb_only_progress_cache(config) is expected


@pytest.mark.parametrize(
    ("filename", "training_mode", "condition_mode", "fusion_mode"),
    [
        (
            "vgm_bridge_maskwam_rm0_rgb_success50_detailed_caption_v1_53f.yaml",
            "none",
            "none",
            "none",
        ),
        (
            "vgm_bridge_maskwam_rm1_role_weight_success50_detailed_caption_v1_53f.yaml",
            "rgb_weight",
            "none",
            "none",
        ),
        (
            "vgm_bridge_maskwam_rm2_joint_role_flow_success50_detailed_caption_v1_53f.yaml",
            "joint_flow",
            "none",
            "latent_channel",
        ),
        (
            "vgm_bridge_maskwam_rm3_joint_role_flow_weight_success50_detailed_caption_v1_53f.yaml",
            "joint_flow_rgb_weight",
            "none",
            "latent_channel",
        ),
        (
            "vgm_bridge_maskwam_rm4_joint_role_flow_prompt_dropout_success50_detailed_caption_v1_53f.yaml",
            "joint_flow",
            "first_prompt_dropout",
            "latent_channel",
        ),
    ],
)
def test_role_mask_experiment_matrix_is_controlled(
    filename: str,
    training_mode: str,
    condition_mode: str,
    fusion_mode: str,
) -> None:
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / filename)

    assert config.common.num_video_frames == 52
    assert config.common.conditioning_mode == "v1_mask_endpoint"
    assert config.common.role_mask_training_mode == training_mode
    assert config.common.role_mask_condition_mode == condition_mode
    assert config.common.role_mask_fusion_mode == fusion_mode
    assert config.dataset.bridge_sampling_mode == "full_episode_uniform"
    assert config.dataset.bridge_sampling_jitter is True
    assert config.dataset.role_mask_render_mode == "role_color"
    assert config.training.batch_size * config.training.gradient_accumulation_steps * 2 == 8
    assert config.training.max_steps == 30000
