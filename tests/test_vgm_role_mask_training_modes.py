from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vgm_bridge_stage1 import VGMBridgeStage1, VGMBridgeStage1Config
from scripts.cache_vgm_progress_latents import (
    split_sampled_trajectory_latents,
    supports_rgb_only_progress_cache,
)


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
        prompt_policy="drop",
    )

    assert torch.count_nonzero(latent) == 0
    assert float(keep_rate) == 0.0


def test_dropped_first_prompt_is_exactly_rm2_zero_condition() -> None:
    model = bare_model(role_mask_prompt_dropout=1.0)
    model.role_mask_condition_mode = "first_prompt_dropout"
    first_role_mask = torch.ones(2, 3, 4, 4)
    condition_video = torch.zeros(2, 3, 9, 4, 4)
    template = torch.randn(2, 48, 3, 2, 2)

    latent, keep_rate = model._encode_role_condition_latent(
        first_role_mask=first_role_mask,
        condition_role=condition_video,
        latent_template=template,
        prompt_policy="drop",
    )

    assert float(keep_rate) == 0.0
    assert torch.count_nonzero(latent) == 0


def test_kept_first_prompt_populates_only_first_latent_slice() -> None:
    model = bare_model()
    model.role_mask_condition_mode = "first_prompt_dropout"
    model.dtype = torch.float32
    encoded = torch.full((2, 48, 1, 2, 2), 3.0)

    class FakeVideoModel:
        @staticmethod
        def encode_video(value: torch.Tensor) -> torch.Tensor:
            assert value.shape == (2, 3, 1, 4, 4)
            return encoded.clone()

    model.video_model = FakeVideoModel()
    latent, keep_rate = model._encode_role_condition_latent(
        first_role_mask=torch.ones(2, 3, 4, 4),
        condition_role=torch.zeros(2, 3, 9, 4, 4),
        latent_template=torch.randn(2, 48, 3, 2, 2),
        prompt_policy="keep",
    )

    assert float(keep_rate) == 1.0
    assert torch.all(latent[:, :, 0:1] == 3.0)
    assert torch.count_nonzero(latent[:, :, 1:]) == 0


def test_formal_eval_loss_always_drops_optional_role_prompt() -> None:
    model = bare_model()
    model.train()
    assert model._training_role_prompt_policy() == "stochastic"
    model.eval()
    assert model._training_role_prompt_policy() == "drop"


def test_role_mask_auxiliary_loss_weight_warms_up_by_optimizer_step() -> None:
    model = bare_model(role_mask_loss_weight=0.1, role_mask_loss_warmup_steps=1000)
    assert model._effective_role_mask_loss_weight(0) == pytest.approx(0.0)
    assert model._effective_role_mask_loss_weight(500) == pytest.approx(0.05)
    assert model._effective_role_mask_loss_weight(1000) == pytest.approx(0.1)
    assert model._effective_role_mask_loss_weight(5000) == pytest.approx(0.1)
    assert model._effective_role_mask_loss_weight(None) == pytest.approx(0.1)


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


def test_joint_layerwise_replay_requires_matching_generated_role_latent() -> None:
    model = bare_model()
    model.role_mask_fusion_mode = "latent_channel"
    model.requires_role_mask_prompt = False
    model.latent_channels = 4
    model.device = torch.device("cpu")
    model.dtype = torch.float32

    with pytest.raises(ValueError, match="same Stage 1 sampling run"):
        model.extract_bridge_hidden_states(
            trajectory_latent=torch.randn(1, 4, 3, 2, 2),
            first_frame=torch.randn(1, 3, 4, 4),
            last_frame=torch.randn(1, 3, 4, 4),
        )

    with pytest.raises(ValueError, match="must share"):
        model.extract_bridge_hidden_states(
            trajectory_latent=torch.randn(1, 4, 3, 2, 2),
            trajectory_role_latent=torch.randn(1, 4, 2, 2, 2),
            first_frame=torch.randn(1, 3, 4, 4),
            last_frame=torch.randn(1, 3, 4, 4),
        )


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


def test_progress_cache_preserves_joint_rgb_and_role_latents() -> None:
    model = SimpleNamespace(predict_role_mask=True, role_mask_fusion_mode="latent_channel")
    rgb = torch.randn(1, 48, 14, 2, 2)
    role = torch.randn_like(rgb)

    actual_rgb, actual_role = split_sampled_trajectory_latents(
        model,
        {"rgb_latent": rgb, "role_latent": role},
    )

    assert actual_rgb is rgb
    assert actual_role is role

    with pytest.raises(RuntimeError, match="did not return role_latent"):
        split_sampled_trajectory_latents(model, {"rgb_latent": rgb})
    with pytest.raises(RuntimeError, match="identical shapes"):
        split_sampled_trajectory_latents(
            model,
            {"rgb_latent": rgb, "role_latent": role[:, :, :-1]},
        )


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
    if training_mode in {"joint_flow", "joint_flow_rgb_weight"}:
        assert config.common.role_mask_loss_weight == pytest.approx(0.1)
        assert config.common.role_mask_loss_warmup_steps == 1000
    else:
        assert config.common.role_mask_loss_weight == pytest.approx(0.0)
        assert config.common.role_mask_loss_warmup_steps == 0
