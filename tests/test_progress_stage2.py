import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from data.progress.progress_cache_dataset import (
    SCHEMA_VERSION,
    ProgressEpisodeCacheDataset,
    progress_episode_collate_fn,
)
from models.progress_stage2 import (
    ProgressStage2Config,
    SharedFrameLatentEncoder,
    TemporalAlignmentHead,
    TokenLateInteractionAlignmentHead,
    build_progress_target_distribution,
    build_progress_model,
    compute_progress_loss,
)
from train.train_progress_stage2 import (
    extract_episode_video_features,
    model_config_from_yaml,
    prepare_episode_queries,
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


def test_default_progress_model_preserves_archived_baseline_structure():
    # New ablation fields default to the old joint-view + pooled-cosine path.
    model = build_progress_model(small_config("serial_latent", num_layers=1))
    assert isinstance(model.alignment_head, TemporalAlignmentHead)
    assert model.frame_encoder.view_encoding_mode == "joint"
    assert model.frame_encoder.output_num_tokens == 4
    assert not any("view_embeddings" in name for name in model.state_dict())


def test_progress_yaml_parser_keeps_defaults_and_reads_ablation_fields():
    legacy = model_config_from_yaml(
        OmegaConf.create(
            {"progress_model": {"fusion_mode": "serial_latent", "num_layers": 6}}
        )
    )
    assert legacy.patch_size == (1, 2, 2)
    assert legacy.view_encoding_mode == "joint"
    assert legacy.alignment_head_mode == "pooled_cosine"

    ablation = model_config_from_yaml(
        OmegaConf.create(
            {
                "progress_model": {
                    "fusion_mode": "serial_latent",
                    "num_layers": 6,
                    "patch_size": [1, 1, 1],
                    "view_encoding_mode": "split_height",
                    "num_views": 2,
                    "tokens_per_view": 12,
                    "alignment_head_mode": "token_late_interaction",
                    "late_interaction_temperature": 0.05,
                }
            }
        )
    )
    assert ablation.patch_size == (1, 1, 1)
    assert ablation.view_encoding_mode == "split_height"
    assert ablation.num_views == 2
    assert ablation.tokens_per_view == 12
    assert ablation.alignment_head_mode == "token_late_interaction"
    assert ablation.late_interaction_temperature == 0.05


def test_frame_encoder_supports_patch_and_split_view_ablations():
    # A1 removes spatial patch reduction but keeps one shared 4-token budget.
    patch_config = small_config("serial_latent", num_layers=1)
    patch_config.patch_size = (1, 1, 1)
    patch_encoder = SharedFrameLatentEncoder(patch_config)
    patch_tokens = patch_encoder(torch.randn(2, 4, 1, 8, 8))
    assert patch_encoder.tokenizer.proj.kernel_size == (1, 1, 1)
    assert patch_tokens.shape == (2, 4, 32)

    # A2 splits the vertical two-view composite, shares encoder weights, and
    # guarantees three resampler tokens per view.
    split_config = small_config("serial_latent", num_layers=1)
    split_config.view_encoding_mode = "split_height"
    split_config.num_views = 2
    split_config.tokens_per_view = 3
    split_encoder = SharedFrameLatentEncoder(split_config)
    split_tokens = split_encoder(torch.randn(2, 4, 1, 8, 8))
    assert split_encoder.output_num_tokens == 6
    assert split_tokens.shape == (2, 6, 32)
    split_tokens.sum().backward()
    assert split_encoder.view_embeddings.grad is not None

    with pytest.raises(ValueError, match="not divisible by num_views"):
        split_encoder(torch.randn(1, 4, 1, 7, 8))


def test_token_late_interaction_preserves_local_tokens_and_backpropagates():
    config = small_config("serial_latent", num_layers=1)
    config.alignment_head_mode = "token_late_interaction"
    config.late_interaction_temperature = 0.07
    model = build_progress_model(config)
    assert isinstance(model.alignment_head, TokenLateInteractionAlignmentHead)

    outputs = model(
        current_latent=torch.randn(3, 4, 1, 8, 8),
        trajectory_frame_latents=torch.randn(1, 53, 4, 1, 8, 8),
    )
    assert outputs["alignment_logits"].shape == (3, 53)
    assert outputs["query_token_weights"].shape == (3, 4)
    assert outputs["late_interaction_scores"].shape == (3, 53, 4)
    torch.testing.assert_close(
        outputs["query_token_weights"].sum(dim=-1),
        torch.ones(3),
    )
    compute_progress_loss(outputs, torch.tensor([0.0, 0.5, 1.0]))[
        "total_loss"
    ].backward()
    assert model.alignment_head.query_importance.weight.grad is not None
    assert model.frame_encoder.tokenizer.proj.weight.grad is not None


def test_combined_detail_preserving_model_supports_mixed_episodes():
    config = small_config("serial_latent", num_layers=1)
    config.patch_size = (1, 1, 1)
    config.view_encoding_mode = "split_height"
    config.num_views = 2
    config.tokens_per_view = 3
    config.alignment_head_mode = "token_late_interaction"
    model = build_progress_model(config).eval()
    outputs = model(
        current_latent=torch.randn(4, 4, 1, 8, 8),
        trajectory_frame_latents=torch.randn(2, 53, 4, 1, 8, 8),
        query_episode_indices=torch.tensor([0, 0, 1, 1]),
    )
    assert model.frame_encoder.output_num_tokens == 6
    assert outputs["alignment_logits"].shape == (4, 53)
    assert outputs["query_token_weights"].shape == (4, 6)


def test_layerwise_model_can_use_token_late_interaction_head():
    config = small_config("layerwise_wvm", num_layers=2)
    config.alignment_head_mode = "token_late_interaction"
    model = build_progress_model(config)
    outputs = model(
        current_latent=torch.randn(2, 4, 1, 8, 8),
        trajectory_frame_latents=torch.randn(1, 53, 4, 1, 8, 8),
        video_hidden_states=[torch.randn(1, 12, 48) for _ in range(2)],
        video_grid_size=(3, 2, 2),
    )
    assert outputs["alignment_logits"].shape == (2, 53)
    assert outputs["late_interaction_scores"].shape == (2, 53, 4)


def test_serial_progress_forward_and_loss_backward():
    # Serial 必须对每个单帧 query 输出严格的 53-bin 分布，并能反向更新 Progress 参数。
    model = build_progress_model(small_config("serial_latent", num_layers=2))
    # 5 个当前帧 query；每帧 latent 为 [C_z=4,T=1,H_z=8,W_z=8]。
    current = torch.randn(5, 4, 1, 8, 8)
    # 同一条 episode 的生成轨迹 memory：[B_m=1,F=53,C_z=4,T=1,H_z=8,W_z=8]。
    trajectory_frames = torch.randn(1, 53, 4, 1, 8, 8)
    # 5 个 query 的连续进度标签，形状 [B_q=5]，取值范围 [0,1]。
    target = torch.linspace(0.0, 1.0, 5)

    # 输出必须为每个 query 独立预测一个 53 帧位置分布。
    outputs = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
    )
    # 连续进度、期望帧编号和离散 argmax 帧编号均为 [B_q=5]。
    assert outputs["progress"].shape == (5,)
    assert outputs["expected_frame"].shape == (5,)
    assert outputs["matched_frame"].shape == (5,)
    # 每个 query 对应 53 个轨迹位置，因此概率矩阵为 [B_q=5,F=53]。
    assert outputs["alignment_probabilities"].shape == (5, 53)
    # softmax 后每一行都是合法概率分布，沿 53 个位置求和必须为 1。
    torch.testing.assert_close(
        outputs["alignment_probabilities"].sum(dim=-1),
        torch.ones(5),
    )
    # total_loss 是标量；backward 后至少一个可训练 Progress 参数必须得到梯度。
    losses = compute_progress_loss(outputs, target)
    losses["total_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_serial_mixed_episode_batch_matches_independent_episode_forwards():
    # E=2 条轨迹，每条 Q=3 个 query；混合前向必须严格等价于分别运行两条 episode。
    torch.manual_seed(7)
    model = build_progress_model(small_config("serial_latent", num_layers=2)).eval()
    current = torch.randn(6, 4, 1, 8, 8)
    trajectory_frames = torch.randn(2, 53, 4, 1, 8, 8)
    query_episode_indices = torch.tensor([0, 0, 0, 1, 1, 1])

    mixed = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        query_episode_indices=query_episode_indices,
    )
    first_episode = model(
        current_latent=current[:3],
        trajectory_frame_latents=trajectory_frames[:1],
    )
    second_episode = model(
        current_latent=current[3:],
        trajectory_frame_latents=trajectory_frames[1:],
    )
    torch.testing.assert_close(
        mixed["alignment_logits"][:3],
        first_episode["alignment_logits"],
    )
    torch.testing.assert_close(
        mixed["alignment_logits"][3:],
        second_episode["alignment_logits"],
    )


def test_serial_mixed_episode_batch_has_no_cross_episode_memory_leakage():
    # 只替换 episode 1 的 memory 时，属于 episode 0 的前三个 query 输出必须完全不变。
    torch.manual_seed(11)
    model = build_progress_model(small_config("serial_latent", num_layers=1)).eval()
    current = torch.randn(6, 4, 1, 8, 8)
    trajectory_frames = torch.randn(2, 53, 4, 1, 8, 8)
    query_episode_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    before = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        query_episode_indices=query_episode_indices,
    )["alignment_logits"]
    changed_memory = trajectory_frames.clone()
    changed_memory[1] = torch.randn_like(changed_memory[1]) * 5.0
    after = model(
        current_latent=current,
        trajectory_frame_latents=changed_memory,
        query_episode_indices=query_episode_indices,
    )["alignment_logits"]
    torch.testing.assert_close(before[:3], after[:3])


def test_mixed_episode_micro_batch_gradients_equal_one_full_forward():
    # E=2,Q=2 一次前向，与两个 E_micro=1 前向各乘 1/2，必须产生相同参数梯度。
    torch.manual_seed(13)
    full_model = build_progress_model(small_config("serial_latent", num_layers=1))
    micro_model = copy.deepcopy(full_model)
    current = torch.randn(4, 4, 1, 8, 8)
    trajectory_frames = torch.randn(2, 53, 4, 1, 8, 8)
    target = torch.tensor([0.0, 1.0, 0.25, 0.75])
    episode_ids = torch.tensor([0, 0, 1, 1])

    full_outputs = full_model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        query_episode_indices=episode_ids,
    )
    compute_progress_loss(
        full_outputs,
        target,
        episode_ids=episode_ids,
    )["total_loss"].backward()

    for episode_index in range(2):
        start = episode_index * 2
        end = start + 2
        micro_outputs = micro_model(
            current_latent=current[start:end],
            trajectory_frame_latents=trajectory_frames[episode_index : episode_index + 1],
            query_episode_indices=torch.zeros(2, dtype=torch.long),
        )
        micro_loss = compute_progress_loss(
            micro_outputs,
            target[start:end],
            episode_ids=torch.zeros(2, dtype=torch.long),
        )["total_loss"]
        (micro_loss * 0.5).backward()

    for full_parameter, micro_parameter in zip(
        full_model.parameters(), micro_model.parameters()
    ):
        assert full_parameter.grad is not None
        assert micro_parameter.grad is not None
        torch.testing.assert_close(
            full_parameter.grad,
            micro_parameter.grad,
            rtol=1e-4,
            atol=1e-5,
        )


def test_layerwise_progress_is_asymmetric_and_uses_every_layer():
    # Layerwise 读取每层 WAN hidden，但不得原地修改 hidden，也不得把梯度写回冻结 WAN。
    model = build_progress_model(small_config("layerwise_wvm", num_layers=3))
    # B_q=4 个当前单帧 latent，每个形状 [C_z=4,T=1,H_z=8,W_z=8]。
    current = torch.randn(4, 4, 1, 8, 8)
    # 一条共享的 53 帧轨迹 memory：[1,53,4,1,8,8]。
    trajectory_frames = torch.randn(1, 53, 4, 1, 8, 8)
    # 用 3 层小模型模拟 WAN hidden：每层 [B_m=1,L=12,D_video=48]。
    # video_grid_size=(3,2,2)，所以 L=3*2*2=12。
    video_hidden = [torch.randn(1, 12, 48) for _ in range(3)]
    # 保存副本，用于验证 Layerwise Progress 不会原地修改冻结 WAN 的输出。
    originals = [item.clone() for item in video_hidden]

    outputs = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        video_hidden_states=video_hidden,
        video_grid_size=(3, 2, 2),
    )
    assert outputs["progress"].shape == (4,)
    assert outputs["alignment_probabilities"].shape == (4, 53)
    # 4 个查询标签 [0,1/3,2/3,1]；标量 total_loss 反向只更新 Progress 模型。
    compute_progress_loss(outputs, torch.linspace(0.0, 1.0, 4))["total_loss"].backward()

    for before, after in zip(originals, video_hidden):
        # 数值必须保持完全一致，且作为冻结输入不应积累 .grad。
        torch.testing.assert_close(before, after)
        assert after.grad is None
    # 3 个 asymmetric block 都必须实际参与计算，而不是只读取最后一层 WAN hidden。
    assert all(block.video_projection.weight.grad is not None for block in model.blocks)
    # 共享的 53 帧 frame encoder 也必须从最终匹配 loss 得到梯度。
    assert model.frame_encoder.tokenizer.proj.weight.grad is not None


def test_layerwise_progress_supports_mixed_episode_query_mapping():
    # Layerwise 同样支持 E=2、Q=2，并让每个 query 只读取对应 episode 的 WAN hidden。
    model = build_progress_model(small_config("layerwise_wvm", num_layers=2)).eval()
    current = torch.randn(4, 4, 1, 8, 8)
    trajectory_frames = torch.randn(2, 53, 4, 1, 8, 8)
    video_hidden = [torch.randn(2, 12, 48) for _ in range(2)]
    query_episode_indices = torch.tensor([0, 0, 1, 1])
    outputs = model(
        current_latent=current,
        trajectory_frame_latents=trajectory_frames,
        video_hidden_states=video_hidden,
        video_grid_size=(3, 2, 2),
        query_episode_indices=query_episode_indices,
    )
    assert outputs["alignment_logits"].shape == (4, 53)
    first_episode = model(
        current_latent=current[:2],
        trajectory_frame_latents=trajectory_frames[:1],
        video_hidden_states=[hidden[:1] for hidden in video_hidden],
        video_grid_size=(3, 2, 2),
    )
    second_episode = model(
        current_latent=current[2:],
        trajectory_frame_latents=trajectory_frames[1:],
        video_hidden_states=[hidden[1:] for hidden in video_hidden],
        video_grid_size=(3, 2, 2),
    )
    torch.testing.assert_close(
        outputs["alignment_logits"][:2],
        first_episode["alignment_logits"],
    )
    torch.testing.assert_close(
        outputs["alignment_logits"][2:],
        second_episode["alignment_logits"],
    )
    losses = compute_progress_loss(
        outputs,
        torch.tensor([0.0, 1.0, 0.25, 0.75]),
        episode_ids=query_episode_indices,
    )
    losses["total_loss"].backward()
    assert all(block.video_projection.weight.grad is not None for block in model.blocks)


def test_progress_cache_dataset_preserves_whole_episode(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    language_dir = tmp_path / "language"
    language_dir.mkdir()
    torch.save(torch.randn(5, 16), language_dir / "task_000001.pt")
    # 构造一条包含 N=7 个真实查询帧的完整 episode 缓存。
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_name": "task/episode_000001",
        "task_index": 1,
        "total_frames": 7,
        "num_progress_bins": 53,
        # 原始时序 latent：[C_z=4,T_z=3,H_z=8,W_z=8]；测试中用 T_z=3 缩小体积。
        "trajectory_latent": torch.randn(4, 3, 8, 8),
        # 解码后逐帧重编码得到的显式 memory：[F=53,C_z=4,T=1,H_z=8,W_z=8]。
        "trajectory_frame_latents": torch.randn(53, 4, 1, 8, 8),
        # episode 的全部真实单帧 query：[N=7,C_z=4,T=1,H_z=8,W_z=8]。
        "current_latents": torch.randn(7, 4, 1, 8, 8),
        # 每个真实帧的归一化时间进度和原始帧编号，二者形状都是 [N=7]。
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
    # Dataset 必须保留整条 episode，不能把它预先拆成 7 个独立 dataset item。
    assert sample["trajectory_frame_latents"].shape == (53, 4, 1, 8, 8)
    assert sample["num_progress_bins"] == 53
    assert sample["current_latents"].shape[0] == 7
    assert sample["progress"].tolist() == torch.linspace(0.0, 1.0, 7).tolist()
    collated = progress_episode_collate_fn([sample, sample])
    # collate 只保留长度 E=2 的 episode list，不会尝试 stack 不同长度的 N 维。
    assert len(collated) == 2
    assert collated[0] is sample
    assert collated[1] is sample


def test_layerwise_episode_features_preserve_variable_language_lengths():
    # 不同任务的 UMT5 token 数 L_e 可以不同；mixed batch 必须以 list 交给 WAN，
    # 不能在 Trainer 中直接 stack 成要求相同 L 的 tensor。
    class FakeFrozenVGM:
        def extract_bridge_hidden_states(self, **kwargs):
            embeddings = kwargs["language_embeddings"]
            assert isinstance(embeddings, list)
            assert [item.shape for item in embeddings] == [
                torch.Size([5, 16]),
                torch.Size([8, 16]),
            ]
            assert kwargs["trajectory_latent"].shape[0] == 2
            return {"hidden_states": [], "grid_sizes": torch.ones(2, 3)}

    episodes = [
        {
            "first_frame": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "last_frame": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "language_embedding": torch.randn(5, 16),
        },
        {
            "first_frame": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "last_frame": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "language_embedding": torch.randn(8, 16),
        },
    ]
    prepared = {
        "episodes": episodes,
        "trajectory_latent": torch.randn(2, 4, 3, 8, 8),
    }
    output = extract_episode_video_features(
        FakeFrozenVGM(),
        SimpleNamespace(progress_model={"feature_timestep": 0.0}),
        prepared,
    )
    assert output["grid_sizes"].shape == (2, 3)


def test_prepare_episode_queries_rejects_empty_episode():
    with pytest.raises(ValueError, match="at least one current-frame query"):
        prepare_episode_queries(
            torch.empty(0, 4, 1, 8, 8),
            torch.empty(0),
            queries_per_episode=4,
        )


def test_progress_soft_targets_use_explicit_53_frame_bins():
    # 归一化进度 0/0.5/1 必须精确落在 53-frame memory 的 0/26/52 三个中心位置。
    # target: [B_q=3]，分别代表 episode 的起点、中点和终点。
    target = torch.tensor([0.0, 0.5, 1.0])
    # distribution: [B_q=3,F=53]；sigma_bins=2 表示标签在相邻帧位置平滑扩散。
    distribution = build_progress_target_distribution(
        target, num_bins=53, sigma_bins=2.0
    )
    assert distribution.shape == (3, 53)
    torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(3))
    assert distribution.argmax(dim=-1).tolist() == [0, 26, 52]


def test_progress_ranking_never_compares_queries_from_different_episodes():
    # 两个 query 分属两条 episode，Q=1 时没有合法的 episode 内 pair，ranking 必须为 0。
    logits = torch.zeros(2, 53)
    outputs = {
        "alignment_logits": logits,
        "progress": torch.tensor([1.0, 0.0], requires_grad=True),
    }
    target = torch.tensor([0.0, 1.0])
    ungrouped = compute_progress_loss(outputs, target)
    grouped = compute_progress_loss(
        outputs,
        target,
        episode_ids=torch.tensor([0, 1]),
    )
    assert ungrouped["ranking_loss"].item() > 0.0
    torch.testing.assert_close(grouped["ranking_loss"], torch.tensor(0.0))


def test_zero_ranking_weight_skips_pairwise_path():
    logits = torch.zeros(2, 53, requires_grad=True)
    probabilities = logits.softmax(dim=-1)
    positions = torch.linspace(0.0, 1.0, 53)
    outputs = {
        "alignment_logits": logits,
        "progress": (probabilities * positions).sum(dim=-1),
    }
    # Invalid episode_ids would fail inside pairwise ranking. Weight zero must
    # bypass that branch entirely.
    losses = compute_progress_loss(
        outputs,
        torch.tensor([0.0, 1.0]),
        episode_ids=torch.tensor([0]),
        ranking_weight=0.0,
    )
    torch.testing.assert_close(losses["ranking_loss"], torch.tensor(0.0))
    torch.testing.assert_close(
        losses["total_loss"],
        losses["alignment_loss"] + losses["regression_loss"],
    )
    losses["total_loss"].backward()
    assert logits.grad is not None


def test_progress_model_rejects_non_53_frame_memory():
    # 禁止把旧的 14 个 VAE temporal slice 当作 53 张可寻址生成帧使用。
    model = build_progress_model(small_config("serial_latent", num_layers=1))
    with pytest.raises(ValueError, match="exactly 53"):
        # query 合法，但 memory 只有 14 个 temporal slice；模型必须拒绝这类语义混用。
        model(
            current_latent=torch.randn(2, 4, 1, 8, 8),
            trajectory_frame_latents=torch.randn(1, 14, 4, 1, 8, 8),
        )


def test_progress_dataset_rejects_legacy_14_slice_schema(tmp_path):
    # Schema v1 缺少逐帧 re-encode 的 53-slot memory，必须显式报错而不是静默兼容。
    # schema_version=1 代表旧 14-slice 缓存，不包含 [53,C_z,1,H_z,W_z] 显式帧 memory。
    manifest_entry = {
        "schema_version": 1,
        "episode_name": "legacy/episode",
        "split": "train",
        "cache_file": "episodes/legacy.pt",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest_entry) + "\n")
    with pytest.raises(ValueError, match="Unsupported Progress cache schema"):
        ProgressEpisodeCacheDataset(tmp_path, split="train")
