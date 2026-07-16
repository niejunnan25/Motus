# VGM Multi-View Stage 1

## Version Boundary

- Integration base: branch `codex/progress-pipeline-v2`, commit `bb1181b`.
- Implementation branch: `codex/vgm-stage1-multiview`.
- Historical Stage 1 and Stage 2 worktrees are not modified by this branch.
- Every checkpoint stores the fully resolved YAML, Git commit/branch/remote/dirty state,
  launch command, seed, world size, per-rank batch, accumulation, and effective global batch.
- Variant YAML files use `_base_` only to keep the controlled settings in one place.

## Fixed Experiment Contract

The 18 configs share this formal contract:

```text
input       high/wrist first frame + goal frame + task language
target      high/wrist complete 53-frame trajectory
sampling    full episode, endpoints fixed, 51 interior bins jittered
condition   V1-proper mask input + independently encoded endpoint adapter
resolution  224x224 per view; canonical composite is 448x224 or 224x448
training    50,004 optimizer updates, global batch 8, seed 0 by default
inference   fixed windows, fixed seed, 50 flow-denoising steps
```

The dataset exposes either:

```text
RGB mosaic: first_frame       [B,3,H,W]
            video_frames      [B,52,3,H,W]

native:     first_view_frames [B,2,3,224,224]
            view_video_frames [B,52,2,3,224,224]
```

With 53 RGB frames, Wan VAE temporal compression produces 14 latent slices. A
`224x224` view becomes approximately `14 x 7 x 7 = 686` DiT tokens after VAE and
patch embedding; two views use 1,372 tokens, the same basic token count as a
double-area RGB mosaic.

The controlled-config validator rejects changes to the 53-frame, two-view,
V1-proper, full-episode, language-conditioned contract. CLI overrides remain
available for smoke tests, while the resolved checkpoint config records them.

## Experiment Matrix

| Group | Config | Representation and communication |
|---|---|---|
| MV0-V | `vgm_bridge_mv0_v_rgb_mosaic_53f.yaml` | Vertical RGB mosaic, one VAE/Wan path |
| MV0-H | `vgm_bridge_mv0_h_rgb_mosaic_53f.yaml` | Horizontal RGB mosaic, one VAE/Wan path |
| MV1 | `vgm_bridge_mv1_mosaic_view_embedding_53f.yaml` | RGB mosaic plus high/wrist band embedding |
| MV2 | `vgm_bridge_mv2_independent_vae_spatial_53f.yaml` | Independent VAE, then spatial latent mosaic |
| MV3 | `vgm_bridge_mv3_independent_vae_channel_53f.yaml` | Independent VAE, then channel concatenation |
| MV4 | `vgm_bridge_mv4_independent_shared_wan_53f.yaml` | Shared Wan applied independently to each view |
| MV5 | `vgm_bridge_mv5_independent_consistency_53f.yaml` | MV4 plus symmetric temporal-stage InfoNCE |
| MV6 | `vgm_bridge_mv6_joint_attention_53f.yaml` | Independent patchify/RoPE, joint global attention |
| MV7 | `vgm_bridge_mv7_cross_view_attention_53f.yaml` | Per-view Wan plus same-time cross-view adapters |
| MV8 | `vgm_bridge_mv8_scene_tokens_53f.yaml` | Per-time shared scene-token bottleneck |
| MV9 | `vgm_bridge_mv9_endpoint_scene_context_53f.yaml` | First/goal scene context injected by cross-attention |
| MV10 | `vgm_bridge_mv10_high_to_wrist_53f.yaml` | Generate high, then condition wrist on high trajectory |

Capacity and parameter controls:

- `vgm_bridge_mv0_v_rgb_mosaic_adapter_control_53f.yaml`: MV1-style view ID plus
  plain zero-init adapters, parameter-matched to MV7 but without view communication.
- `vgm_bridge_mv4_independent_adapter_control_53f.yaml`: native-view parameter control
  matched to MV7, without cross-view communication.
- `vgm_bridge_mv4_independent_separate_wan_53f.yaml`: independent high/wrist Wan
  backbones as a capacity upper bound.
- `vgm_bridge_mv6_joint_attention_view_adapter_53f.yaml`: MV6 plus view-specific
  residual bottleneck adapters.
- `vgm_bridge_mv7_cross_view_attention_view_adapter_53f.yaml`: MV7 plus view-specific
  residual bottleneck adapters.
- `vgm_bridge_mv7_cross_view_attention_separate_wan_53f.yaml`: separate Wan backbones
  with shared same-time cross-view adapters.

MV5 forms negatives only from different latent times within the same episode.
Other episodes are not negatives because that would encourage task-identity
classification rather than high/wrist stage alignment.

MV7 inserts adapters after Wan blocks `[2,6,10,14,18,22,26,29]`. Each adapter only
attends to the other camera at the same latent time. Its output projection is zero
initialized, so the initial function is exactly the independent per-view path.

MV10 uses teacher-forced clean high-view trajectory context during training and a
generated high-view trajectory during inference. This exposure gap is an explicit
risk of the sequential baseline, not hidden by the implementation.

MV9/MV10 context patches carry the same learned camera IDs as query tokens plus a
fixed ordered token-position embedding before the zero-initialized cross-attention
adapter. First/goal and generated-high context therefore remain camera- and
position-addressable instead of becoming an unordered patch set.

## Training

Use standard two-process DDP. Shared models use per-rank batch 4. Separate-Wan
configs already override this to per-rank batch 1 with four accumulation steps;
both settings keep global batch 8.

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus-stage1-multiview
CUDA_VISIBLE_DEVICES=0,1 /mnt/workspace/users/niejunnan/envs/motus/bin/accelerate launch \
  --num_processes 2 \
  --main_process_port 29671 \
  train/train_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_mv7_cross_view_attention_53f.yaml
```

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus-stage1-multiview
CUDA_VISIBLE_DEVICES=0,1 /mnt/workspace/users/niejunnan/envs/motus/bin/accelerate launch \
  --num_processes 2 \
  --main_process_port 29672 \
  train/train_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_mv7_cross_view_attention_separate_wan_53f.yaml
```

Do not add `--deepspeed` to this experiment matrix. Real first-update probes on
MV4, MV5, and MV7 showed finite raw gradients but NaN model parameters after the
tested ZeRO-2 optimizer step. Standard DDP completed synchronized updates for both
the 5B shared and approximately 10B separate-Wan variants, so the unvalidated
ZeRO-2 config was removed rather than published as a training option.

For another seed, override both identity and run name:

```bash
... train/train_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_mv7_cross_view_attention_53f.yaml \
  --seed 1 \
  --run_name vgm_bridge_mv7_cross_view_attention_53f_2gpu_gbs8_50k_s1
```

The dataset maps sampler indices to deterministic episode/window draws, while
training noise, timestep draws, and dropout use rank-specific RNG streams after
DDP initialization. This avoids the old failure mode where different ranks ignored
their sampler indices and replayed the same random episode sequence.

## Evaluation

Create the first window manifest once, then reuse it for every checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/workspace/users/niejunnan/envs/motus/bin/python \
  train/eval_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_mv0_v_rgb_mosaic_53f.yaml \
  --checkpoint /absolute/path/to/checkpoint_step_50004 \
  --output_dir eval_outputs/multiview_stage1 \
  --num_samples 16 \
  --num_inference_steps 50 \
  --seed 0
```

For subsequent groups, add:

```bash
--windows_file /absolute/path/to/first/evaluation/windows.json
```

The window fingerprint excludes machine-specific absolute paths and hashes the
episode, task, frame indices, and episode length. Keep evaluation batch size fixed;
it is recorded in `metrics.json` together with seed and denoising steps.

Every sample writes all 53 frames, without temporal subsampling:

```text
sample_NNN_sheet.png / sample_NNN_gt_pred.mp4
sample_NNN_high_sheet.png / sample_NNN_high_gt_pred.mp4
sample_NNN_wrist_sheet.png / sample_NNN_wrist_gt_pred.mp4
sample_NNN_high_similarity_53x53.png
sample_NNN_wrist_similarity_53x53.png
```

The sheet has a complete GT row and complete prediction row with readable row and
slot labels. Metrics include:

- full, middle, generated-region, and endpoint MSE/PSNR/MAE;
- high/wrist metrics and view-macro values;
- motion-, edge-, and interaction-weighted pixel errors;
- cross-view motion-profile, peak-time, and retrieved-slot synchronization;
- 53x53 slot alignment, backward jumps, large jumps, collisions, and endpoint error;
- optional local DINO/DINOv2 and LPIPS metrics.

V1-proper does not hard-clamp the goal latent. The goal is an input condition, but
its output frame is generated; therefore endpoint and generated-region metrics
include it. Only legacy V0 hard-clamped tail frames are excluded.

Gripper, object, and contact-state consistency still requires a trusted detector
or role-mask annotation. The evaluator does not invent a proxy label; use the full
per-view sheets for manual review or enable an annotated role-mask evaluation.

## Stage 2 Contract

The cache stores:

```text
trajectory_latent         canonical 14-slice composite latent for Layerwise replay
trajectory_view_latent    native [V,C,T,H,W] latent for diagnosis/future replay
trajectory_frame_latents  53 independently encoded canonical RGB slots for Serial
first_view_frames / last_view_frames
multiview mode/layout, checkpoint, seed, and resolved config
```

Native-view RGB is decoded once and used directly to build the 53 Serial memory
slots. It is not re-decoded from a composite VAE latent, which would add avoidable
blur only to MV2-MV10.

All groups can be compared by regenerating one cache per Stage 1 checkpoint and
retraining the same Serial Stage 2 configuration. Never reuse a cache across
different Stage 1 checkpoints.

Current Layerwise replay supports mosaic paths MV0/MV1 and the mosaic adapter
control. MV2-MV10 fail explicitly because their hidden sequence has a native camera
dimension. Replaying only the canonical composite latent would silently evaluate a
different model; native multi-view Layerwise support belongs in a separate Stage 2
change.

## Verification

- Final-review standard DDP shared MV7: real Wan2.2-TI2V-5B, two H200 GPUs,
  per-rank batch 4, one synchronized optimizer update, loss `1.1132`, `11.29s`.
- Real MV3 channel-fusion smoke expanded Wan input/output to `196/96` channels and
  completed a synchronized update with loss `1.1766` in `10.97s`.
- Second-review MV9 endpoint-context smoke, with explicit context camera/position
  identity, completed a synchronized update with loss `1.0315` in `10.94s`.
- Second-review MV10 sequential high-to-wrist smoke, with explicit context
  camera/position identity, completed a synchronized update with loss `1.0315`
  in `11.48s`.
- Standard DDP separate MV7: approximately 10.016B trainable parameters, two H200
  GPUs, per-rank batch 1 with accumulation 4, synchronized update passed, loss
  `1.3811`, approximately `11.41s`.
- Second review: `154 passed`, including complete multi-view cache write/reuse,
  strict checkpoint state round trips, the explicit 53-frame contact-sheet
  regression, and CUDA forward/backward/sampling coverage for all native-view modes.
