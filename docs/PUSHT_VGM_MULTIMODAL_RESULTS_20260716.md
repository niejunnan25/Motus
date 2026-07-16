# PushT Endpoint-Conditioned VGM Multimodality Results

Date: 2026-07-16

## Executive Conclusion

The experiment provides partial, but not yet sufficient, evidence for the VGM
trajectory-memory story. Different initial-noise seeds produce visibly different
pusher approaches for the same held-out first/goal frames, and these differences
remain after 50 or 55 denoising steps. However, strict complete-path validity is
0/32 at every solver setting because the small blue pusher often becomes blurry,
fragments, or disappears in later frames.

Under the explicitly weaker post-hoc task-level diagnostic, 50 and 55 steps each produce
12/32 task-valid samples (37.5%). Three of four held-out endpoint conditions have
two task-valid approach sectors. This supports the claim that the model contains
candidate path diversity, but it does not yet establish a high-fidelity multimodal
trajectory generator suitable for cached visual memory.

## Question and Decision Rule

The controlled question is:

> For identical held-out first and goal frames, can one endpoint-conditioned VGM
> produce at least two valid robot-object interaction paths by changing only the
> initial latent-noise seed?

Useful VGM multimodality requires both validity and diversity. Pixel differences,
texture changes, broken geometry, temporal jumps, or failure to reach the goal do
not count as alternative paths. The test is a prerequisite for the VGM-memory
story, not a downstream proof that generated memory beats real trajectory memory.

Here, validity is an image-space proxy: color masks measure visibility, centroid
motion, contact distance, and endpoint agreement. It does not recover actions,
replay a generated path in the simulator, or explicitly score T-block orientation.

## Traceability

- Repository worktree: `/Users/n/Documents/DreamZero/worktrees/Motus-pusht`
- Branch: `codex/pusht-vgm-diversity`
- Experiment base commit: `946e1ad8a04d1c103474f1beaa1efe7ffd4fd984`
- Remote code: `/mnt/workspace1/users/niejunnan/codebase/Motus-pusht`
- Training config: `configs/vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k.yaml`
- Evaluation config: `configs/vgm_bridge_pusht_v1_proper_17f_eval.yaml`
- Protocol: `docs/PUSHT_VGM_MULTIMODAL_VALIDATION.md`

Final checkpoint:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus-pusht/checkpoints/
vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k/
vgm_bridge_pusht_v1_proper_256x256_17f_7gpu_gbs14_5k/checkpoint_step_5000
```

## Data

- Official PushT replay dataset: 206 successful episodes, 25,650 frames.
- Train/test split: 178/28 episodes; test is a held-out initial-pose cluster.
- Each episode is sampled into 17 uniformly spaced frames, including true first
  and last frames; training jitters the 15 interior bins.
- Resolution: 256 x 256.
- Full-dataset endpoint audit: all 206 eight-nearest endpoint neighborhoods have
  at least two approach sectors; 174/206 have at least three.
- Train-only audit: 177/178 training endpoint neighborhoods have at least two
  sectors; 145/178 have at least three.
- Held-out-to-train audit: all 28 test endpoint queries have at least two sectors
  among their eight nearest training endpoints; 26/28 have at least three.
- These are nearby endpoint pairs, not exactly pixel-identical endpoint conditions.

Local dataset audit:

```text
/Users/n/Documents/DreamZero/datasets/pusht_vgm_v1/audit
```

## Training

- Model: Wan2.2 TI2V 5B through Motus `V1-proper` mask-guided endpoint mode.
- Conditions: first frame and last frame; no language or state condition.
- Output: 17 frames, with 15 unknown interior frames.
- Trainable parameters: 5,001,044,160.
- Hardware: cdf GPUs 0-6, seven H200 GPUs.
- Per-GPU/global batch: 2/14.
- Budget: 5,000 optimizer steps, BF16, learning rate 1e-5.
- DDP gradient synchronization was explicitly verified across all seven ranks.
- Duration: 2,461.91 seconds, about 41.0 minutes.
- Median step time excluding the first step: 0.44 seconds.
- Final instantaneous loss: 0.0517.
- Final 100-step mean loss: 0.050171.

Training artifacts:

```text
/Users/n/Documents/DreamZero/artifacts/pusht_vgm_multimodal_20260716/training
```

The loss falls quickly and reaches a stable plateau after roughly 1,000 steps.
There is no NaN, OOM, rank failure, or optimization divergence. The later pusher
failure is therefore a visual-detail/fidelity limitation, not a failed optimizer.

## Standard Held-Out Reconstruction

The normal 50-step evaluator uses 16 held-out samples:

| Metric | Result |
| --- | ---: |
| Flow-matching eval loss | 0.069529 +/- 0.040104 |
| Generated MSE | 0.002956 |
| Generated PSNR | 25.428 dB |
| Generated MAE | 0.009954 |
| Generated edge MSE | 0.007079 |
| Generated interaction MSE | 0.006657 |
| Generated motion MSE | 0.069903 |

These full-image metrics look reasonable because the white background and large
T block dominate the image. They do not expose the small-pusher disappearance,
which is why entity-level validity is required.

## Multiseed Solver Ablation

Four fixed held-out endpoint conditions are each generated with seeds 0-7. The
same seed is reused for every solver setting.

| Steps | Strict valid | Post-hoc task valid | Mean pusher detection | Mean task-valid modes/condition | Task-valid pusher RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0% | 0.0% | 9.2% | 0.00 | N/A |
| 4 | 0.0% | 0.0% | 25.4% | 0.00 | N/A |
| 10 | 0.0% | 12.5% | 51.5% | 0.75 | 0.0664 |
| 20 | 0.0% | 25.0% | 57.4% | 1.00 | 0.1216 |
| 50 | 0.0% | 37.5% | 60.8% | 1.75 | 0.1146 |
| 55 | 0.0% | 37.5% | 61.0% | 1.75 | 0.1150 |

At 50 steps, task-valid samples by condition are `4/8, 3/8, 1/8, 4/8`.
Conditions 00, 01, and 03 each contain two task-valid approach sectors; condition
02 contains only one task-valid sample and is the clearest failure case.

Full local outputs:

```text
/Users/n/Documents/DreamZero/artifacts/pusht_vgm_multimodal_20260716/full
```

The `summary/full_17f_overviews` directory contains one image per condition and
solver setting. Each row contains all 17 frames, and every overview includes the
ground truth followed by all eight seeds. Individual MP4, NPZ tracks, metrics,
and 17-frame GT/generated sheets remain under each `steps_*` directory.

## Determinism and the One-Step Hypothesis

For a fixed checkpoint, condition, seed, and deterministic scheduler, sampling is
deterministic. Repeating the four-step smoke run for seeds 0 and 1 produced arrays
that are byte-identical to the formal four-step run.

Changing the seed changes the initial Gaussian latent. It is the seed, not a low
solver-step count, that selects another candidate trajectory. A 55-step result is
therefore deterministic conditional on its seed, but the distribution across
seeds can remain multimodal.

The measurements contradict the hypothesis that fewer undistilled steps preserve
more useful modes:

- Across-seed frame MSE rises from 0.000154 at one step to 0.001898 at 50 steps.
- Secondary task validity rises from 0% at one/four steps to 37.5% at 50/55.
- Same-seed 50-vs-55 frame MSE is only 0.00000242, showing solver convergence.
- Same-seed 1-vs-50 frame MSE is 0.001172, showing substantial under-denoising
  error rather than a clean uncertainty representation.

One Euler update from the maximum noise level to the clean endpoint is not a
trained one-step generator. A genuine one-step model would require consistency
training, reflow, or distillation. An early noisy latent may contain uncertainty,
but it is not directly a clean trajectory-memory representation.

## Failure Analysis

The principal defect is the pusher, not the T block or the endpoint clamp:

- T-block detection is 100% in all solver settings and its mean final goal error
  is approximately 0.013-0.016 in normalized image coordinates.
- At 50/55 steps, pusher detection averages only about 61% of frames.
- Human inspection confirms the tracker result: the blue pusher becomes faint,
  splits into small blobs, or disappears, especially near the final frames.
- Full-image latent loss underweights this small, task-critical object. The model
  can obtain low global MSE while losing the manipulation agent.

This means the present checkpoint demonstrates a multimodal tendency but does not
yet satisfy the original strict statement: multiple complete, physically usable
paths for the same endpoints.

## Next Decisive Experiment

Keep the endpoint split, four conditions, eight seeds, solver settings, and all
thresholds unchanged. Retrain only the fidelity objective:

1. Add object-centric weighting from PushT's deterministic color masks, with
   separate pusher and T-block weights instead of generic full-frame motion/edge
   weighting.
2. Report strict validity first; accept the multimodality claim only when at least
   two strict-valid sectors occur for more than one held-out condition.
3. Compare 50 and 55 steps; use 50 if quality remains equivalent.
4. After strict validity succeeds, compare downstream memory sources: one real
   demonstration, K retrieved real demonstrations, one generated trajectory, and
   K generated trajectories. This is the control needed to establish that VGM
   coverage, rather than trajectory matching alone, provides the benefit.

For online Progress inference, cache K high-quality generated trajectories once
and marginalize or select over them. Reducing an undistilled model to one or four
steps is not supported by this experiment.
