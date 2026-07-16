# PushT Endpoint-Conditioned VGM Multimodality Results

Date: 2026-07-16

## Executive Conclusion

The data supports the premise that nearby PushT endpoint pairs admit multiple
expert path modes, but the current 5K-step VGM does not reproduce that
multimodality reliably.

For the four held-out endpoint conditions, the 32 nearest training endpoint
neighbors contain an average of `3.75/4` contact-side modes. In contrast, the
50-step VGM has `0/32` strict-valid samples. Under the weaker post-hoc task tier,
`12/32` samples pass, but every held-out condition contains only one valid
contact-side mode. Average valid-mode coverage is only `27.1%`.

Official-simulator replay makes the failure clearer. The learned inverse
dynamics recovers held-out real actions with only `0.54 px` coordinate RMSE, and
full real pusher paths reach their conditioned final states in `28/28` episodes.
No generated sample has a complete 17-frame pusher track. After allowing a
relaxed interpolation of at most two consecutive missing detections, only
`12/32` samples can be replayed, and `0/12` reaches the conditioned final state.

Therefore the current checkpoint shows seed-dependent variation inside one
dominant path mode, not multiple physically valid successful paths. The proposed
VGM trajectory-memory benefit remains a plausible research hypothesis, but this
experiment does not yet establish it.

## Traceability

- Repository: `/Users/n/Documents/DreamZero/worktrees/Motus-pusht`
- Branch: `codex/pusht-vgm-diversity`
- Experiment base commit: `946e1ad8a04d1c103474f1beaa1efe7ffd4fd984`
- Initial PushT implementation commit: `a534033`
- Initial report commit: `c4346ed`
- Corrected validity/mode/physics code commit: `5e9f9b3`
- Remote code: `/mnt/workspace1/users/niejunnan/codebase/Motus-pusht`
- Official simulator source: Diffusion Policy commit
  `5ba07ac6661db573af695b419a7947ecb704690f`

Training checkpoint:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus-pusht/checkpoints/
vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k/
vgm_bridge_pusht_v1_proper_256x256_17f_7gpu_gbs14_5k/checkpoint_step_5000
```

The formal solver-step set is now exactly:

```text
1, 4, 10, 20, 50
```

The existing 55-step raw run is archival only and is excluded by
`summary_max50/summary_manifest.json`.

## Data Multimodality

The converted dataset contains 206 demonstrations and 25,650 frames, split into
178 training and 28 held-out episodes. These should not be called official
PushT successes: recorded actions reproduce each demonstration's final state,
but none of the 28 test trajectories crosses the official simulator's 95%
coverage termination threshold.

The training data nevertheless contains clear local path diversity:

| Endpoint-neighborhood audit | Result |
| --- | ---: |
| Train episodes with at least 2 modes among 8 nearest train endpoints | 177/178 |
| Train episodes with at least 3 modes among 8 nearest train endpoints | 145/178 |
| Held-out queries with at least 2 train modes among 8 nearest endpoints | 28/28 |
| Held-out queries with at least 3 train modes among 8 nearest endpoints | 26/28 |

For the four generated conditions, using a larger 32-neighbor local reference:

| Condition | Nearest-train modes | Effective train modes |
| ---: | ---: | ---: |
| 00 | 3/4 | 2.82 |
| 01 | 4/4 | 3.60 |
| 02 | 4/4 | 3.77 |
| 03 | 4/4 | 3.34 |

These are nearby, not exactly pixel-identical, endpoint pairs. They establish
that the local dataset contains multiple routing/contact modes, not that every
exact endpoint image pair has multiple demonstrations.

## Training Health

- Model: Wan2.2 TI2V 5B through Motus V1-proper endpoint conditioning.
- Output: 17 frames, with 15 generated interior frames.
- Trainable parameters: 5,001,044,160.
- Hardware: seven H200 GPUs.
- Per-GPU/global batch: 2/14.
- Budget: 5,000 optimizer steps, BF16, learning rate `1e-5`.
- Duration: 2,461.91 seconds, about 41 minutes.
- Final instantaneous loss: 0.0517.
- Final 100-step mean loss: 0.050171.

There was no NaN, OOM, DDP mismatch, or optimization divergence. Standard
held-out full-image metrics also appear reasonable (`MSE=0.002956`,
`PSNR=25.428 dB`). These global metrics are dominated by the static background
and large T block and do not expose loss of the small blue pusher.

## Corrected Solver Ablation

The previous report compared cardinal training modes with diagonal generated
quadrants. Commit `5e9f9b3` rescored both sides with the same cardinal
first-contact definition. This changes the interpretation materially.

| Steps | Strict valid | Post-hoc task valid | Mean pusher detection | Mean task-valid modes/condition | Task-valid pusher APD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0% | 0.0% | 9.2% | 0.00 | N/A |
| 4 | 0.0% | 0.0% | 25.4% | 0.00 | N/A |
| 10 | 0.0% | 12.5% | 51.5% | 0.75 | 0.0721 |
| 20 | 0.0% | 25.0% | 57.4% | 0.75 | 0.1093 |
| 50 | 0.0% | 37.5% | 60.8% | 1.00 | 0.1163 |

At 50 steps, post-hoc task-valid sample counts are `4/8, 3/8, 1/8, 4/8`
for conditions 00-03. In every condition, all passing samples share the same
contact-side mode:

```text
C00: up
C01: up
C02: left
C03: left
```

The pusher APD is nonzero, so seeds do produce different curves. However, APD
here measures variation within one contact mode and also remains sensitive to
interpolated missing detections. It cannot be cited as successful multimodality.

Mode-distribution diagnostics at 50 steps:

| Condition | Train modes | Task-valid VGM modes | Mode coverage | Mode JS divergence |
| ---: | ---: | ---: | ---: | ---: |
| 00 | 3 | 1 | 33.3% | 0.233 |
| 01 | 4 | 1 | 25.0% | 0.251 |
| 02 | 4 | 1 | 25.0% | 0.406 |
| 03 | 4 | 1 | 25.0% | 0.533 |

## Physical Replay

The official PushT action is an absolute 2D pusher target. A shared linear
inverse-dynamics model is fitted on the 178 training trajectories. On all 28
held-out trajectories it achieves:

| Inverse-action metric | Result |
| --- | ---: |
| Coordinate RMSE | 0.541 px |
| Coordinate MAE | 0.311 px |
| Mean 2D vector error | 0.495 px |
| 95th-percentile 2D vector error | 1.475 px |

This is accurate enough to use as a replay diagnostic. The real-path controls
show where the proxy loses information:

| Replay source, 28 held-out episodes | Official success | Conditioned-goal reached | Mean max reward | Mean final goal position error |
| --- | ---: | ---: | ---: | ---: |
| Recorded actions | 0.0% | 100.0% | 0.8849 | approximately 0 px |
| Inverse actions from full real pusher path | 0.0% | 100.0% | 0.8834 | 0.17 px |
| Inverse actions from 17 real waypoints | 0.0% | 67.9% | 0.6933 | 31.28 px |

The official-success column is zero even for recorded actions because this
dataset's demonstration endpoints do not cross the environment's stricter 95%
coverage termination threshold. Conditioned-goal arrival is therefore the fair
primary endpoint metric for this dataset.

Generated replay results:

| Generated-path test | Result |
| --- | ---: |
| Complete pusher tracks, all 17 frames | 0/32 |
| Relaxed recoverable tracks, gap at most 2 frames | 12/32 |
| Conditioned-goal reached among recoverable | 0/12 |
| Official success among recoverable | 0/12 |
| Mean max official reward among recoverable | 0.2076 |
| Mean final conditioned-goal position error | 165.95 px |
| Simulator pusher vs generated pusher RMS | 0.00034 normalized |
| Simulator block vs generated block RMS | 0.1503 normalized |

The final two rows are especially informative. The replay controller follows the
recovered generated pusher path closely, but the physically simulated T block
does not follow the block motion shown by the generated video. This is direct
evidence of interaction-dynamics inconsistency, not merely a weak inverse
controller.

## Visual Evidence

Local root:

```text
/Users/n/Documents/DreamZero/artifacts/pusht_vgm_multimodal_20260716/full
```

Training distribution and four-condition mode summary:

```text
mode_diagnostics_50step/training_and_inference_mode_atlas.png
```

All four inference path panels:

```text
mode_diagnostics_50step/inference_50step_path_atlas.png
```

Per-condition detailed comparisons, including endpoint images, 32 nearest
training paths, all eight generated paths, mode occupancy, PCA, and metrics:

```text
mode_diagnostics_50step/by_condition/condition_00_mode_diagnostic.png
mode_diagnostics_50step/by_condition/condition_01_mode_diagnostic.png
mode_diagnostics_50step/by_condition/condition_02_mode_diagnostic.png
mode_diagnostics_50step/by_condition/condition_03_mode_diagnostic.png
```

Physical replay summary:

```text
physical_replay_50step/physical_replay_summary.png
```

Every physical replay sheet retains all 17 temporal frames for the recorded
trajectory, the real 17-waypoint replay, and each of eight generated seeds:

```text
physical_replay_50step/by_condition/condition_00_physics_replay_17f.png
physical_replay_50step/by_condition/condition_01_physics_replay_17f.png
physical_replay_50step/by_condition/condition_02_physics_replay_17f.png
physical_replay_50step/by_condition/condition_03_physics_replay_17f.png
```

Remote root:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus-pusht/eval_outputs/
pusht_vgm_multimodal_step5000_20260716
```

## What This Says About VGM Necessity

The intended reviewer-facing reason for a VGM is not that it can encode a
trajectory. A normal video encoder can already encode a provided trajectory.
The VGM is useful only if it amortizes a conditional distribution over plausible
trajectories and can synthesize several valid modes for a novel first/goal pair
without storing demonstrations for every endpoint combination.

This experiment separates the premise from the current implementation:

1. The premise is plausible: local PushT endpoint neighborhoods contain several
   real path modes.
2. The current VGM does not realize it: generated valid paths collapse to one
   contact mode and are not physically consistent.
3. Therefore it is not yet defensible to claim that VGM memory is superior to a
   retrieved real trajectory memory.

Even after generation quality improves, the decisive downstream control remains:

```text
Real-1 vs Retrieved-K vs VGM-1 vs VGM-K vs Oracle same-episode memory
```

The VGM claim becomes convincing only if `VGM-K` improves unseen-endpoint mode
coverage, Progress stability, or downstream reward while using less stored
trajectory memory or lower online latency than `Retrieved-K`.

## Next Experiment

The immediate bottleneck is fidelity and physical interaction, not the number
of denoising steps. The next training iteration should keep this exact split and
evaluation protocol while changing only the trajectory model:

1. Add deterministic object-centric supervision for the blue pusher and T block,
   with substantially higher weight on the small pusher.
2. Add an auxiliary pusher/block coordinate or mask prediction objective so the
   model cannot reduce global latent loss by dropping the manipulator.
3. Increase temporal resolution or predict a state/action-consistency signal;
   even real 17-waypoint replay reaches only 67.9% of conditioned endpoints.
4. Accept multimodality only after at least two strict-valid or physically
   goal-reaching modes appear for more than one held-out endpoint condition.
5. Then run the real-memory versus VGM-memory Progress ablation above.

One-, four-, or ten-step sampling of this undistilled checkpoint is not a route
to better multimodality. It mainly reduces rendering validity. The formal maximum
remains 50 denoising steps.
