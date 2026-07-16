# PushT Endpoint-Conditioned VGM Multimodality Validation

## Question

Can one endpoint-conditioned video generation model produce multiple valid PushT
paths for the same first and goal frames, including held-out start conditions?

This is a diagnostic for the proposed VGM trajectory-memory story. It does not
by itself prove that generated memory improves a downstream reward model. It
tests the prerequisite that a VGM can represent more than one useful trajectory
instead of reconstructing one memorized demonstration or emitting random noise.

## Data

- Source: official PushT replay dataset, 206 successful episodes and 25,650 frames.
- Train split: 178 episodes.
- Test split: 28 episodes from a held-out KMeans cluster over initial pusher and
  block pose. The split is by episode.
- Each episode is sampled uniformly into 17 frames, including the true first and
  last frame. Training jitters the 15 interior temporal bins; evaluation uses the
  fixed bin centers.
- Resolution: 256 x 256.

Local converted data:

```text
/Users/n/Documents/DreamZero/datasets/pusht_vgm_v1
```

Remote converted data:

```text
/mnt/workspace1/users/niejunnan/datasets/pusht_vgm_v1
```

## Model

- Base: Wan2.2 TI2V 5B used by the Motus VGM branch.
- Conditioning: V1-proper endpoint mode (`v1_mask_endpoint`).
- Input: one first frame and one last frame.
- Output: all 17 frames, with the 15 unknown interior frames generated.
- Trainable parameters: approximately 5.0B.
- Training: 7 H200 GPUs, per-GPU batch 2, global batch 14, 5,000 optimizer
  steps, BF16, learning rate 1e-5.

Training config:

```text
configs/vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k.yaml
```

## Controlled Evaluation

For four evenly spaced held-out endpoint conditions (dataset indices 0, 7, 14,
and 21), generate eight samples using seeds 0 through 7. Reuse the exact same
initial Gaussian noise seed across every solver-step setting:

```text
55, 50, 20, 10, 4, 1 denoising steps
```

Changing the number of denoising steps does not create stochasticity by itself.
For a fixed seed and deterministic scheduler, every setting is deterministic.
The seed selects the initial latent; solver steps control how accurately that
latent is transported through the learned velocity field.

Here, `1 step` means one full Euler update from the initial noise level to the
clean endpoint of the scheduler. It is not the same as taking only the first
small update of a 55-step schedule and decoding the still-noisy intermediate
latent.

## Metrics

The evaluator tracks the blue pusher and gray T block in all 17 generated frames.
Area, displacement, path-length, and contact-distance thresholds are calibrated
once from the 178 fixed real trajectories in the training split. The broad
endpoint and full-image rejection thresholds are fixed constants. Every threshold
is then applied unchanged to all held-out conditions and solver settings.

A sample is counted as valid only when all six checks pass:

1. Both entities are trackable in at least 16 of 17 frames and have plausible
   rendered area.
2. Pusher and block frame-to-frame displacement stay below calibrated real-data
   limits.
3. The block exhibits nontrivial motion rather than remaining static.
4. During block motion, the rendered pusher stays near the T-block surface at a
   distance consistent with real PushT trajectories.
5. Full-frame error stays below a broad fixed threshold that rejects globally
   broken renders while permitting different moving-object paths.
6. The generated first and final pusher/block positions remain close to both
   conditioned endpoint frames.

The report also includes a post-hoc secondary task-level diagnostic. It requires the T
block to be trackable in at least 16/17 frames, the pusher to be trackable in at
least 12/17 frames with plausible area, all motion/contact/smoothness checks to
pass, the first frame to match, and the final T-block position to match the goal.
It intentionally does not require the final pusher position to match. The 12/17
pusher threshold is held constant across all solver settings, and the continuous
pusher-detection rate is reported alongside it. This is a useful PushT task
diagnostic, but it cannot replace the primary complete-path validity result.

Diversity is then measured only among valid samples:

- pairwise RMS distance between 17-slot pusher paths;
- pairwise RMS distance between block paths;
- number and entropy of approach sectors around the T block.

Strict and post-hoc task-level samples are aggregated separately. A diversity
number is never computed from samples that fail the corresponding validity tier.

Pixel MSE to the single recorded ground-truth path is reported as a diagnostic,
not as the primary criterion. A different valid path can legitimately have high
MSE to that one demonstration.

Every detailed sample sheet contains all 17 ground-truth and all 17 generated
frames. The summary additionally creates one 17-column overview per endpoint
condition and solver setting, with ground truth followed by all eight seeds; no
temporal frames are dropped from either view.

This is an image-space proxy for trajectory validity. It does not recover actions,
replay generated paths in the PushT simulator, or explicitly estimate the gray
T-block orientation. Simulator success and orientation-aware mask matching remain
stronger follow-up tests.

## Interpretation Rules

Evidence supports useful VGM multimodality only if the same endpoint condition
has at least two valid path modes, with controlled endpoint error, and this occurs
for more than one held-out condition.

The following do not support the claim:

- different seeds that only change texture or produce broken geometry;
- high pairwise distance caused by temporal jumps;
- samples that miss the conditioned goal;
- modes observed only on training episodes;
- higher apparent diversity obtained by under-denoising an undistilled model.

The expected solver ablation is a validity-diversity Pareto curve. If 1-step
sampling is substantially less valid, it means the original model is not a
one-step generator; it does not mean that one-step flow models are impossible.
Testing true one-step inference would require distillation, consistency training,
or reflow.

## Reproduction

Training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
/mnt/workspace/users/niejunnan/envs/motus/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=7 \
  train/train_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k.yaml \
  --report_to none
```

One solver-step evaluation, using one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/workspace/users/niejunnan/envs/motus/bin/python \
  scripts/evaluate_pusht_vgm_multimodal.py \
  --config configs/vgm_bridge_pusht_v1_proper_17f_eval.yaml \
  --checkpoint /absolute/path/to/checkpoint_step_5000 \
  --output_dir eval_outputs/pusht_vgm_multimodal/steps_050 \
  --num_inference_steps 50 \
  --episode_indices 0,7,14,21 \
  --seeds 0,1,2,3,4,5,6,7
```

Aggregate all completed `steps_*` runs:

```bash
/mnt/workspace/users/niejunnan/envs/motus/bin/python \
  scripts/summarize_pusht_vgm_multimodal.py \
  --input_root eval_outputs/pusht_vgm_multimodal \
  --output_dir eval_outputs/pusht_vgm_multimodal/summary
```

Run all six solver settings in parallel on GPUs 0 through 5:

```bash
bash scripts/run_pusht_vgm_solver_ablation.sh \
  /absolute/path/to/checkpoint_step_5000 \
  eval_outputs/pusht_vgm_multimodal
```
