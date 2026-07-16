# PushT Endpoint-Conditioned VGM Multimodality Validation

## Core Question

For one held-out PushT first/goal image pair, can an endpoint-conditioned VGM
generate multiple trajectories that are all visually valid, supported by the
training distribution, physically executable, and successful at the conditioned
goal?

This is the prerequisite for using VGM-generated trajectories as Stage-2
Progress memory. Pixel differences across seeds are not enough. A useful VGM
mode must preserve the pusher and T block, follow a coherent interaction path,
and reach the same endpoint through a meaningfully different contact or routing
mode.

The experiment does not by itself prove that generated memory is better than
retrieved real memory. That claim requires a later downstream control comparing
`Real-1`, `Retrieved-K`, `VGM-1`, and `VGM-K` under the same Progress matcher.

## Data and Split

- Source: the official PushT replay dataset used by
  [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
- Converted data: 206 demonstrations and 25,650 frames.
- Train/test split: 178/28 episodes.
- Test episodes come from a held-out KMeans cluster over initial pusher and
  block pose, so the split is by episode and initial-condition region.
- Every episode is represented by 17 uniformly spaced frames, including the
  true first and final frame. Training jitters the 15 interior temporal bins;
  evaluation uses deterministic bin centers.
- Resolution: 256 x 256.

The 206 demonstrations must not be described as official-environment successes.
Replaying all 28 held-out recorded action sequences in the official simulator
reaches their recorded final states, but none crosses the simulator's stricter
95% coverage termination threshold. The conditioned final demonstration state
is therefore the primary endpoint target, while official PushT reward and
success are reported separately.

Local data:

```text
/Users/n/Documents/DreamZero/datasets/pusht_vgm_v1
```

Remote data:

```text
/mnt/workspace1/users/niejunnan/datasets/pusht_vgm_v1
```

## Model and Training

- Base: Wan2.2 TI2V 5B through the Motus VGM branch.
- Conditioning: V1-proper mask-guided endpoint mode (`v1_mask_endpoint`).
- Input: one first frame and one final frame.
- Output: a 17-frame trajectory with 15 generated interior frames.
- Trainable parameters: approximately 5.0B.
- Training: 7 H200 GPUs, per-GPU batch 2, global batch 14, 5,000 optimizer
  steps, BF16, learning rate 1e-5.

Training config:

```text
configs/vgm_bridge_pusht_v1_proper_17f_7gpu_gbs14_5k.yaml
```

## Controlled Generation

Four evenly spaced held-out endpoint conditions use dataset indices `0,7,14,21`.
For each condition, generate eight samples with seeds `0..7`. The same condition
and seed are reused for each solver setting.

The formal denoising-step ablation is capped at 50 steps:

```text
1, 4, 10, 20, 50
```

The previous 55-step run was caused by a wording mistake. Its raw files can
remain archived, but the summarizer excludes it from all formal tables and
figures. A fixed seed remains deterministic; changing the seed changes the
initial Gaussian latent, while solver steps change numerical integration
quality. Under-denoising an undistilled model is not treated as a principled
multimodal representation.

## Common Path-Mode Definition

Training and generated trajectories use exactly the same four interpretable
contact-side modes:

```text
right, down, left, up
```

For a real trajectory, the mode is the pusher-minus-block direction at the first
simulator contact; if contact metadata is absent, the closest frame is used as a
fallback. For a generated video, color masks locate the pusher and block, and the
first frame within the calibrated surface-contact distance is used. A closest
frame fallback is recorded but is not silently promoted to contact evidence.

This replaces the previous inconsistent comparison in which training paths used
four cardinal sectors while generated paths used four diagonal quadrants. All
stored NPZ trajectories are re-scored with the common definition, so the model
does not need to be regenerated.

For each held-out endpoint pair, the 32 nearest training endpoint pairs provide
the local reference distribution. The endpoint descriptor contains first and
final pusher position, block position, and block angle encoded as sine/cosine.

## Image-Space Validity

The strict primary validity tier requires:

1. The pusher and T block are detected in at least 16 of 17 frames with
   plausible area.
2. Pusher and block motion are temporally smooth under real-data-calibrated
   thresholds.
3. The block exhibits nontrivial motion.
4. Pusher and block motion remain contact-coupled.
5. The rendered video is not globally corrupted.
6. Both first and final pusher/block positions match the conditioned endpoints.

A secondary post-hoc task tier keeps the block and motion checks but allows the
pusher to be visible in only 12 of 17 frames and does not require its final
position to match. This tier is useful for diagnosis but cannot establish a
complete usable trajectory.

Diversity is measured only after the corresponding validity filter:

- distinct contact-side mode count and entropy;
- effective mode count, `exp(entropy)`;
- average pairwise pusher-path RMS (APD);
- nearest-training-path RMS;
- mode coverage and precision against the 32-neighbor training distribution;
- Jensen-Shannon divergence between training and generated mode occupancy.

## Physical Replay Calibration

The VGM outputs observations, not actions. Physical replay is therefore a proxy
with an explicit calibration chain rather than a direct policy evaluation.

The evaluator uses the official Diffusion Policy PushT simulator at commit
`5ba07ac6661db573af695b419a7947ecb704690f` with `legacy=False`. It fits one
shared linear inverse-dynamics map on the 178 training episodes:

```text
action_t = b + w_prev * pusher_(t-1)
             + w_now  * pusher_t
             + w_next * pusher_(t+1)
```

It then evaluates four replay levels:

1. Recorded expert actions.
2. Actions recovered from the full real pusher-state trajectory.
3. Actions recovered from 17 real pusher waypoints with linear interpolation.
4. Actions recovered from generated 17-frame pusher tracks.

Generated tracks must contain both endpoints. Strict recovery requires all 17
pusher detections. A separately labeled relaxed diagnostic allows linear
interpolation only when no missing run exceeds two frames. Long gaps and missing
endpoints remain unrecoverable.

The replay reports both official PushT reward/success and arrival at the
conditioned final demonstration state. Conditioned-goal arrival requires final
block position error at most 20 pixels and angle error at most 15 degrees.

## Decision Rules

The current VGM multimodality hypothesis is supported only when:

1. More than one held-out endpoint condition contains at least two distinct
   strict-valid or physically goal-reaching modes.
2. Generated modes are supported by the local training distribution rather than
   being out-of-manifold corruption.
3. Diversity is not explained by missing objects, temporal jumps, or solver
   under-denoising.

The necessity of the VGM for Stage-2 Progress requires an additional downstream
test. With the same matcher, query set, and compute budget, compare:

```text
Real-1 vs Retrieved-K vs VGM-1 vs VGM-K vs Oracle same-episode memory
```

The VGM is justified only if it amortizes a conditional trajectory distribution
for novel endpoint pairs and gives better mode coverage, Progress robustness, or
memory/latency trade-offs than storing and retrieving real demonstrations.

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

Run the five formal solver settings in parallel:

```bash
bash scripts/run_pusht_vgm_solver_ablation.sh \
  /absolute/path/to/checkpoint_step_5000 \
  eval_outputs/pusht_vgm_multimodal
```

Rebuild a max-50 summary from existing raw outputs:

```bash
/mnt/workspace/users/niejunnan/envs/motus/bin/python \
  scripts/summarize_pusht_vgm_multimodal.py \
  --input_root eval_outputs/pusht_vgm_multimodal \
  --output_dir eval_outputs/pusht_vgm_multimodal/summary_max50 \
  --steps 1,4,10,20,50
```

Visualize training modes and 50-step generated modes:

```bash
/mnt/workspace/users/niejunnan/envs/motus/bin/python \
  scripts/visualize_pusht_endpoint_modes.py \
  --dataset_dir /mnt/workspace1/users/niejunnan/datasets/pusht_vgm_v1 \
  --evaluation_dir eval_outputs/pusht_vgm_multimodal/steps_050 \
  --output_dir eval_outputs/pusht_vgm_multimodal/mode_diagnostics_50step \
  --nearest_train 32
```

Run official-simulator replay:

```bash
/mnt/workspace/users/niejunnan/envs/motus/bin/python \
  scripts/evaluate_pusht_physical_replay.py \
  --dataset_dir /mnt/workspace1/users/niejunnan/datasets/pusht_vgm_v1 \
  --evaluation_dir eval_outputs/pusht_vgm_multimodal/steps_050 \
  --output_dir eval_outputs/pusht_vgm_multimodal/physical_replay_50step \
  --official_repo_root /mnt/workspace1/users/niejunnan/third_party/pusht_official_5ba07ac/source \
  --dependency_root /mnt/workspace1/users/niejunnan/third_party/pusht_official_5ba07ac/python_deps \
  --max_missing_run 2
```
