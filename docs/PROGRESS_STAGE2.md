# Progress Stage 2

## Scope

Stage 2 freezes the validated V1-proper endpoint-conditioned VGM and learns
single-frame task Progress. The online observation is exactly one RGB frame.
The start/goal-conditioned trajectory is generated once per episode and reused
as a visual trajectory memory.

Both structures classify the online observation against exactly 53 generated
frames. They share the same cache, labels, output, and loss:

- `serial_latent`: the final clean `z_trajectory` is decoded to 53 RGB frames;
  each frame and the online observation are independently VAE-encoded, then a
  6-layer Progress Transformer reads the resulting ordered frame memory.
- `layerwise_wvm`: a 30-layer Progress expert reads the corresponding frozen
  Wan hidden state after every Video DiT block. Its masked joint attention is
  asymmetric: Progress reads Video and Progress tokens, while Video never reads
  or receives gradients from Progress. The final classifier still scores the
  same explicit 53-frame memory as `serial_latent`.

`layerwise_wvm` follows WVM's layer-aligned asymmetric information flow, but it
is adapted to Motus's endpoint-generated 53F memory and one-frame online input;
it is not a parameter-identical reproduction of WVM's 4F input model.

## Cache Design

Cache generation is deliberately separated from Progress optimization. For
every source episode it stores:

- one 50-step V1-proper `z_trajectory`;
- 53 decoded generated frames, each independently re-encoded by the same Wan
  VAE into `trajectory_frame_latents[53,C,1,H,W]`;
- every source frame independently encoded by Wan VAE as a one-frame latent;
- exact normalized labels `frame_index / (total_frames - 1)`;
- first/last RGB frames for the layerwise clean-latent Wan pass;
- one deduplicated task language embedding.

The decoded RGB frames are not retained, avoiding roughly 27 GB of redundant
uint8 storage over the current 1,693 episodes. Their independently encoded
frame latents are sufficient to preserve exactly 53 addressable memory slots.

Therefore neither training variant performs 50-step generation in its training
loop. One optimizer step consumes one complete episode. Its frames are divided
into query micro-batches to bound activation memory; DDP synchronization occurs
only on the final micro-batch, so all frames contribute to one episode-level
gradient without requiring them to fit in GPU memory simultaneously.

The shared trainable frame encoder processes current-frame and trajectory-frame
latents with the same weights. A deterministic frame-position embedding keeps
the 53 memory slots ordered. The output is:

```text
alignment_logits:       [B, 53]
alignment_probabilities:[B, 53]
matched_frame:          argmax_k p(k)
progress:               sum_k p(k) * k / 52
```

For a source episode frame `i` among `N` frames, the normalized target is
`y=i/(N-1)` and its 53-bin soft-label center is `k*=52*y`. Training uses a
Gaussian target with `sigma=2` frame bins, plus Smooth-L1 on expected Progress
and a within-episode pairwise ranking term.

The serial variant only reads cached latents. The layerwise variant additionally
runs one frozen 30-block Wan pass per episode visit, then reuses those 30 hidden
memories for every frame in that episode. This avoids repeating VGM work for
each single-frame query without writing hundreds of megabytes of intermediate
hidden states per episode to disk.

## Build The Shared Cache

Run once; both experiments use the resulting directory:

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus

torchrun --standalone --nproc_per_node=8 \
  scripts/cache_vgm_progress_latents.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml
```

For a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  scripts/cache_vgm_progress_latents.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml \
  --max_episodes 2 \
  --cache_dir /tmp/motus_progress_cache_smoke \
  --num_inference_steps 2 \
  --overwrite
```

The cache format is schema v2. Schema v1 stored only the 14-slice VAE video
latent and is deliberately rejected because it cannot represent a 53-frame
classification target. The smoke command deliberately uses a separate cache
directory and only two denoising steps. Formal cache generation must keep the
YAML's 50 steps. Pass
`--overwrite` when the checkpoint, inference-step count, or seed intentionally
changes an existing isolated cache.

## Train

Serial latent memory:

```bash
accelerate launch --num_processes 8 \
  train/train_progress_stage2.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml
```

WVM-style layerwise expert:

```bash
accelerate launch --num_processes 8 \
  train/train_progress_stage2.py \
  --config configs/progress_v1_proper_layerwise_wvm_53f.yaml
```

An isolated trainer smoke test can override the cache and runtime without
editing the formal YAML:

```bash
accelerate launch --num_processes 2 \
  train/train_progress_stage2.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml \
  --cache_dir /tmp/motus_progress_cache_smoke \
  --max_steps 1 \
  --max_epochs 1 \
  --report_to none
```

`query_batch_size` is only the per-episode activation micro-batch. With
`queries_per_episode: 0`, every cached source frame is used once per epoch.
The current LIBERO source has 1,693 episodes and 273,465 frames. Both configs
run 20 complete epochs; on 8 GPUs this is roughly 3,860 global optimizer steps
after the deterministic validation split, or about 4.9 million training-frame
queries. The cosine schedule is derived from the actual distributed
steps-per-epoch. `max_steps` is only a safety ceiling.

## Evaluate

```bash
python train/eval_progress_stage2.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml \
  --checkpoint /absolute/path/to/checkpoint_step_N \
  --output_dir eval_outputs/progress_serial_step_N
```

Use `--cache_dir /absolute/cache/path` when evaluating an isolated smoke cache.

Evaluation writes aggregate and per-episode metrics plus diagnostics containing
the GT/predicted Progress curve and the single-frame-to-trajectory alignment
probability heatmap over columns 0 through 52. Primary comparisons are Progress
MAE/RMSE, expected-frame MAE, argmax matched-frame MAE, within-one-frame
accuracy, ordering accuracy, Spearman correlation, monotonic violation rate,
endpoint calibration, latency, and memory use.
