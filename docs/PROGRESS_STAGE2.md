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
loop. The trainer supports an episode-query mixed batch: each rank loads `E`
complete episodes and samples `Q` current-frame queries from each episode. The
`E` episodes may be divided into episode micro-batches to bound activation
memory; DDP synchronization and `optimizer.step()` occur only after every
episode micro-batch has contributed its gradient.

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
`y=i/(N-1)` and its 53-bin soft-label center is `k*=52*y`. Current formal
training uses a Gaussian target with `sigma=2` frame bins plus Smooth-L1 on
expected Progress. The implementation retains an optional within-episode
pairwise ranking term, but every checked-in formal config sets its weight to
zero.

The serial variant only reads cached latents. The layerwise variant additionally
runs one frozen 30-block Wan pass per episode visit, then reuses those 30 hidden
memories for every frame in that episode. This avoids repeating VGM work for
each single-frame query without writing hundreds of megabytes of intermediate
hidden states per episode to disk.

## Detail-Preservation Ablations

The archived baseline remains byte-for-byte compatible at the parameter level:
joint-view encoding with `patch_size: [1,2,2]`, 16 resampler tokens, and a
pooled-cosine alignment head. Three independent switches isolate where small
robot interaction details may be lost:

```yaml
progress_model:
  patch_size: [1, 2, 2]              # set [1,1,1] to keep every VAE cell
  view_encoding_mode: "joint"        # or "split_height"
  num_views: 1                        # use 2 for the vertical LIBERO composite
  tokens_per_view: 16                 # used only by split_height
  alignment_head_mode: "pooled_cosine" # or "token_late_interaction"
  late_interaction_temperature: 0.07
```

`split_height` divides the latent along height before tokenization. Both camera
views use the same tokenizer and resampler weights, while learned view IDs keep
their tokens distinguishable. This prevents one visually dominant view from
consuming every learned resampler slot. The token late-interaction head does not
average a frame to one vector. Each current-frame token finds a smooth maximum
over the local tokens in each of the 53 candidate frames, and learned query-token
weights combine those local matches into frame logits. Its diagnostic outputs
are `query_token_weights[B,K]` and `late_interaction_scores[B,53,K]`.

The formal serial ablation matrix uses exactly the same cache, loss, seed,
epochs, and query batching:

| Experiment | Patch | Views / token budget | Alignment | Config |
| --- | --- | --- | --- | --- |
| A0 baseline | `1x2x2` | joint / 16 total | pooled cosine | `progress_v1_proper_serial_latent_53f.yaml` |
| A1 patch only | `1x1x1` | joint / 16 total | pooled cosine | `progress_v1_proper_serial_latent_53f_ablate_patch1.yaml` |
| A2 view split only | `1x2x2` | 2 x 8 = 16 total | pooled cosine | `progress_v1_proper_serial_latent_53f_ablate_split_views.yaml` |
| A3 head only | `1x2x2` | joint / 16 total | token late interaction | `progress_v1_proper_serial_latent_53f_ablate_token_match.yaml` |
| Combined | `1x1x1` | 2 x 16 = 32 total | token late interaction | `progress_v1_proper_serial_latent_53f_detail_preserving.yaml` |

A1 through A3 are single-factor comparisons against A0. The Combined run is an
upper-capacity test, not a single-factor ablation, because it also doubles the
post-resampler token budget. All five runs consume the existing schema-v2 cache;
none needs to regenerate the 50-step VGM trajectories.

Interpret A1 specifically as a patchification ablation: changing `1x2x2` to
`1x1x1` removes spatial stride, but also changes the convolution kernel and its
parameter count. A2 keeps the total token budget fixed, although the per-view
quota and learned view IDs are necessarily part of the split-view mechanism.

## Build The Shared Cache

The checked-in experiments deliberately use one canonical Stage-1 source:
`vgm_bridge_v1_proper_detailed_caption_v1_53f_full_jitter` at
`checkpoint_step_30000`, sampled with 50 denoising steps. It is a 53-frame,
RGB-only, state-free `first + goal + language` model, so its inputs match the
trajectory-memory contract used by Progress. On the fixed 16-window 53-frame
evaluation it also has lower generated MSE than the state-conditioned and
success50 candidates evaluated alongside it.

Run cache generation once; all serial ablations and the layerwise experiment
use the resulting directory:

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus

torchrun --standalone --nproc_per_node=8 \
  scripts/cache_vgm_progress_latents.py \
  --config configs/progress_v1_proper_serial_latent_53f.yaml
```

Do not overwrite this directory with another Stage-1 checkpoint. A cache is
part of the experimental condition: V1-proper language-only, V2-B state,
success50, role-mask, 17-frame, or a different checkpoint step must each use a
new descriptive `cache_dir`. The current cache script intentionally rejects
state-conditioned and role-mask sources because it does not provide their
additional endpoint inputs. First compare Progress architectures on the shared
canonical cache; compare Stage-1 sources only as a separate experiment axis.

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

The batching controls are:

```yaml
training:
  episode_batch_size: 8        # E per GPU and optimizer step
  episode_micro_batch_size: 2  # trajectories encoded in one forward
  queries_per_episode: 8       # Q queries sampled independently per episode
  query_batch_size: 16         # must cover E_micro * Q
```

The local query batch per optimizer step is `E*Q`; the global query batch is
`world_size*E*Q`. A query carries an explicit episode index, so it can only read
its corresponding 53-frame memory. Alignment/regression are averaged over the
fixed `Q` queries. If ranking is enabled in a future experiment, its pairs are
formed only inside the same episode. No attention operation mixes samples
through the model batch dimension.

With `accelerate launch --num_processes W`, `accelerator.prepare(...)` wraps the
trainable Progress model in DDP. `split_batches` is left disabled, so every rank
receives a complete local batch of `E` episodes. DDP averages the `W` rank-local
gradients, making the update equivalent to the macro average over `W*E`
episodes. The frozen VGM used by the layerwise variant is intentionally not
DDP-wrapped: each rank owns a read-only copy and extracts features only for its
local episodes.

The checked-in configs keep `E=1,Q=0` to reproduce the archived all-frame
baseline. In that mode every selected episode contributes all of its cached
source frames and `query_batch_size` remains an activation micro-batch. Accelerate
uses even distributed batches by default, so when the episode count is not
divisible by `world_size*E`, at most `world_size*E-1` episode visits are repeated
at the epoch boundary to keep all DDP ranks on the same number of optimizer
steps. To enable mixed batching, set `E>1`, choose a positive fixed `Q`, and ensure
`episode_micro_batch_size*Q <= query_batch_size`. The current LIBERO source has
1,693 episodes and 273,465 frames. Increasing `E` reduces optimizer steps per
epoch, so comparisons must hold total optimizer steps or total query exposures
constant rather than blindly reusing the same epoch count. The cosine schedule
is derived from the actual distributed steps per epoch; `max_steps` remains a
safety ceiling.

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
