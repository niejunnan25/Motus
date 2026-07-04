# VGM Bridge Stage1 Visualization

This document records how the Stage1 bridge evaluation images in
`assets/vgm_bridge_stage1_eval/` were generated.

The goal of this visualization is to compare different VGM bridge variants on
the same fixed LIBERO cases, so visual differences come from the model variant
rather than from different sampled episodes, frame windows, or denoising noise.

## Evaluation Setup

Remote host:

```bash
ssh 25605
cd /mnt/workspace1/users/niejunnan/codebase/Motus
conda activate /mnt/workspace/users/niejunnan/envs/motus
```

Common evaluation arguments:

```bash
--num_samples 16 \
--batch_size 1 \
--loss_repeats 1 \
--num_inference_steps 50 \
--seed 50 \
--fps 4
```

These settings generate 16 deterministic LIBERO windows. The eval script writes
the selected windows to `windows.json`; comparing `episode_name`,
`condition_idx`, `frame_indices`, and `total_frames` verifies that all models
use the same cases.

Each per-model eval output contains:

```text
metrics.json
metrics.csv
windows.json
samples/sample_000_sheet.png
samples/sample_000_gt_pred.mp4
...
```

The per-model sheet layout is:

```text
row 1: GT
row 2: model prediction
```

## Per-Model Eval Commands

Run the four 10k checkpoints on separate idle GPUs. In our run, GPUs 4-7 on
`25605` were used while GPUs 0-3 continued training.

```bash
COMMON_ARGS="--num_samples 16 --batch_size 1 --loss_repeats 1 --num_inference_steps 50 --seed 50 --fps 4"

CUDA_VISIBLE_DEVICES=4 python train/eval_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_v0.yaml \
  --checkpoint checkpoints/vgm_bridge_v0/vgm_bridge_v0_lerobot_video_448x224_2gpu_gbs8_50k/checkpoint_step_10000 \
  --output_dir eval_outputs/vgm_bridge_v0_step10000_more16_s50 \
  $COMMON_ARGS

CUDA_VISIBLE_DEVICES=5 python train/eval_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_v0_lang.yaml \
  --checkpoint checkpoints/vgm_bridge_v0_lang/vgm_bridge_v0_lerobot_video_448x224_lang_17f_2gpu_gbs8_50k/checkpoint_step_10000 \
  --output_dir eval_outputs/vgm_bridge_v0_lang_step10000_more16_s50 \
  $COMMON_ARGS

CUDA_VISIBLE_DEVICES=6 python train/eval_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_v0_tail4.yaml \
  --checkpoint checkpoints/vgm_bridge_v0_tail4/vgm_bridge_v0_lerobot_video_448x224_tail4_17f_2gpu_gbs8_50k/checkpoint_step_10000 \
  --output_dir eval_outputs/vgm_bridge_tail4_step10000_more16_s50 \
  $COMMON_ARGS

CUDA_VISIBLE_DEVICES=7 python train/eval_vgm_bridge_stage1.py \
  --config configs/vgm_bridge_v0_tail4_lang.yaml \
  --checkpoint checkpoints/vgm_bridge_v0_tail4_lang/vgm_bridge_v0_lerobot_video_448x224_tail4_lang_17f_2gpu_gbs8_50k/checkpoint_step_10000 \
  --output_dir eval_outputs/vgm_bridge_tail4_lang_step10000_more16_s50 \
  $COMMON_ARGS
```

The actual run used `nohup` so the jobs could continue after SSH disconnects.
That only changes process management, not the generated outputs.

## Output Directories Used

The four 10k outputs are:

```text
eval_outputs/vgm_bridge_v0_step10000_more16_s50/
eval_outputs/vgm_bridge_v0_lang_step10000_more16_s50/
eval_outputs/vgm_bridge_tail4_step10000_more16_s50/
eval_outputs/vgm_bridge_tail4_lang_step10000_more16_s50/
```

The combined comparison directory is:

```text
eval_outputs/vgm_bridge_compare_17f_step10000_more16_s50/
```

It contains:

```text
metrics_summary.csv
metrics_summary.json
window_compare.json
windows.json
samples/comparison_sample_000_sheet.png
samples/comparison_sample_000.mp4
per_group_sheets/sample_000_v0_10k.png
per_group_sheets/sample_000_v0_lang_10k.png
per_group_sheets/sample_000_tail4_10k.png
per_group_sheets/sample_000_tail4_lang_10k.png
...
```

## Window Consistency Check

Before making comparison sheets, compare each model's `windows.json`:

```python
import json
from pathlib import Path

keys = ["episode_name", "condition_idx", "frame_indices", "total_frames"]

paths = [
    Path("eval_outputs/vgm_bridge_v0_step10000_more16_s50/vgm_bridge_v0_lerobot_video_448x224_2gpu_gbs8_50k/step_10000"),
    Path("eval_outputs/vgm_bridge_v0_lang_step10000_more16_s50/vgm_bridge_v0_lerobot_video_448x224_lang_17f_2gpu_gbs8_50k/step_10000"),
    Path("eval_outputs/vgm_bridge_tail4_step10000_more16_s50/vgm_bridge_v0_lerobot_video_448x224_tail4_17f_2gpu_gbs8_50k/step_10000"),
    Path("eval_outputs/vgm_bridge_tail4_lang_step10000_more16_s50/vgm_bridge_v0_lerobot_video_448x224_tail4_lang_17f_2gpu_gbs8_50k/step_10000"),
]

base = json.loads((paths[0] / "windows.json").read_text())
for path in paths[1:]:
    cur = json.loads((path / "windows.json").read_text())
    assert len(cur) == len(base)
    assert all(all(a.get(k) == b.get(k) for k in keys) for a, b in zip(cur, base))
```

The generated `window_compare.json` in the comparison directory records this
check for all compared models.

## Per-Group Image Generation

The final GitHub assets use one image per model group. Each image contains only:

```text
row 1: GT
row 2: one model prediction
```

The script below reads the combined comparison sheet and splits it into
per-group images:

```python
from pathlib import Path
from PIL import Image, ImageDraw
import json

root = Path("eval_outputs/vgm_bridge_compare_17f_step10000_more16_s50")
out = root / "per_group_sheets"
out.mkdir(parents=True, exist_ok=True)

windows = json.loads((root / "windows.json").read_text())
models = [
    ("v0_10k", "V0@10k", 1),
    ("v0_lang_10k", "V0+Lang@10k", 2),
    ("tail4_10k", "Tail4@10k", 3),
    ("tail4_lang_10k", "Tail4+Lang@10k", 4),
]

label_w = 170
header_h = 34

for sample_idx, window in enumerate(windows):
    src = Image.open(root / "samples" / f"comparison_sample_{sample_idx:03d}_sheet.png").convert("RGB")
    row_h = (src.height - header_h) // 6
    gt = src.crop((label_w, header_h, src.width, header_h + row_h))

    for slug, label, row_idx in models:
        pred_y0 = header_h + row_idx * row_h
        pred = src.crop((label_w, pred_y0, src.width, pred_y0 + row_h))

        canvas = Image.new("RGB", (label_w + gt.width, header_h + 2 * row_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"sample_{sample_idx:03d}  {window.get('episode_name')}  "
            f"cond={window.get('condition_idx')}  {label}"
        )
        draw.text((10, 9), title, fill=(20, 24, 31))

        for y, name, image in [(header_h, "GT", gt), (header_h + row_h, label, pred)]:
            draw.rectangle((0, y, label_w, y + row_h), fill=(248, 248, 248))
            draw.text((10, y + 12), name, fill=(20, 24, 31))
            canvas.paste(image, (label_w, y))

        canvas.save(out / f"sample_{sample_idx:03d}_{slug}.png")
```

The four images copied into `assets/vgm_bridge_stage1_eval/` are the
`sample_000` outputs from this `per_group_sheets/` directory.

## Copying Results Back To Local

The remote node did not have `rsync`, so the directory was copied with `tar`:

```bash
mkdir -p /Users/n/Documents/DreamZero/artifacts

ssh 25605 \
  'cd /mnt/workspace1/users/niejunnan/codebase/Motus/eval_outputs && tar -cf - vgm_bridge_compare_17f_step10000_more16_s50' \
  | tar -C /Users/n/Documents/DreamZero/artifacts -xf -
```

Then the four `sample_000` images were copied into the repository:

```bash
mkdir -p assets/vgm_bridge_stage1_eval

cp /Users/n/Documents/DreamZero/artifacts/vgm_bridge_compare_17f_step10000_more16_s50/per_group_sheets/sample_000_v0_10k.png \
  assets/vgm_bridge_stage1_eval/vgm_bridge_v0_sample_000_step10000.png

cp /Users/n/Documents/DreamZero/artifacts/vgm_bridge_compare_17f_step10000_more16_s50/per_group_sheets/sample_000_v0_lang_10k.png \
  assets/vgm_bridge_stage1_eval/vgm_bridge_v0_lang_sample_000_step10000.png

cp /Users/n/Documents/DreamZero/artifacts/vgm_bridge_compare_17f_step10000_more16_s50/per_group_sheets/sample_000_tail4_10k.png \
  assets/vgm_bridge_stage1_eval/vgm_bridge_v0_tail4_sample_000_step10000.png

cp /Users/n/Documents/DreamZero/artifacts/vgm_bridge_compare_17f_step10000_more16_s50/per_group_sheets/sample_000_tail4_lang_10k.png \
  assets/vgm_bridge_stage1_eval/vgm_bridge_v0_tail4_lang_sample_000_step10000.png
```
