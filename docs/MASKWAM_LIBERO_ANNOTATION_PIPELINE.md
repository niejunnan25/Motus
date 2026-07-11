### MaskWAM-Style LIBERO Mask Annotation Pipeline

This pipeline replaces the heavy 16-case SAM3 debug dashboard with a lighter annotation loop:

```text
automatic SAM3 first pass
-> simple human review
-> point/box correction
-> SAM3 re-propagation
-> trainable mask sequence
```

The current implementation is intentionally scoped to the 16 fixed VGM evaluation windows. It should not be used for full-dataset mask generation until the 16-case loop is accepted.

### Current View-Aware Status

The pipeline was revised after visual QA showed that the original first pass mostly segmented the main/top view and missed the wrist/bottom view. Wrist-view masks are required because the V3 mask objective is supposed to emphasize robot-operation details, not only scene-level objects in the main camera.

The current formal train roles are therefore view-aware:

```text
main_object
main_target
main_robot
wrist_object
wrist_target
wrist_robot
```

The base roles `object`, `target`, and `robot` are still saved for debugging and prompt propagation, but they are not sufficient for the formal training mask. `final_train_mask` is the union of the configured view-aware roles.

Current 16-case view-aware first pass:

```text
remote:
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/view_schema_first_pass_qa16

local:
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/view_schema_first_pass_qa16
```

Human review entry:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/view_schema_first_pass_qa16/review/index.html
```

Task-queue annotator:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/view_schema_first_pass_qa16/review/task_queue_annotator/index.html
```

Readable per-task workbench:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/view_schema_first_pass_qa16/review/correction_workbench/index.html
```

Current status:

```text
verify.status = invalid
train_mask_sequences_strict.status = not_ready
correction task count = 74
task priorities = 58 critical, 12 high, 4 medium
task roles:
  wrist_object = 13
  wrist_target = 18
  wrist_robot = 16
  main_object = 13
  main_target = 9
  main_robot = 4
  final_train_mask = 1 aggregate diagnostic
```

Interpretation:

```text
The current first pass is not train-ready.
This is expected and correct: the strict exporter now rejects missing wrist roles and unaccepted fallback/manual masks.
The next required step is human point/box correction on the view-aware tasks, then SAM3 re-propagation.
```

If a view-aware task is genuinely not visible in the selected view, do not draw a box on empty background. In `task_queue_annotator/index.html`, click `mark not visible`. This writes an `accepted_absent=true` correction for that exact `sample_id / role / name`. During re-propagation, this role/name/view is kept empty and recorded as human-accepted absent instead of being treated as a failed mask.

### Mask Semantics

- `object`: base debug role for the manipulated object.
- `target`: base debug role for the local destination, receptacle, handle, drawer front, plate, basket, or placement/contact area.
- `robot`: base debug role for the visible arm and gripper.
- `context`: human review only; excluded from training mask.
- `main_object`, `main_target`, `main_robot`: train roles restricted to the main/top view.
- `wrist_object`, `wrist_target`, `wrist_robot`: train roles restricted to the wrist/bottom view.
- `final_train_mask`: union of configured view-aware train roles.

### Files

Config:

```text
configs/maskwam_libero_annotation_debug16.json
```

Manifest builder:

```text
scripts/maskwam_build_libero_annotation_manifest.py
```

SAM3 video propagation runner:

```text
scripts/maskwam_run_sam3_video_annotation.py
```

View-aware role helpers:

```text
scripts/maskwam_view_schema.py
```

Lightweight review generator:

```text
scripts/maskwam_make_annotation_review.py
```

Verifier:

```text
scripts/maskwam_verify_annotation_run.py
```

Correction validator:

```text
scripts/maskwam_validate_corrections.py
```

Correction merger:

```text
scripts/maskwam_merge_corrections.py
```

Correction coverage checker:

```text
scripts/maskwam_check_correction_coverage.py
```

Correction preparation wrapper:

```text
scripts/maskwam_prepare_corrections_for_repropagate.py
```

Full correction loop wrapper:

```text
scripts/maskwam_run_full_correction_loop.py
```

Local correction import + 7779 preflight wrapper:

```text
scripts/maskwam_import_and_preflight_corrections.py
```

Corrected-run orchestrator:

```text
scripts/maskwam_run_corrected_pipeline.py
```

Correction task list generator:

```text
scripts/maskwam_make_correction_tasks.py
```

Pending correction summary generator:

```text
scripts/maskwam_summarize_pending_corrections.py
```

Train manifest exporter:

```text
scripts/maskwam_export_train_mask_sequences.py
```

Corrected-vs-first-pass comparison review:

```text
scripts/maskwam_make_correction_comparison_review.py
```

Corrected-run audit:

```text
scripts/maskwam_audit_corrected_run.py
```

### Debug16 Flow

Build the manifest:

```bash
python scripts/maskwam_build_libero_annotation_manifest.py \
  --config configs/maskwam_libero_annotation_debug16.json
```

Run a small SAM3 first-pass smoke test on one sample:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/maskwam_run_sam3_video_annotation.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --output_dir artifacts/maskwam_libero_annotation_debug16/first_pass_smoke \
  --sample_ids 0 \
  --skip_context \
  --sam3_version sam3
```

Generate the review page:

```bash
python scripts/maskwam_make_annotation_review.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --annotation_dir artifacts/maskwam_libero_annotation_debug16/first_pass_smoke \
  --output_dir artifacts/maskwam_libero_annotation_debug16/review_first_pass_smoke
```

Human review entry:

```text
artifacts/maskwam_libero_annotation_debug16/review_first_pass_smoke/index.html
```

Human correction template:

```text
artifacts/maskwam_libero_annotation_debug16/review_first_pass_smoke/corrections_template.json
```

If a sample is wrong, fill `corrections_template.json` with point or box prompts on the composed 448x224 frame, then rerun:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/maskwam_run_sam3_video_annotation.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_smoke/corrections_template.json \
  --output_dir artifacts/maskwam_libero_annotation_debug16/repropagated_smoke \
  --sample_ids 0 \
  --skip_context \
  --sam3_version sam3
```

### Output Layout

Each annotated sample contains:

```text
sample_XXX/
  rgb/frame_000.png
  masks/object/frame_000.png
  masks/target/frame_000.png
  masks/robot/frame_000.png
  masks/main_object/frame_000.png
  masks/main_target/frame_000.png
  masks/main_robot/frame_000.png
  masks/wrist_object/frame_000.png
  masks/wrist_target/frame_000.png
  masks/wrist_robot/frame_000.png
  masks/context/frame_000.png
  masks/final_train_mask/frame_000.png
  masks/by_prompt/<role>__<name>/frame_000.png
  annotation_metadata.json
```

This is the trainable mask sequence format for later V3B/V3C work. RGB and mask frames share the same composed 448x224 geometry. For the current formal schema, the train mask comes from the view-aware roles, not directly from the base `object/target/robot` masks.

The exporter converts this directory layout into a training manifest:

```bash
python scripts/maskwam_export_train_mask_sequences.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --annotation_dir artifacts/maskwam_libero_annotation_debug16/repropagated_qa16 \
  --review_csv artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/human_review_template.csv \
  --require_review_ok \
  --output_json artifacts/maskwam_libero_annotation_debug16/train_manifests/repropagated_qa16.json
```

The output JSON contains one entry per sample with:

```text
rgb_paths
mask_paths                  # final_train_mask paths
role_mask_paths             # base debug roles + view-aware train roles + final_train_mask
frame_indices
source_video_paths
review_status
final_zero_frames
final_mean_area_ratio
training_ready
issues
warnings
```

`status=ready` means the sequence is structurally usable for training. It does not replace visual QA. Large or tiny mask area ratios are warnings because they often indicate over-segmentation or empty operation regions.

### Legacy Debug16 First-Pass Result (Superseded)

This section records the original base-role run. It is superseded by the view-aware run above because it does not separately validate the wrist view.

The first SAM3 video-propagation pass has been run on 7779 with SAM3 base:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/first_pass_qa16
```

The lightweight local review pack is:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/index.html
```

The same review pack on 7779 is:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/index.html
```

This first pass proved that the end-to-end base-role pipeline worked:

```text
manifest -> composed RGB frames -> SAM3 video propagation -> role masks -> final_train_mask -> review HTML
```

But it also showed two problems. First, pure text prompts are not reliable enough for small LIBERO objects. Many object prompts, such as `alphabet soup`, `ketchup`, `black bowl`, or `wine bottle`, produce empty masks. Second, base-role validation can hide wrist-view failures because the main view may have non-empty masks while the wrist role is still empty. Therefore the current pipeline uses view-aware roles and first-frame/keyframe point or box correction.

The review HTML includes clickable raw frames. It can generate positive points, negative points, positive boxes, and negative boxes with absolute 224x448 composed-image coordinates.

Each sample now has its own `prompt` selector above the sheet. Prefer this selector over manually typing `role/name`; its options come from the manifest, for example:

```text
object: white mug
target: left plate
target: right plate
robot: robot arm
```

Use `manual role/name` only when the configured prompts are not enough. This reduces correction JSON mismatch errors before re-propagation.

After corrections are filled, rerun:

```bash
python scripts/maskwam_merge_corrections.py \
  --inputs \
    artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_tasks_draft.json \
    artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json \
  --output artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged.json
```

The merger drops empty draft prompts, deduplicates identical prompts, sorts prompts by sample/frame/role/name, and keeps only active point/box corrections.

For the normal corrected run, use the orchestrator instead of running every script manually:

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus
export HF_HOME=/mnt/workspace1/users/niejunnan/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/workspace1/users/niejunnan/codebase/sam3:$PYTHONPATH

/mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  scripts/maskwam_run_corrected_pipeline.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged.json \
  --correction_tasks artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json \
  --coverage_require_all_critical \
  --run_name repropagated_qa16 \
  --python /mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  --cuda_visible_devices 0 \
  --sam3_version sam3 \
  --require_all_actionable
```

This runs:

```text
check correction coverage
-> validate corrections
-> SAM3 re-propagate
-> make review pack
-> verify annotation run
-> export train mask sequences
```

The main outputs are:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/annotations
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/review/index.html
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/correction_coverage.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/train_mask_sequences.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/pipeline_summary.json
```

The coverage gate checks whether the active corrections cover all critical object/target/robot TODOs before any GPU SAM3 work starts. The validator then checks sample IDs, roles, prompt matching, frame indices, point/box coordinate bounds, and point/box label counts.

After the corrected run finishes, generate a first-pass vs corrected-pass comparison review:

```bash
python scripts/maskwam_make_correction_comparison_review.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --before_annotation_dir artifacts/maskwam_libero_annotation_debug16/first_pass_qa16 \
  --after_annotation_dir artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/annotations \
  --output_dir artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/comparison_review
```

Open:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/comparison_review/index.html
```

Each sheet shows:

```text
RGB
before roles
after roles
before final
after final
change overlay
```

In the change row, green means final mask was added by correction, yellow means final mask was removed, and red means unchanged final mask. This is the recommended visual QA step before accepting warnings such as fallback prompts or large mask area.

After the corrected run finishes, audit the whole run before using `train_mask_sequences.json` for training:

```bash
python scripts/maskwam_audit_corrected_run.py \
  --run_dir artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16 \
  --expected_samples 16
```

The audit checks:

```text
pipeline_summary.json status
correction_coverage.json critical coverage
correction_validation.json validity
review/verification.json mask/review structure
train_mask_sequences.json training readiness
```

It writes:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/audit.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/audit.md
```

`status=ready` means the corrected run has complete pipeline outputs, valid corrections, no uncovered critical actionable task, a valid annotation/review structure, and a ready train mask manifest for all 16 samples. Use `--allow_train_warnings` only when warnings such as fallback prompts or large mask area are visually inspected and accepted.

### Legacy Debug16 Corrected Run (Superseded)

This run is no longer accepted as the formal training source. It is structurally useful as a historical base-role correction smoke test, but it predates the view-aware schema and does not satisfy the current requirement that `main_*` and `wrist_*` roles are separately complete and human-accepted.

The legacy corrected run is:

```text
ai_box_draft_repropagated_v4
```

Remote 7779 path:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4
```

Local synced path:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4
```

Main review entry:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4/review/index.html
```

Trainable manifest:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4/train_mask_sequences.json
```

Audit files:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4/audit_strict.json
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/ai_box_draft_repropagated_v4/audit_allow_train_warnings.json
```

Legacy structural status under the old base-role schema:

```text
pipeline_summary.status = complete
verification.status = valid_unapproved
verification.samples_with_role_zero = []
verification.samples_with_final_zero = []
train_mask_sequences.status = ready
train_mask_sequences.ready_count = 16 / 16
train_mask_sequences.issue_count = 0
train_mask_sequences.warning_count = 15
```

The strict audit reports `not_ready` only because 15 samples have fallback/manual correction warnings:

```text
audit_strict.status = not_ready
audit_strict.issues = ["train sample warning count: 15"]
```

The training-format audit with accepted warnings is ready:

```text
audit_allow_train_warnings.status = ready
audit_allow_train_warnings.issues = []
audit_allow_train_warnings.warnings = ["train sample warning count: 15"]
```

Current interpretation:

```text
The run is not the formal train-ready source for V3B/V3C.
It should not be used to train mask-conditioned models unless it is regenerated or audited under the current view-aware schema.
```

Counts in the synced local run:

```text
review sheets: 16
click frames: 80
rgb frames: 272
final train mask frames: 272
```

Manual commands for each step are still listed below for debugging.

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  scripts/maskwam_run_sam3_video_annotation.py \
  --manifest artifacts/maskwam_libero_annotation_debug16/manifest.json \
  --corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_template.json \
  --output_dir artifacts/maskwam_libero_annotation_debug16/repropagated_qa16 \
  --skip_context \
  --sam3_version sam3
```

SAM3.1 was probed but is not currently available in the 7779 offline Hugging Face cache, so the validated runner uses SAM3 base.

### Correction Semantics

The correction runner now supports two human correction types:

```json
{
  "name": "white mug",
  "role": "object",
  "frame_index": 0,
  "points_abs": [[170, 102]],
  "point_labels": [1],
  "boxes_xywh_abs": [],
  "box_labels": []
}
```

or:

```json
{
  "name": "white mug",
  "role": "object",
  "frame_index": 0,
  "points_abs": [],
  "point_labels": [],
  "boxes_xywh_abs": [[142, 70, 58, 64]],
  "box_labels": [1]
}
```

Coordinates are absolute pixels on the composed `224x448` RGB frame. `point_labels` and `box_labels` use `1` for positive and `0` for negative. For point clicks, the fallback path expands a point into a small square controlled by `--point_box_size`.

The priority order is:

```text
1. Try SAM3 video propagation with the provided point/box correction.
2. If SAM3 returns an all-zero correction mask, use a manual correction fallback mask.
3. If SAM3 raises on the correction prompt, use the same fallback mask.
```

For point corrections, the runner passes `start_frame_index` to SAM3 propagation. This is required by SAM3 base video tracking; otherwise point-only corrections can fail with `No prompts are received`.

The fallback is intentionally explicit in metadata:

```text
fallback_used: true
fallback_reason: sam3_correction_all_zero
```

or:

```text
fallback_reason: sam3_exception:RuntimeError:No prompts are received ...
```

This prevents a bad interactive SAM3 prompt from silently producing empty training masks.

The review HTML supports:

```text
embedded correction task list
live coverage summary
click task -> scroll to sample and select prompt
per-sample prompt selector
positive point
negative point
positive box
negative box
undo last prompt
load JSON from textarea
download corrections JSON
```

Box mode uses two clicks on the same raw frame: first corner, second corner.

Current local review entry:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/index.html
```

Current review verification:

```text
status=needs_human_corrections
sheets=16
click_frames=80
human_review_rows=16
```

The review directory also contains an automatically generated correction TODO:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.csv
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.html
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_tasks_draft.json
```

Current task summary:

```text
tasks=25
samples=15/16
critical=15
high=6
medium=4
```

The draft correction JSON is intentionally coordinate-empty and validates as:

```text
status=valid
active=0
issues=0
warnings=0
```

Use the embedded `Correction Tasks` panel in `index.html` first. Clicking a task scrolls to the relevant sample and selects the matching `object` / `target` / `robot` prompt when it exists in the manifest. Critical tasks default to box mode because empty masks usually need a tight positive box. High and medium tasks default to point mode because they often need a keyframe correction or a negative point.

The workbench updates coverage live while you click:

```text
actionable covered / total object-target-robot tasks
critical covered / total critical actionable tasks
high covered / total high actionable tasks
medium covered / total medium actionable tasks
active prompt count
unmatched active prompt count
```

Task cards turn green when the current browser JSON covers them. Uncovered critical task cards keep a red border. The browser coverage is a convenience check for annotation speed; the authoritative gate is still `maskwam_check_correction_coverage.py` or the coverage step inside `maskwam_run_corrected_pipeline.py`.

The page also supports:

```text
undo last prompt
load JSON from textarea
local browser autosave / restore
clear saved copy
download corrections JSON
```

The workbench autosaves the current correction JSON to browser local storage under the current review manifest key. This protects against accidental refreshes while annotating. It is not a trainable artifact and is not visible to command-line scripts, so still download the JSON when the corrections are ready.

After adding clicks, press `download corrections JSON` in the review page. Save that browser download as:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json
```

If the browser downloaded the file into `~/Downloads`, the local helper can import it, sync it to 7779, and run the no-GPU preflight:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
python3 scripts/maskwam_import_and_preflight_corrections.py
```

The helper searches in this order:

```text
explicit --downloaded_corrections
review_first_pass_qa16_click/corrections_from_html.json
~/Downloads/corrections_from_html*.json
```

It writes:

```text
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_import_preflight_summary.json
```

It also writes a run-specific copy that is not overwritten by later helper runs:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/<run_name>/local_import_preflight_summary.json
```

When `--run_full_after_preflight` is used, the run-specific copy is stored under `<full_run_name>`.

Use `--dry_run` to print the rsync and remote preflight commands without copying or running anything.

After the helper preflight is clean, you can either run the full GPU command manually, or let the helper continue only after preflight succeeds:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
python3 scripts/maskwam_import_and_preflight_corrections.py \
  --run_full_after_preflight \
  --run_name repropagated_qa16_preflight \
  --full_run_name repropagated_qa16
```

`--run_full_after_preflight` refuses to run with `--skip_preflight`. The full remote command still passes `--require_all_actionable`; it simply removes `--stop_after_prepare` after the preflight command returns successfully.

If you want the handoff to happen automatically after the browser download, start the watcher before annotation:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
python3 scripts/maskwam_wait_for_corrections.py \
  --run_full_after_preflight \
  --run_name repropagated_qa16_preflight \
  --full_run_name repropagated_qa16
```

By default this waits for a `corrections_from_html.json` modified after the watcher starts, checks that the file is stable, then calls `maskwam_import_and_preflight_corrections.py`. Use `--accept_existing` if the JSON was already downloaded before starting the watcher. Use `--timeout_sec <seconds>` for a bounded wait.

Quick status check:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
python3 scripts/maskwam_status.py --check_remote \
  --output_json artifacts/maskwam_libero_annotation_debug16/annotation_pipeline_status.json
```

This is a read-only helper. It checks whether a browser-exported `corrections_from_html.json` exists, summarizes validation and train-ready coverage if it does, and optionally checks the 7779 `repropagated_qa16` run for `audit.json` and `train_mask_sequences.json`. The formal run is ready to start only when this reports `ready_for_formal_repropagate=true` in the JSON output, which means all object/target/robot actionable rows are covered.

Use the pending summary as the annotation worklist:

```bash
python scripts/maskwam_summarize_pending_corrections.py
```

For easier human inspection, generate the per-task correction workbench:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
/Users/n/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/maskwam_make_correction_workbench.py
```

Current output:

```text
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_workbench/index.html
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_workbench/tasks/*.png
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_workbench/workbench_manifest.json
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_workbench/workbench_manifest.csv
```

The workbench contains one readable PNG per train-ready actionable row. Each image shows the task text, role/name to fix, suggested point/box action, recommended frame, and all five click frames with the recommended frame highlighted. It is only a visual guide; the actual point/box corrections still need to be entered and downloaded from `index.html`.

For the shortest point/box entry path, generate and open the task-queue annotator:

```bash
cd /Users/n/Documents/DreamZero/repos/Motus
python3 scripts/maskwam_make_task_queue_annotator.py
```

Current output:

```text
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/task_queue_annotator/index.html
artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/task_queue_annotator/task_queue_manifest.json
```

This page only contains the 24 formal object/target/robot correction tasks. It automatically binds each click or box to the current task's exact `sample_id`, `role`, `name`, and selected `frame_index`, then downloads the same `corrections_from_html.json` schema used by the full review page. Use this page for actual correction entry when the full review page is too broad.

Current 16-case first-pass worklist:

```text
pending summary:
  artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.md
  artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.csv
  artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/pending_corrections_summary.json

default prepare gate:
  14 unique actionable prompts across 11 samples

formal train-ready gate:
  24 unique actionable prompts across 15 samples
```

The default prepare gate only checks critical actionable tasks. For the formal 16-case corrected run, use the train-ready gate and fix all object/target/robot actionable rows, including high/medium rows, because the final audit can still fail when a role has zero-mask frames. `final_train_mask` rows are aggregate diagnostics; fix the object/target/robot rows in the same sample instead of selecting `final_train_mask` directly.

Each actionable row also includes a recommended frame:

```text
recommended_frame_index
recommended_frame_reason
zero_frame_indices
```

For all-zero masks, the default recommended frame is the middle review frame, usually `F8`; still inspect the five click frames and use the first clearly visible frame if another frame is better. For partially missing masks, the recommendation is the review frame nearest to the earliest zero-mask frame. The generated `corrections_tasks_draft.json` also uses this recommendation as the default `frame_index`.

For manual inspection outside the workbench, `correction_tasks.html` still groups tasks by priority and links directly to each sample review sheet plus the clickable raw frames.

After saving `corrections_from_html.json`, the recommended path is the full correction loop wrapper. It runs:

```text
prepare corrections
-> corrected SAM3 re-propagation
-> first-pass vs corrected comparison review
-> corrected-run audit
```

Run it on 7779:

```bash
cd /mnt/workspace1/users/niejunnan/codebase/Motus
export HF_HOME=/mnt/workspace1/users/niejunnan/cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/workspace1/users/niejunnan/codebase/sam3:$PYTHONPATH

/mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  scripts/maskwam_run_full_correction_loop.py \
  --review_dir artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click \
  --downloaded_corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json \
  --run_name repropagated_qa16 \
  --python /mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  --cuda_visible_devices 0 \
  --sam3_version sam3
```

It writes a top-level summary:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/full_correction_loop_summary.json
```

The wrapper keeps run-specific intermediate files inside the same corrected-run directory, so every re-propagation attempt is self-contained:

```text
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/corrections_merged.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/corrections_prepare_summary.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/correction_coverage.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/correction_validation.json
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/comparison_review/index.html
artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/audit.json
```

Use `--dry_run` to print the full command chain without running SAM3, and `--stop_after_prepare` to check the downloaded corrections without using a GPU.

Recommended no-GPU preflight before using SAM3:

```bash
/mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  scripts/maskwam_run_full_correction_loop.py \
  --review_dir artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click \
  --downloaded_corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json \
  --run_name repropagated_qa16_preflight \
  --python /mnt/workspace1/users/niejunnan/envs/sam3/bin/python \
  --cuda_visible_devices 0 \
  --sam3_version sam3 \
  --require_all_actionable \
  --stop_after_prepare
```

This preflight must report `ready_for_repropagate=true` before running the full GPU path. The preparation script no longer falls back to stale `corrections_merged.json` files in the review directory; formal runs must use either `--downloaded_corrections .../corrections_from_html.json` or a freshly saved `corrections_from_html.json` in the review directory.

For debugging, the lower-level preparation wrapper can still be run directly. It merges the draft and browser JSON, runs coverage, runs validation, writes all reports, and prints the next re-propagation command:

```bash
python scripts/maskwam_prepare_corrections_for_repropagate.py \
  --review_dir artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click \
  --downloaded_corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json \
  --output_corrections artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/corrections_merged.json \
  --summary_json artifacts/maskwam_libero_annotation_debug16/corrected_runs/repropagated_qa16/corrections_prepare_summary.json \
  --require_all_critical
```

It writes:

```text
corrections_merged.json
corrections_merged_coverage.json
corrections_merged_coverage.csv
corrections_merged_validation.json
corrections_prepare_summary.json
```

`ready_for_repropagate=true` means the merged corrections are valid, have active prompts, and cover all critical actionable tasks. High and medium tasks are still reported, but they are not a hard gate unless `--require_all_high` or `--require_all_actionable` is added.

If corrections come from multiple sources and you want manual control, normalize them directly:

```bash
python scripts/maskwam_merge_corrections.py \
  --inputs \
    artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_tasks_draft.json \
    artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_from_html.json \
  --output artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged.json
```

By default, the merger drops empty draft prompts, deduplicates identical prompts, sorts prompts by sample/frame/role/name, and keeps only active point/box corrections. Use the merged file as the input to `maskwam_run_corrected_pipeline.py`.

The orchestrator runs coverage automatically when `--correction_tasks` points to the generated TODO file. To run the same check manually:

```bash
python scripts/maskwam_check_correction_coverage.py \
  --tasks artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_tasks.json \
  --corrections artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged.json \
  --require_all_critical \
  --output_json artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged_coverage.json \
  --output_csv artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/corrections_merged_coverage.csv
```

Coverage semantics:

```text
object / target / robot tasks are actionable human corrections
final_train_mask is an aggregate task, not a directly corrected prompt
manual role-only corrections can cover tasks with the same sample and role
unmatched active corrections are reported, because they are legal but do not fix any current TODO
```

Current coverage smoke results:

```text
empty correction draft:
  actionable=0/24
  active_corrections=0

sample000 box smoke:
  actionable=0/24
  active_corrections=1
  unmatched_active=1
  reason: sample000 has no current correction task

sample002 target plate fixture:
  actionable=1/24
  active_corrections=1
  unmatched_active=0
  covered: critical sample002 target:plate
```

Merge smoke:

```text
inputs:
  corrections_tasks_draft.json
  sample000_object_box.json

output:
  artifacts/maskwam_libero_annotation_debug16/correction_smoke/merged_draft_plus_sample000_box.json

result:
  samples=1
  prompts=1
  active=1
  validation=status=valid, matched_active=1, issues=0
```

### Correction Validation Smoke Status

The correction validator has been checked locally and on 7779.

Empty template:

```text
status=valid
active=0
issues=0
```

Sample 000 box smoke:

```text
status=valid
active=1
matched_active=1
issues=0
```

Intentional invalid box fixture:

```text
status=invalid
ISSUE sample 0 prompt 0: box out of bounds for 224x448: [210, 430, 50, 40]
```

Local validation reports:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_first_pass_qa16_click/correction_validation_template.json
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/correction_smoke/sample000_object_box_validation.json
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/correction_smoke/invalid_out_of_bounds_box_validation.json
```

### Re-Propagation Smoke Status

Two correction smoke tests have been run on 7779 for `sample_000`.

Box correction output:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/repropagate_smoke_sample000_box_v2
```

Local review pack:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_repropagate_smoke_sample000_box_v2/index.html
```

Verifier status:

```text
valid_unapproved
```

Point correction output:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/repropagate_smoke_sample000_point_v3
```

Local review pack:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/review_repropagate_smoke_sample000_point_v3/index.html
```

Verifier status:

```text
valid_unapproved
```

Observed smoke behavior:

- The explicit box correction was accepted by SAM3 but SAM3 returned an all-zero object mask, so the manual box fallback was used.
- After passing `start_frame_index`, point correction can run through SAM3 without fallback.
- A single positive point over-segmented the table heavily in `sample_000`, so small LIBERO objects should use a tight box or positive point plus negative background points, not just one positive point.
- In both cases, `object`, `target`, `robot`, and `final_train_mask` produced complete 17-frame mask sequences with no zero frames for `final_train_mask`.

This means the engineering pipeline is now usable for the debug16 human-QA loop, but mask quality still needs human inspection before scaling.

### Train Manifest Smoke Status

The exporter has been validated on 7779.

Full corrected-run orchestrator smoke:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/sample000_box_orchestrator_smoke
```

Local copy:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/corrected_runs/sample000_box_orchestrator_smoke
```

Pipeline status:

```text
pipeline_summary.json: status=complete
correction_validation.json: status=valid, active=1, matched_active=1
review/verification.json: status=valid_unapproved
train_mask_sequences.json: status=ready, ready=1/1, issues=0, warnings=1
```

Point correction smoke train manifest:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/train_manifests/repropagate_smoke_sample000_point_v3.json
```

Local copy:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/train_manifests/repropagate_smoke_sample000_point_v3.json
```

Status:

```text
ready=1/1
issues=0
warnings=1
```

The warning is expected for this smoke because the single positive point over-segmented the table:

```text
final_mean_area_ratio too large: 0.669054
```

First-pass 16-case candidate manifest:

```text
/mnt/workspace1/users/niejunnan/codebase/Motus/artifacts/maskwam_libero_annotation_debug16/train_manifests/first_pass_qa16_candidate.json
```

Local copy:

```text
/Users/n/Documents/DreamZero/repos/Motus/artifacts/maskwam_libero_annotation_debug16/train_manifests/first_pass_qa16_candidate.json
```

Status:

```text
ready=15/16
issues=1
warnings=15
```

This candidate is intentionally not accepted as final training data. `sample_002` has zero `final_train_mask` frames, and most samples still have role-level empty masks. It should be corrected through the review HTML and rerun through SAM3 re-propagation before exporting a final training manifest.
