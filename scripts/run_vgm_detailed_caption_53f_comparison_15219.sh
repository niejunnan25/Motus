#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace1/users/niejunnan/codebase/Motus"
PYTHON="/mnt/workspace/users/niejunnan/envs/motus/bin/python"
OUTPUT_ROOT="${ROOT}/eval_outputs/vgm_bridge_detailed_vs_success50_original16_step30000_s50"
LOG_ROOT="${OUTPUT_ROOT}/logs"

V1_CONFIG="configs/vgm_bridge_v1_proper_detailed_caption_v1_53f_full_jitter.yaml"
V2_STATE_CONFIG="configs/vgm_bridge_v2_b_state_detailed_caption_v1_53f_full_jitter.yaml"
V2_PLUS_CONFIG="configs/vgm_bridge_v2_b_plus_state_first_last_delta_detailed_caption_v1_53f_full_jitter.yaml"

V1_BASELINE="${ROOT}/checkpoints/vgm_bridge_v1_proper_detailed_caption_v1_53f_full_jitter/vgm_bridge_v1_proper_lerobot_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"
V1_SUCCESS50="${ROOT}/checkpoints/vgm_bridge_v1_proper_success50_detailed_caption_v1_53f_full_jitter/vgm_bridge_v1_proper_lerobot_success50_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"
V2_STATE_BASELINE="${ROOT}/checkpoints/vgm_bridge_v2_b_state_detailed_caption_v1_53f_full_jitter/vgm_bridge_v2_b_state_lerobot_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"
V2_STATE_SUCCESS50="${ROOT}/checkpoints/vgm_bridge_v2_b_state_success50_detailed_caption_v1_53f_full_jitter/vgm_bridge_v2_b_state_lerobot_success50_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"
V2_PLUS_BASELINE="${ROOT}/checkpoints/vgm_bridge_v2_b_plus_state_first_last_delta_detailed_caption_v1_53f_full_jitter/vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"
V2_PLUS_SUCCESS50="${ROOT}/checkpoints/vgm_bridge_v2_b_plus_state_first_last_delta_success50_detailed_caption_v1_53f_full_jitter/vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_success50_video_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs8_30k/checkpoint_step_30000"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

run_eval() {
  local gpu="$1"
  local slug="$2"
  local config="$3"
  local checkpoint="$4"
  local output_dir="${OUTPUT_ROOT}/${slug}"
  local log_file="${LOG_ROOT}/${slug}.log"

  echo "[$(date '+%F %T')] starting ${slug} on GPU ${gpu}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" train/eval_vgm_bridge_stage1.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output_dir "${output_dir}" \
    --num_samples 16 \
    --batch_size 1 \
    --loss_repeats 1 \
    --num_inference_steps 50 \
    --seed 50 \
    --fps 4 \
    >>"${log_file}" 2>&1
  echo "[$(date '+%F %T')] finished ${slug} on GPU ${gpu}" | tee -a "${log_file}"
}

gpu0_queue() {
  run_eval 0 baseline_v1 "${V1_CONFIG}" "${V1_BASELINE}"
  run_eval 0 baseline_v2_state "${V2_STATE_CONFIG}" "${V2_STATE_BASELINE}"
  run_eval 0 baseline_v2_plus "${V2_PLUS_CONFIG}" "${V2_PLUS_BASELINE}"
}

gpu1_queue() {
  run_eval 1 success50_v1 "${V1_CONFIG}" "${V1_SUCCESS50}"
  run_eval 1 success50_v2_state "${V2_STATE_CONFIG}" "${V2_STATE_SUCCESS50}"
  run_eval 1 success50_v2_plus "${V2_PLUS_CONFIG}" "${V2_PLUS_SUCCESS50}"
}

gpu0_queue &
gpu0_pid=$!
gpu1_queue &
gpu1_pid=$!

wait "${gpu0_pid}"
wait "${gpu1_pid}"
echo "[$(date '+%F %T')] all six evaluations completed"
