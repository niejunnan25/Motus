#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace1/users/niejunnan/codebase/Motus"
PYTHON="/mnt/workspace/users/niejunnan/envs/motus/bin/python"
OUTPUT_ROOT="${ROOT}/eval_outputs/vgm_role_mask_8way_step30000_s50_16samples"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MANIFEST="${ROOT}/configs/vgm_role_mask_8way_step30000_s50_16samples.json"
COMMON_ARGS=(--num_samples 16 --batch_size 1 --loss_repeats 1 --num_inference_steps 50 --seed 50 --fps 4)

export WANDB_MODE=offline
export HF_HOME="/mnt/workspace1/users/niejunnan/huggingface"
export HF_DATASETS_CACHE="/mnt/workspace1/users/niejunnan/huggingface/datasets"
export XDG_CACHE_HOME="/mnt/workspace1/users/niejunnan/.cache"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

checkpoint() {
  local config_stem="$1"
  local run_name="$2"
  printf '%s/checkpoints/%s/%s/checkpoint_step_30000' "${ROOT}" "${config_stem}" "${run_name}"
}

run_eval() {
  local gpu="$1"
  local slug="$2"
  local config="$3"
  local checkpoint_path="$4"
  local log_file="${LOG_ROOT}/${slug}.log"
  if [[ ! -d "${checkpoint_path}" ]]; then
    echo "Missing checkpoint: ${checkpoint_path}" >&2
    return 1
  fi
  echo "[$(date '+%F %T')] start ${slug} gpu=${gpu}" | tee "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" train/eval_vgm_bridge_stage1.py \
    --config "${config}" \
    --checkpoint "${checkpoint_path}" \
    --output_dir "${OUTPUT_ROOT}/${slug}" \
    "${COMMON_ARGS[@]}" >>"${log_file}" 2>&1
  echo "[$(date '+%F %T')] finish ${slug} gpu=${gpu}" | tee -a "${log_file}"
}

MOSAIC_BINARY_V1_CONFIG="configs/vgm_bridge_v1_proper_success50_role_mask_mosaic_binary_detailed_caption_v1_53f_full_jitter.yaml"
MOSAIC_BINARY_V2_CONFIG="configs/vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_mosaic_binary_detailed_caption_v1_53f_full_jitter.yaml"
MOSAIC_COLOR_V1_CONFIG="configs/vgm_bridge_v1_proper_success50_role_mask_mosaic_color_detailed_caption_v1_53f_full_jitter.yaml"
MOSAIC_COLOR_V2_CONFIG="configs/vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_mosaic_color_detailed_caption_v1_53f_full_jitter.yaml"
CHANNEL_BINARY_V1_CONFIG="configs/vgm_bridge_v1_proper_success50_role_mask_channel_binary_detailed_caption_v1_53f_full_jitter.yaml"
CHANNEL_BINARY_V2_CONFIG="configs/vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_channel_binary_detailed_caption_v1_53f_full_jitter.yaml"
CHANNEL_COLOR_V1_CONFIG="configs/vgm_bridge_v1_proper_success50_role_mask_channel_color_detailed_caption_v1_53f_full_jitter.yaml"
CHANNEL_COLOR_V2_CONFIG="configs/vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_channel_color_detailed_caption_v1_53f_full_jitter.yaml"

MOSAIC_BINARY_V1_CKPT="$(checkpoint vgm_bridge_v1_proper_success50_role_mask_mosaic_binary_detailed_caption_v1_53f_full_jitter vgm_bridge_v1_proper_lerobot_success50_role_mask_mosaic_binary_448x448_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
MOSAIC_BINARY_V2_CKPT="$(checkpoint vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_mosaic_binary_detailed_caption_v1_53f_full_jitter vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_success50_role_mask_mosaic_binary_448x448_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
MOSAIC_COLOR_V1_CKPT="$(checkpoint vgm_bridge_v1_proper_success50_role_mask_mosaic_color_detailed_caption_v1_53f_full_jitter vgm_bridge_v1_proper_lerobot_success50_role_mask_mosaic_color_448x448_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
MOSAIC_COLOR_V2_CKPT="$(checkpoint vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_mosaic_color_detailed_caption_v1_53f_full_jitter vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_success50_role_mask_mosaic_color_448x448_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
CHANNEL_BINARY_V1_CKPT="$(checkpoint vgm_bridge_v1_proper_success50_role_mask_channel_binary_detailed_caption_v1_53f_full_jitter vgm_bridge_v1_proper_lerobot_success50_role_mask_channel_binary_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
CHANNEL_BINARY_V2_CKPT="$(checkpoint vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_channel_binary_detailed_caption_v1_53f_full_jitter vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_success50_role_mask_channel_binary_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
CHANNEL_COLOR_V1_CKPT="$(checkpoint vgm_bridge_v1_proper_success50_role_mask_channel_color_detailed_caption_v1_53f_full_jitter vgm_bridge_v1_proper_lerobot_success50_role_mask_channel_color_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"
CHANNEL_COLOR_V2_CKPT="$(checkpoint vgm_bridge_v2_b_plus_state_first_last_delta_success50_role_mask_channel_color_detailed_caption_v1_53f_full_jitter vgm_bridge_v2_b_plus_state_first_last_delta_lerobot_success50_role_mask_channel_color_448x224_detailed_caption_v1_53f_full_jitter_2gpu_gbs4_ddpfix_datashardfix_30k)"

gpu0_queue() {
  run_eval 0 mosaic_binary_v1 "${MOSAIC_BINARY_V1_CONFIG}" "${MOSAIC_BINARY_V1_CKPT}"
  run_eval 0 mosaic_color_v2plus "${MOSAIC_COLOR_V2_CONFIG}" "${MOSAIC_COLOR_V2_CKPT}"
  run_eval 0 channel_binary_v2plus "${CHANNEL_BINARY_V2_CONFIG}" "${CHANNEL_BINARY_V2_CKPT}"
  run_eval 0 channel_color_v1 "${CHANNEL_COLOR_V1_CONFIG}" "${CHANNEL_COLOR_V1_CKPT}"
}

gpu1_queue() {
  run_eval 1 mosaic_binary_v2plus "${MOSAIC_BINARY_V2_CONFIG}" "${MOSAIC_BINARY_V2_CKPT}"
  run_eval 1 mosaic_color_v1 "${MOSAIC_COLOR_V1_CONFIG}" "${MOSAIC_COLOR_V1_CKPT}"
  run_eval 1 channel_binary_v1 "${CHANNEL_BINARY_V1_CONFIG}" "${CHANNEL_BINARY_V1_CKPT}"
  run_eval 1 channel_color_v2plus "${CHANNEL_COLOR_V2_CONFIG}" "${CHANNEL_COLOR_V2_CKPT}"
}

gpu0_queue &
gpu0_pid=$!
gpu1_queue &
gpu1_pid=$!
set +e
wait "${gpu0_pid}"
gpu0_status=$?
wait "${gpu1_pid}"
gpu1_status=$?
set -e
if [[ ${gpu0_status} -ne 0 || ${gpu1_status} -ne 0 ]]; then
  echo "Evaluation queue failed: gpu0=${gpu0_status}, gpu1=${gpu1_status}" >&2
  exit 1
fi

"${PYTHON}" scripts/build_vgm_role_mask_eval_details.py \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_ROOT}/summary/details" \
  --frames_per_block 18 \
  --full_frames_per_block 9

date '+completed_at=%F %T' >"${OUTPUT_ROOT}/_SUCCESS"
echo "[$(date '+%F %T')] all evaluations and detail rendering completed"
