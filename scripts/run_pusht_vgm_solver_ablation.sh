#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR [OUTPUT_ROOT]" >&2
  exit 2
fi

checkpoint=$1
output_root=${2:-eval_outputs/pusht_vgm_multimodal}
python_bin=/mnt/workspace/users/niejunnan/envs/motus/bin/python
config=configs/vgm_bridge_pusht_v1_proper_17f_eval.yaml
steps=(55 50 20 10 4 1)
gpus=(0 1 2 3 4 5)
pids=()

mkdir -p "${output_root}/logs"

for index in "${!steps[@]}"; do
  step=${steps[$index]}
  gpu=${gpus[$index]}
  step_name=$(printf "%03d" "${step}")
  log_path="${output_root}/logs/steps_${step_name}.log"
  echo "Launching ${step} denoising steps on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 "${python_bin}" \
    scripts/evaluate_pusht_vgm_multimodal.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --output_dir "${output_root}/steps_${step_name}" \
    --num_inference_steps "${step}" \
    --episode_indices 0,7,14,21 \
    --seeds 0,1,2,3,4,5,6,7 \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Evaluation with ${steps[$index]} steps failed" >&2
    failed=1
  fi
done

if [[ ${failed} -ne 0 ]]; then
  exit 1
fi

"${python_bin}" scripts/summarize_pusht_vgm_multimodal.py \
  --input_root "${output_root}" \
  --output_dir "${output_root}/summary"

echo "All solver-step evaluations completed: ${output_root}"
