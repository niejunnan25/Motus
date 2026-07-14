#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released Robo-Dopamine-Bench LIBERO protocol on four GPUs.
# Paths can be overridden through environment variables without editing this file.
REPO=${ROBODOPAMINE_REPO:-/mnt/workspace1/users/niejunnan/codebase/Robo-Dopamine}
BENCH=${ROBODOPAMINE_BENCH:-/mnt/workspace/users/niejunnan/assets/Robo-Dopamine-Bench}
MODEL_ROOT=${ROBODOPAMINE_MODEL_ROOT:-/mnt/workspace/users/niejunnan/assets}
PYTHON=${ROBODOPAMINE_PYTHON:-/mnt/workspace/users/niejunnan/envs/robo-dopamine/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/workspace1/users/niejunnan/eval_outputs/robo_dopamine_official_libero}
GPU_LIST=${GPU_LIST:-0,1,2,3}
INTERVAL=${INTERVAL:-30}
INVERSE_MODES=${INVERSE_MODES:-"false true"}
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
read -r -a INVERSE_MODE_ARGS <<<"$INVERSE_MODES"

LABELS=(grm2_4b_preview grm2_8b_preview grm3b grm8b)
MODELS=(
  Robo-Dopamine-GRM-2.0-4B-Preview
  Robo-Dopamine-GRM-2.0-8B-Preview
  Robo-Dopamine-GRM-3B
  Robo-Dopamine-GRM-8B
)

if [[ ${#GPUS[@]} -ne ${#MODELS[@]} ]]; then
  echo "GPU_LIST must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi
for path in "$REPO/eval/evaluation_grm.py" "$BENCH/jsons/libero_test_100.json" "$PYTHON"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
for model in "${MODELS[@]}"; do
  [[ -d "$MODEL_ROOT/$model" ]] || { echo "Missing model: $MODEL_ROOT/$model" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/logs"
NVIDIA_LIBS=$(find "$(dirname "$PYTHON")/../lib/python3.10/site-packages/nvidia" \
  -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd: -)

pids=()
for index in "${!MODELS[@]}"; do
  gpu=${GPUS[$index]}
  label=${LABELS[$index]}
  model=${MODELS[$index]}
  log="$OUTPUT_ROOT/logs/$label.log"
  mkdir -p "$OUTPUT_ROOT/$label" "/tmp/vllm-cache-$label" "/tmp/torchinductor-$label"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    export VLLM_ATTENTION_BACKEND=TORCH_SDPA
    export VLLM_CACHE_ROOT="/tmp/vllm-cache-$label"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$label"
    export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$PYTHON" -m eval.evaluation_grm \
      --model_path "$MODEL_ROOT/$model" \
      --input_json_dir "$BENCH/jsons" \
      --base_dir "$BENCH/images" \
      --out_root_dir "$OUTPUT_ROOT/$label" \
      --interval "$INTERVAL" \
      --batch_size 16 \
      --evaluation_list libero_test_100 \
      --inverse_modes "${INVERSE_MODE_ARGS[@]}"
  ) >"$log" 2>&1 &
  pids+=("$!")
  echo "$label gpu=$gpu pid=$! log=$log"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${LABELS[$index]} failed; inspect $OUTPUT_ROOT/logs/${LABELS[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
