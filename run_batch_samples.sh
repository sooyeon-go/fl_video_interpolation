#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BATCH_ROOT=${BATCH_ROOT:-/data/project-vilab/sy/qwen/VQA_edit/out_batch}
OUT_ROOT=${OUT_ROOT:-"$ROOT_DIR/outputs/batch_cat1"}
FRAMES=${FRAMES:-81}
NUM_GPUS=${NUM_GPUS:-1}
# Comma-separated GPU ids, e.g. 2,3
GPU_IDS=${GPU_IDS:-2,3}
# Comma-separated: wan,ltx,omni  (default: all)
MODELS=${MODELS:-wan,ltx,omni}
# Max concurrent jobs per GPU (1 = one sample at a time per GPU)
JOBS_PER_GPU=${JOBS_PER_GPU:-1}

SOURCE="${BATCH_ROOT}/cat_1__to__cat_3/inputs/image_a.png"

PROMPT_CAT="The cats gently change their poses with a smooth natural motion."
PROMPT_OBJECT="The subject slightly changes its pose with a smooth natural motion."

# name|target_path|prompt
SAMPLES=(
  "cat_1__to__cat_3|${BATCH_ROOT}/cat_1__to__cat_3/inputs/image_b.png|${PROMPT_CAT}"
  "cat_1__to__cat_4|${BATCH_ROOT}/cat_1__to__cat_4/inputs/image_b.png|${PROMPT_CAT}"
  "cat_1__to__dog_2|${BATCH_ROOT}/cat_1__to__dog_2/inputs/image_b.png|${PROMPT_OBJECT}"
  "cat_1__to__dog_3|${BATCH_ROOT}/cat_1__to__dog_3/inputs/image_b.png|${PROMPT_OBJECT}"
  "cat_1__to__dog_4|${BATCH_ROOT}/cat_1__to__dog_4/inputs/image_b.png|${PROMPT_OBJECT}"
  "cat_1__to__dog_5|${BATCH_ROOT}/cat_1__to__dog_5/inputs/image_b.png|${PROMPT_OBJECT}"
  "cat_1__to__dog_6|${BATCH_ROOT}/cat_1__to__dog_6/inputs/image_b.png|${PROMPT_OBJECT}"
  "cat_1__to__dog_7|${BATCH_ROOT}/cat_1__to__dog_7/inputs/image_b.png|${PROMPT_OBJECT}"
)

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPU_WORKERS=${#GPU_LIST[@]}

echo "========================================"
echo "Source : $SOURCE"
echo "Out    : $OUT_ROOT"
echo "GPUs   : ${GPU_LIST[*]}"
echo "Models : ${MODEL_LIST[*]}"
echo "Samples: ${#SAMPLES[@]}"
echo "========================================"

if [[ ! -f "$SOURCE" ]]; then
    echo "Source image not found: $SOURCE" >&2
    exit 1
fi

run_sample_on_gpu() {
    local gpu_id=$1
    local name=$2
    local target=$3
    local prompt=$4
    local out_dir="$OUT_ROOT/$name"
    local log_file="$out_dir/run_gpu${gpu_id}.log"
    local model

    mkdir -p "$out_dir"

    {
        echo "########################################"
        echo "# Sample: $name"
        echo "# GPU   : $gpu_id"
        echo "# Target: $target"
        echo "# Prompt: $prompt"
        echo "########################################"

        export CUDA_VISIBLE_DEVICES="$gpu_id"

        for model in "${MODEL_LIST[@]}"; do
            model="${model// /}"
            case "$model" in
                wan)
                    echo ">>> [GPU $gpu_id][$model] $out_dir/wan.mp4"
                    bash "$ROOT_DIR/run_flf.sh" wan "$SOURCE" "$target" \
                        "$out_dir/wan.mp4" "$prompt" \
                        --frames "$FRAMES" --width 832 --height 480
                    ;;
                ltx)
                    echo ">>> [GPU $gpu_id][$model] $out_dir/ltx.mp4"
                    bash "$ROOT_DIR/run_flf.sh" ltx "$SOURCE" "$target" \
                        "$out_dir/ltx.mp4" "$prompt" \
                        --frames "$FRAMES" --width 832 --height 512
                    ;;
                omni)
                    echo ">>> [GPU $gpu_id][$model] $out_dir/omni.mp4"
                    bash "$ROOT_DIR/run_flf.sh" omni "$SOURCE" "$target" \
                        "$out_dir/omni.mp4" "$prompt" \
                        --frames "$FRAMES" --num-gpus "$NUM_GPUS" --think
                    ;;
                *)
                    echo "Unknown model: $model" >&2
                    exit 2
                    ;;
            esac
            echo ">>> Done: $out_dir/${model}.mp4"
        done
    } 2>&1 | tee "$log_file"
}

# Filter existing targets and round-robin assign to GPUs.
VALID_SAMPLES=()
for entry in "${SAMPLES[@]}"; do
    IFS='|' read -r name target prompt <<< "$entry"
    if [[ ! -f "$target" ]]; then
        echo "Skip missing target: $target" >&2
        continue
    fi
    VALID_SAMPLES+=("$entry")
done

PIDS=()
STATUS_FILES=()
MAX_JOBS=$((NUM_GPU_WORKERS * JOBS_PER_GPU))
idx=0
for entry in "${VALID_SAMPLES[@]}"; do
    IFS='|' read -r name target prompt <<< "$entry"
    gpu_id="${GPU_LIST[$((idx % NUM_GPU_WORKERS))]}"
    gpu_id="${gpu_id// /}"
    status_file="$(mktemp)"
    STATUS_FILES+=("$status_file")

    # Keep at most MAX_JOBS running; wait for any one to finish.
    while (( ${#PIDS[@]} >= MAX_JOBS )); do
        wait -n || true
        alive=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        PIDS=("${alive[@]}")
    done

    echo "Schedule $name -> GPU $gpu_id"
    (
        if run_sample_on_gpu "$gpu_id" "$name" "$target" "$prompt"; then
            echo 0 > "$status_file"
        else
            echo $? > "$status_file"
        fi
    ) &
    PIDS+=("$!")
    idx=$((idx + 1))
done

fail=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

for status_file in "${STATUS_FILES[@]}"; do
    code=$(cat "$status_file" 2>/dev/null || echo 1)
    rm -f "$status_file"
    if [[ "$code" != "0" ]]; then
        fail=1
    fi
done

echo
if [[ "$fail" -eq 0 ]]; then
    echo "Batch finished successfully. Results under: $OUT_ROOT"
else
    echo "Batch finished with errors. Check logs under: $OUT_ROOT/*/run_gpu*.log" >&2
    exit 1
fi
