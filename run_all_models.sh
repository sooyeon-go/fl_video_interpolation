#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

FIRST=${1:-/data/project-vilab/sy/qwen/VQA_edit/out_batch/cat_1__to__cat_2/inputs/image_a.png}
LAST=${2:-/data/project-vilab/sy/qwen/VQA_edit/out_batch/cat_1__to__cat_2/inputs/image_b.png}
PROMPT=${3:-"고양이가 일어나서 눕는다."}
OUT_DIR=${4:-"$ROOT_DIR/outputs/cat_1__to__cat_2"}
FRAMES=${FRAMES:-81}
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p "$OUT_DIR"

echo "========================================"
echo "First : $FIRST"
echo "Last  : $LAST"
echo "Prompt: $PROMPT"
echo "Out   : $OUT_DIR"
echo "========================================"

run_one() {
    local model=$1
    local output=$2
    shift 2
    echo
    echo ">>> Running: $model"
    bash "$ROOT_DIR/run_flf.sh" "$model" "$FIRST" "$LAST" "$output" "$PROMPT" "$@"
    echo ">>> Done: $output"
}

run_one wan "$OUT_DIR/wan.mp4" \
    --frames "$FRAMES" --width 832 --height 480

run_one ltx "$OUT_DIR/ltx.mp4" \
    --frames "$FRAMES" --width 832 --height 512

run_one omni "$OUT_DIR/omni.mp4" \
    --frames "$FRAMES" --num-gpus "$NUM_GPUS" --think

echo
echo "All models finished."
echo "Outputs:"
echo "  $OUT_DIR/wan.mp4"
echo "  $OUT_DIR/ltx.mp4"
echo "  $OUT_DIR/omni.mp4"
