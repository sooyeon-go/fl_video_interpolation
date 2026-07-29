#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
    cat <<'EOF'
Usage:
  ./run_flf.sh MODEL FIRST_FRAME LAST_FRAME OUTPUT PROMPT [extra options]

MODEL:
  omni | ltx | wan

Examples:
  ./run_flf.sh ltx first.png last.png outputs/ltx.mp4 \
    "A woman turns toward the camera and smiles."

  ./run_flf.sh wan first.png last.png outputs/wan.mp4 \
    "The camera slowly moves forward." --width 832 --height 480

  ./run_flf.sh omni first.png last.png outputs/omni.mp4 \
    "The person naturally walks across the room." --think --num-gpus 8

Default model root:
  /data/shared-vilab/pretrained_models

Environment variables:
  FLF_PYTHON       Fallback Python executable
  OMNI_PYTHON      Override OmniWeaving Python
  LTX_PYTHON       Override LTX Python
  WAN_PYTHON       Override Wan Python
  OMNI_MODEL_PATH  OmniWeaving checkpoint directory
  LTX_MODEL_PATH   LTX-2.3 checkpoint directory
  WAN_MODEL_PATH   Wan Diffusers checkpoint directory
  OMNI_REPO_PATH   Cloned OmniWeaving repository
  LTX_GEMMA_PATH   Gemma-3 text encoder directory
  ENV_WAN / ENV_OMNI / ENV_LTX  conda env names
EOF
    exit 2
fi

MODEL=$1
FIRST_FRAME=$2
LAST_FRAME=$3
OUTPUT=$4
PROMPT=$5
shift 5

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/data/shared-vilab/pretrained_models}
DEFAULT_PYTHON=${FLF_PYTHON:-python}

ENV_WAN=${ENV_WAN:-flf-wan}
ENV_OMNI=${ENV_OMNI:-flf-omni}
ENV_LTX=${ENV_LTX:-flf-ltx}

conda_env_python() {
    local env_name=$1
    local python_path
    if ! command -v conda >/dev/null 2>&1; then
        return 1
    fi
    python_path="$(conda info --base)/envs/${env_name}/bin/python"
    if [[ -x "$python_path" ]]; then
        printf '%s\n' "$python_path"
        return 0
    fi
    return 1
}

case "$MODEL" in
    omni)
        AUTO_PYTHON="$(conda_env_python "$ENV_OMNI" || true)"
        [[ -n "${AUTO_PYTHON:-}" ]] || AUTO_PYTHON=$DEFAULT_PYTHON
        BACKEND_PYTHON=${OMNI_PYTHON:-$AUTO_PYTHON}
        MODEL_PATH=${OMNI_MODEL_PATH:-$MODEL_ROOT/HY-OmniWeaving}
        REPO_ARGS=(--repo-path "${OMNI_REPO_PATH:-$ROOT_DIR/third_party/OmniWeaving}")
        EXTRA_PATH_ARGS=()
        ;;
    ltx)
        AUTO_PYTHON="$(conda_env_python "$ENV_LTX" || true)"
        [[ -n "${AUTO_PYTHON:-}" ]] || AUTO_PYTHON=$DEFAULT_PYTHON
        BACKEND_PYTHON=${LTX_PYTHON:-$AUTO_PYTHON}
        MODEL_PATH=${LTX_MODEL_PATH:-$MODEL_ROOT/LTX-2.3}
        REPO_ARGS=()
        EXTRA_PATH_ARGS=()
        if [[ -n "${LTX_GEMMA_PATH:-}" ]]; then
            EXTRA_PATH_ARGS+=(--ltx-gemma-path "$LTX_GEMMA_PATH")
        elif [[ -d "$MODEL_ROOT/LTX-2.3/gemma-3-12b" ]]; then
            EXTRA_PATH_ARGS+=(--ltx-gemma-path "$MODEL_ROOT/LTX-2.3/gemma-3-12b")
        fi
        ;;
    wan)
        AUTO_PYTHON="$(conda_env_python "$ENV_WAN" || true)"
        [[ -n "${AUTO_PYTHON:-}" ]] || AUTO_PYTHON=$DEFAULT_PYTHON
        BACKEND_PYTHON=${WAN_PYTHON:-$AUTO_PYTHON}
        MODEL_PATH=${WAN_MODEL_PATH:-$MODEL_ROOT/Wan2.2-I2V-A14B-Diffusers}
        REPO_ARGS=()
        EXTRA_PATH_ARGS=()
        ;;
    *)
        echo "Unknown model '$MODEL'; expected omni, ltx, or wan." >&2
        exit 2
        ;;
esac

exec "$BACKEND_PYTHON" "$ROOT_DIR/generate_flf.py" \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --python "$BACKEND_PYTHON" \
    --first-frame "$FIRST_FRAME" \
    --last-frame "$LAST_FRAME" \
    --output "$OUTPUT" \
    --prompt "$PROMPT" \
    "${REPO_ARGS[@]}" \
    "${EXTRA_PATH_ARGS[@]}" \
    "$@"
