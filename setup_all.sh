#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/data/shared-vilab/pretrained_models}

cat <<EOF
This will:
  1) Create conda envs: flf-tools, flf-wan, flf-omni, flf-ltx
  2) Download models under: ${MODEL_ROOT}
     - Wan2.2-I2V-A14B-Diffusers
     - HY-OmniWeaving (+ auxiliary encoders)
     - LTX-2.3/gemma-3-12b

OmniWeaving requires approved Hugging Face access to
black-forest-labs/FLUX.1-Redux-dev.
EOF

"$ROOT_DIR/setup_envs.sh" all

echo
echo "If needed, log in to Hugging Face for gated models:"
echo "  conda run -n flf-tools hf auth login"
echo

MODEL_ROOT="$MODEL_ROOT" "$ROOT_DIR/prepare_models.sh" all

echo "All conda environments and models are ready."
echo "Model root: $MODEL_ROOT"
