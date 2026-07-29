#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/data/shared-vilab/pretrained_models}
TARGET=${1:-}
ENV_TOOLS=${ENV_TOOLS:-flf-tools}

usage() {
    cat <<'EOF'
Usage:
  ./prepare_models.sh all        # Wan Diffusers + Omni + LTX Gemma
  ./prepare_models.sh omni       # OmniWeaving and auxiliary encoders
  ./prepare_models.sh wan        # Wan 2.2 Diffusers-format weights
  ./prepare_models.sh ltx        # LTX Gemma text encoder
  ./prepare_models.sh omni-code  # Clone OmniWeaving code only

Models are downloaded under:
  /data/shared-vilab/pretrained_models

Override with MODEL_ROOT=/path/to/models if needed.

OmniWeaving depends on gated FLUX.1-Redux-dev. Accept the license on Hugging Face
and run: conda run -n flf-tools hf auth login
EOF
}

if [[ -z "$TARGET" ]]; then
    usage
    exit 2
fi

init_conda() {
    local candidate
    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
        if [[ -f "$candidate" ]]; then
            # shellcheck disable=SC1091
            source "$candidate"
            return 0
        fi
    fi
    for candidate in \
        /opt/conda/etc/profile.d/conda.sh \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh"
    do
        if [[ -f "$candidate" ]]; then
            # shellcheck disable=SC1091
            source "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_tools() {
    local conda_base
    if ! init_conda; then
        echo "conda was not found. Run: bash setup_envs.sh tools" >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    TOOLS_BIN="${conda_base}/envs/${ENV_TOOLS}/bin"
    if [[ ! -x "$TOOLS_BIN/hf" ]]; then
        echo "Hugging Face CLI missing in conda env '${ENV_TOOLS}'." >&2
        echo "Run: ./setup_envs.sh tools" >&2
        exit 1
    fi
    HF_CMD="$TOOLS_BIN/hf"
    if [[ -x "$TOOLS_BIN/modelscope" ]]; then
        MODELSCOPE_CMD="$TOOLS_BIN/modelscope"
    else
        echo "ModelScope CLI missing in conda env '${ENV_TOOLS}'." >&2
        echo "Run: ./setup_envs.sh tools" >&2
        exit 1
    fi
}

clone_omni() {
    mkdir -p "$ROOT_DIR/third_party"
    if [[ ! -d "$ROOT_DIR/third_party/OmniWeaving/.git" ]]; then
        git clone https://github.com/Tencent-Hunyuan/OmniWeaving.git \
            "$ROOT_DIR/third_party/OmniWeaving"
    fi
}

clone_ltx() {
    mkdir -p "$ROOT_DIR/third_party"
    if [[ ! -d "$ROOT_DIR/third_party/LTX-2/.git" ]]; then
        git clone https://github.com/Lightricks/LTX-2.git \
            "$ROOT_DIR/third_party/LTX-2"
    fi
}

download_omni() {
    resolve_tools
    clone_omni
    local target="$MODEL_ROOT/HY-OmniWeaving"
    mkdir -p "$target"
    echo "Downloading OmniWeaving into $target"
    "$HF_CMD" download tencent/HY-OmniWeaving --local-dir "$target"
    "$HF_CMD" download Qwen/Qwen2.5-VL-7B-Instruct \
        --local-dir "$target/text_encoder/llm"
    "$HF_CMD" download google/byt5-small \
        --local-dir "$target/text_encoder/byt5-small"
    "$MODELSCOPE_CMD" download --model AI-ModelScope/Glyph-SDXL-v2 \
        --local_dir "$target/text_encoder/Glyph-SDXL-v2"
    "$HF_CMD" download black-forest-labs/FLUX.1-Redux-dev \
        --local-dir "$target/vision_encoder/siglip"
}

download_wan() {
    resolve_tools
    mkdir -p "$MODEL_ROOT"
    local target="$MODEL_ROOT/Wan2.2-I2V-A14B-Diffusers"
    echo "Downloading Wan Diffusers weights into $target"
    "$HF_CMD" download Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --local-dir "$target"
}

download_ltx_support() {
    resolve_tools
    clone_ltx
    mkdir -p "$MODEL_ROOT/LTX-2.3"
    local target="$MODEL_ROOT/LTX-2.3/gemma-3-12b"
    echo "Downloading LTX Gemma encoder into $target"
    "$HF_CMD" download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir "$target"
}

case "$TARGET" in
    all)
        download_wan
        download_omni
        download_ltx_support
        ;;
    omni-code)
        clone_omni
        ;;
    omni)
        download_omni
        ;;
    wan)
        download_wan
        ;;
    ltx)
        download_ltx_support
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

echo
echo "Model download complete under: $MODEL_ROOT"
