#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET=${1:-all}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

ENV_TOOLS=${ENV_TOOLS:-flf-tools}
ENV_WAN=${ENV_WAN:-flf-wan}
ENV_OMNI=${ENV_OMNI:-flf-omni}
ENV_LTX=${ENV_LTX:-flf-ltx}

usage() {
    cat <<'EOF'
Usage:
  ./setup_envs.sh all
  ./setup_envs.sh wan
  ./setup_envs.sh omni
  ./setup_envs.sh ltx
  ./setup_envs.sh tools

Optional:
  PYTHON_VERSION=3.12 ./setup_envs.sh all
  ENV_WAN=my-wan-env ./setup_envs.sh wan

Creates conda environments:
  flf-tools  Hugging Face / ModelScope download CLIs
  flf-wan    Wan 2.2 Diffusers FLF
  flf-omni   OmniWeaving interpolation
  flf-ltx    LTX-2.3 keyframe interpolation

FlashAttention/SageAttention are not compiled automatically.
EOF
}

find_conda_sh() {
    local candidate
    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    for candidate in \
        /opt/conda/etc/profile.d/conda.sh \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        /usr/local/miniconda3/etc/profile.d/conda.sh
    do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

CONDA_SH="$(find_conda_sh || true)"
if [[ -z "${CONDA_SH:-}" ]]; then
    echo "conda was not found. Install Miniconda/Anaconda first," >&2
    echo "or run: source /opt/conda/etc/profile.d/conda.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_SH"

conda_python() {
    local env_name=$1
    local python_path
    python_path="$(conda info --base)/envs/${env_name}/bin/python"
    if [[ ! -x "$python_path" ]]; then
        echo "conda env python not found: $python_path" >&2
        exit 1
    fi
    printf '%s\n' "$python_path"
}

ensure_conda_env() {
    local env_name=$1
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "Using existing conda env: $env_name"
    else
        echo "Creating conda env: $env_name (Python ${PYTHON_VERSION})"
        conda create -y -n "$env_name" "python=${PYTHON_VERSION}"
    fi
}

clone_repo() {
    local url=$1
    local destination=$2
    if [[ ! -d "$destination/.git" ]]; then
        git clone "$url" "$destination"
    fi
}

setup_tools() {
    ensure_conda_env "$ENV_TOOLS"
    local python
    python="$(conda_python "$ENV_TOOLS")"
    "$python" -m pip install --upgrade pip
    "$python" -m pip install --upgrade \
        "huggingface_hub[cli]" modelscope
}

setup_wan() {
    ensure_conda_env "$ENV_WAN"
    local python
    python="$(conda_python "$ENV_WAN")"
    "$python" -m pip install --upgrade pip
    "$python" -m pip install \
        torch==2.6.0 torchvision==0.21.0 \
        transformers accelerate imageio imageio-ffmpeg pillow sentencepiece \
        ftfy protobuf regex
    "$python" -m pip install --upgrade \
        "diffusers @ git+https://github.com/huggingface/diffusers.git"
}

setup_omni() {
    local repo="$ROOT_DIR/third_party/OmniWeaving"
    mkdir -p "$ROOT_DIR/third_party"
    clone_repo https://github.com/Tencent-Hunyuan/OmniWeaving.git "$repo"

    ensure_conda_env "$ENV_OMNI"
    local python
    python="$(conda_python "$ENV_OMNI")"
    "$python" -m pip install --upgrade pip wheel setuptools
    "$python" -m pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 ftfy
    "$python" -m pip install -r "$repo/requirements.txt"
}

setup_ltx() {
    local repo="$ROOT_DIR/third_party/LTX-2"
    mkdir -p "$ROOT_DIR/third_party"
    clone_repo https://github.com/Lightricks/LTX-2.git "$repo"

    ensure_conda_env "$ENV_LTX"
    local python
    python="$(conda_python "$ENV_LTX")"
    "$python" -m pip install --upgrade pip wheel setuptools
    "$python" -m pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 ftfy
    # Install LTX packages into the conda env instead of uv's local .venv.
    "$python" -m pip install -e "$repo/packages/ltx-core"
    "$python" -m pip install -e "$repo/packages/ltx-pipelines"
    # ltx deps can pull mismatched torchvision; force a compatible pair again.
    "$python" -m pip install --force-reinstall --no-deps \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
    "$python" -m pip install --upgrade \
        "transformers>=4.51.0" accelerate pillow ftfy
}

case "$TARGET" in
    all)
        setup_tools
        setup_wan
        setup_omni
        setup_ltx
        ;;
    tools)
        setup_tools
        ;;
    wan)
        setup_wan
        ;;
    omni)
        setup_omni
        ;;
    ltx)
        setup_ltx
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

echo
echo "Environment setup complete: $TARGET"
echo "Activate examples:"
echo "  conda activate $ENV_TOOLS"
echo "  conda activate $ENV_WAN"
echo "  conda activate $ENV_OMNI"
echo "  conda activate $ENV_LTX"
