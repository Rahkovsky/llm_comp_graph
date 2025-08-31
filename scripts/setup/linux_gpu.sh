#!/usr/bin/env bash
# Linux GPU setup for llama.cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${PROJECT_ROOT}/llama_env.sh"
source "${SCRIPT_DIR}/common_setup.sh"

# Check GPU
command -v nvidia-smi >/dev/null 2>&1 || error "NVIDIA GPU/drivers not found"

# Install dependencies
sudo dnf install -y git cmake ninja-build python3.13 python3.13-devel >/dev/null 2>&1 || true

# Setup environment
setup_python
setup_uv
setup_venv

# Build llama.cpp
clone_llama
log "Building with CUDA support..."
cmake -S "${LLAMA_INSTALL_DIR}" -B "${LLAMA_FULL_BUILD_DIR}" \
    -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
ninja -C "${LLAMA_FULL_BUILD_DIR}" >/dev/null 2>&1

# Setup model and test
download_model
test_install
show_complete
