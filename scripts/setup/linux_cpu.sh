#!/usr/bin/env bash
# Linux CPU setup for llama.cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${PROJECT_ROOT}/llama_env.sh"
source "${SCRIPT_DIR}/common_setup.sh"

# Install dependencies
sudo dnf install -y git cmake python3.13 python3.13-devel >/dev/null 2>&1 || true

# Setup environment
setup_python
setup_uv
setup_venv

# Build llama.cpp
clone_llama
log "Building for CPU..."
cmake -S "${LLAMA_INSTALL_DIR}" -B "${LLAMA_FULL_BUILD_DIR}" \
    -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build "${LLAMA_FULL_BUILD_DIR}" -j"$(nproc)" >/dev/null 2>&1

# Setup model and test
download_model
test_install
show_complete
