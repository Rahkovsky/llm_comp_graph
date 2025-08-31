#!/usr/bin/env bash
# macOS Metal setup for llama.cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${PROJECT_ROOT}/llama_env.sh"
source "${SCRIPT_DIR}/common_setup.sh"

# Install dependencies
command -v brew >/dev/null 2>&1 || { echo "Install Homebrew first: https://brew.sh/"; exit 1; }
brew install cmake ninja git >/dev/null 2>&1

# Setup environment
setup_python
setup_uv
setup_venv

# Build llama.cpp
clone_llama
log "Building with Metal support..."
cmake -S "${LLAMA_INSTALL_DIR}" -B "${LLAMA_FULL_BUILD_DIR}" \
    -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
ninja -C "${LLAMA_FULL_BUILD_DIR}" >/dev/null 2>&1

# Setup model and test
download_model
test_install
show_complete
