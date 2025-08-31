#!/usr/bin/env bash
# Llama.cpp Environment Configuration
# This file contains environment variables for Llama installation and usage

# Base directories
export LLAMA_PROJECT_ROOT="${PWD}"
export LLAMA_INSTALL_DIR="${LLAMA_PROJECT_ROOT}/llama"
export LLAMA_MODELS_DIR="${LLAMA_PROJECT_ROOT}/models"
export LLAMA_BUILD_DIR="${LLAMA_INSTALL_DIR}/build"

# Executable paths
export LLAMA_CLI="${LLAMA_BUILD_DIR}/bin/llama-cli"
export LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"

# Model file
export LLAMA_MODEL="${LLAMA_MODELS_DIR}/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"

# Build configuration
export LLAMA_CMAKE_BUILD_TYPE="Release"
export LLAMA_CCACHE_ENABLED="ON"

# Platform-specific settings
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    export LLAMA_METAL_ENABLED="ON"
    export LLAMA_CUDA_ENABLED="OFF"
    export LLAMA_BUILD_SUFFIX="metal"
elif command -v nvidia-smi >/dev/null 2>&1; then
    # Linux with NVIDIA GPU
    export LLAMA_METAL_ENABLED="OFF"
    export LLAMA_CUDA_ENABLED="ON"
    export LLAMA_BUILD_SUFFIX="gpu"
else
    # Linux CPU-only
    export LLAMA_METAL_ENABLED="OFF"
    export LLAMA_CUDA_ENABLED="OFF"
    export LLAMA_BUILD_SUFFIX="cpu"
fi

# Full build directory path
export LLAMA_FULL_BUILD_DIR="${LLAMA_BUILD_DIR}_${LLAMA_BUILD_SUFFIX}"

# Update executable paths with correct build suffix
export LLAMA_CLI="${LLAMA_FULL_BUILD_DIR}/bin/llama-cli"
export LLAMA_SERVER="${LLAMA_FULL_BUILD_DIR}/bin/llama-server"

# Common flags for llama-cli
export LLAMA_COMMON_FLAGS="-m ${LLAMA_MODEL} --color --repeat_penalty 1.1 -c 4096"

# GPU-specific flags (when applicable)
if [[ "$LLAMA_CUDA_ENABLED" == "ON" ]] || [[ "$LLAMA_METAL_ENABLED" == "ON" ]]; then
    export LLAMA_GPU_FLAGS="-ngl 999"
else
    export LLAMA_GPU_FLAGS=""
fi

# Display configuration
echo "=== Llama.cpp Environment Configuration ==="
echo "Project Root: ${LLAMA_PROJECT_ROOT}"
echo "Install Directory: ${LLAMA_INSTALL_DIR}"
echo "Models Directory: ${LLAMA_MODELS_DIR}"
echo "Build Directory: ${LLAMA_FULL_BUILD_DIR}"
echo "LLama CLI: ${LLAMA_CLI}"
echo "Model: ${LLAMA_MODEL}"
echo "Build Type: ${LLAMA_CMAKE_BUILD_TYPE}"
echo "Platform: ${LLAMA_BUILD_SUFFIX}"
echo "Python Version: $(python3.13 --version 2>/dev/null || echo 'Python 3.13 not found')"
echo "Virtual Environment: .venv (managed by uv)"
echo "Package Manager: uv"
echo "=========================================="
