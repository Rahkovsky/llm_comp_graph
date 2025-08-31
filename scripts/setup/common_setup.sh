#!/usr/bin/env bash
# Common setup functions for Llama.cpp installation

set -euo pipefail

# Simple logging
log() { echo "► $1"; }
error() { echo "✗ Error: $1" >&2; exit 1; }

# Install Python 3.13
setup_python() {
    log "Setting up Python 3.13..."

    if ! command -v python3.13 >/dev/null 2>&1; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install python@3.13 >/dev/null 2>&1
        else
            sudo dnf install -y python3.13 python3.13-devel >/dev/null 2>&1 || true
        fi
    fi

    command -v python3.13 >/dev/null 2>&1 || error "Python 3.13 installation failed"
}

# Install uv package manager
setup_uv() {
    log "Installing uv package manager..."

    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        export PATH="$HOME/.local/bin:$PATH"
    fi

    command -v uv >/dev/null 2>&1 || error "uv installation failed"
}

# Setup Python environment
setup_venv() {
    log "Creating Python virtual environment..."
    uv venv --python python3.13 >/dev/null 2>&1
    source .venv/bin/activate
}

# Clone llama.cpp repository
clone_llama() {
    log "Cloning llama.cpp repository..."
    [ -d "${LLAMA_INSTALL_DIR}" ] || git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_INSTALL_DIR}" >/dev/null 2>&1
}

# Download model
download_model() {
    log "Downloading model..."
    mkdir -p "${LLAMA_MODELS_DIR}"
    local model_file="${LLAMA_MODELS_DIR}/$(basename "${LLAMA_MODEL}")"

    if [ ! -f "${model_file}" ]; then
        curl -L -o "${model_file}" \
            "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/$(basename "${LLAMA_MODEL}")" 2>/dev/null
    fi
}

# Test installation
test_install() {
    log "Testing installation..."

    # Test that binary exists and runs
    if ! "${LLAMA_CLI}" --version >/dev/null 2>&1; then
        error "llama-cli not working"
    fi

    # Test model file exists
    if [ ! -f "${LLAMA_MODEL}" ]; then
        error "Model file not found"
    fi

    echo "✓ Installation verified successfully"
}

# Show completion message
show_complete() {
    echo ""
    echo "✓ Setup complete!"
    echo "  CLI: ${LLAMA_CLI}"
    echo "  Model: ${LLAMA_MODEL}"
    echo "  Run: ./setup_llama.sh --help for usage"
}
