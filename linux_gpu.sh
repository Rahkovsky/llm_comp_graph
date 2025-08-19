#!/usr/bin/env bash
set -euo pipefail

# Amazon Linux 2023 (or similar) CUDA/Metal not used here; CUDA-only GPU build for llama.cpp
# REQUIREMENTS: NVIDIA driver + CUDA toolkit installed; nvidia-smi works.

# Optional: set a specific CUDA arch to speed up compile (e.g., 86=A10G, 89=L4, 80=A100)
: "${GGML_CUDA_ARCHITECTURES:=}"

# 0) Python venv (optional)
python3.11 -m venv venv || python3 -m venv venv
source venv/bin/activate || true
pip install -U pip uv || true
uv pip install -r requirements.txt || true

# 1) Sanity checks
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver + CUDA before running." >&2
  exit 1
fi

# 2) System deps
sudo dnf upgrade -y
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git cmake ninja-build ccache || true
if ! command -v cmake >/dev/null 2>&1; then
  sudo dnf install -y cmake3
  sudo alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 1 --force
fi
cmake --version

# 3) Clone
cd ~
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp.git

# 4) GPU build (CUDA)
CMAKE_ARGS=(-S ~/llama.cpp -B ~/llama.cpp/build_gpu -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=ON)
if [ -n "$GGML_CUDA_ARCHITECTURES" ]; then
  CMAKE_ARGS+=(-DGGML_CUDA_ARCHITECTURES="$GGML_CUDA_ARCHITECTURES")
fi
cmake "${CMAKE_ARGS[@]}"
ninja -C ~/llama.cpp/build_gpu

# 5) Model
mkdir -p ~/models
cd ~/models
[ -f Meta-Llama-3.1-8B-Instruct-Q6_K.gguf ] || \
  curl -L -o Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

# 6) Smoke test (full offload)
~/llama.cpp/build_gpu/bin/llama-cli -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -n 16 -ngl 999 -p "Hello"

# 7) Example prompt (GPU — use -ngl)
~/llama.cpp/build_gpu/bin/llama-cli \
  -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  -n 128 \
  --color \
  --repeat_penalty 1.1 \
  -c 4096 \
  -ngl 999 \
  -p "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Today is May 17, 2025.<|eot_id|><|start_header_id|>user<|end_header_id|>
Explain the concept fear of missing out in a few sentences."


export LLAMA_CLI="$HOME/llama.cpp/build_gpu/bin/llama-cli"
export LLAMA_MODEL="$HOME/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"

