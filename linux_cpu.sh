#!/usr/bin/env bash
set -euo pipefail

# Amazon Linux 2023 (or similar dnf-based) CPU-only setup for llama.cpp

# 0) Python venv (optional)
python3.11 -m venv venv || python3 -m venv venv
source venv/bin/activate || true
pip install -U pip uv || true
uv pip install -r requirements.txt || true

# 1) System deps
sudo dnf upgrade -y
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git cmake || true
if ! command -v cmake >/dev/null 2>&1; then
  sudo dnf install -y cmake3
  sudo alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 1 --force
fi
cmake --version

# 2) Clone
cd ~
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp.git

# 3) CPU build
cmake -S ~/llama.cpp -B ~/llama.cpp/build_cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build_cpu --config Release -j"$(nproc)"

# 4) Model
mkdir -p ~/models
cd ~/models
[ -f Meta-Llama-3.1-8B-Instruct-Q6_K.gguf ] || \
  curl -L -o Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

# 5) Smoke test
~/llama.cpp/build_cpu/bin/llama-cli -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -n 16 -p "Hello"

# 6) Example prompt (CPU — no -ngl)
~/llama.cpp/build_cpu/bin/llama-cli \
  -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  -n 128 \
  --color \
  --repeat_penalty 1.1 \
  -c 4096 \
  -p "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Today is May 17, 2025.<|eot_id|><|start_header_id|>user<|end_header_id|>
Explain the concept of \"emergent abilities\" in large language models in a few sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
