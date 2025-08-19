#!/usr/bin/env bash
set -euo pipefail

# macOS (Apple Silicon) Metal build for llama.cpp

# 0) Homebrew deps
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew not found. Install from https://brew.sh/ and re-run." >&2
  exit 1
fi
brew update
brew install cmake ninja ccache git

# 1) Python venv (optional)
python3 -m venv venv || true
source venv/bin/activate || true
pip install -U pip uv || true
uv pip install -r requirements.txt || true

# 2) Clone
cd ~
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp.git

# 3) Metal build
cmake -S ~/llama.cpp -B ~/llama.cpp/build_metal -G Ninja -DGGML_METAL=ON -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release
ninja -C ~/llama.cpp/build_metal

# 4) Model
mkdir -p ~/models
cd ~/models
[ -f Meta-Llama-3.1-8B-Instruct-Q6_K.gguf ] || \
  curl -L -o Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

# 5) Smoke test (Metal offload with -ngl)
~/llama.cpp/build_metal/bin/llama-cli -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -n 16 -ngl 999 -p "Hello"

# 6) Example prompt (Metal — use -ngl)
~/llama.cpp/build_metal/bin/llama-cli \
  -m ~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf \
  -n 128 \
  --color \
  --repeat_penalty 1.1 \
  -c 4096 \
  -ngl 999 \
  -p "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Today is May 17, 2025.<|eot_id|><|start_header_id|>user<|end_header_id|>
Explain the concept of \"emergent abilities\" in large language models in a few sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
