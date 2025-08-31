#!/usr/bin/env bash
# Llama.cpp setup script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="${SCRIPT_DIR}/scripts/setup"

show_usage() {
    cat << EOF
Llama.cpp Setup Script

Usage: $0 [OPTIONS] <platform>

Platforms:
  mac_metal    - macOS with Metal acceleration
  linux_cpu    - Linux CPU-only
  linux_gpu    - Linux with NVIDIA CUDA

Options:
  -h, --help   - Show this help
  --detect     - Auto-detect platform

Examples:
  $0 mac_metal
  $0 --detect
EOF
}

detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "mac_metal"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "linux_gpu"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux_cpu"
    else
        echo "unknown"
    fi
}

run_setup() {
    local platform="$1"
    local setup_script="${SETUP_DIR}/${platform}.sh"
    
    [[ -f "$setup_script" ]] || { echo "Error: Setup script not found: $setup_script"; exit 1; }
    [[ -x "$setup_script" ]] || chmod +x "$setup_script"
    
    echo "Running $platform setup..."
    "$setup_script"
}

main() {
    local platform=""
    
    case "${1:-}" in
        -h|--help) show_usage; exit 0 ;;
        --detect) platform=$(detect_platform) ;;
        mac_metal|linux_cpu|linux_gpu) platform="$1" ;;
        "") echo "Error: Platform required"; show_usage; exit 1 ;;
        *) echo "Error: Invalid platform: $1"; show_usage; exit 1 ;;
    esac
    
    [[ "$platform" != "unknown" ]] || { echo "Error: Could not detect platform"; exit 1; }
    [[ -f "llama_env.sh" ]] || { echo "Error: Run from project root"; exit 1; }
    
    run_setup "$platform"
}

# Run main function with all arguments
main "$@"
