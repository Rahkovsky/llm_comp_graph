# Llama.cpp Setup

Simple, professional local installation of llama.cpp with automated platform detection.

## New Script Structure

```
llm_comp_graph/
├── setup_llama.sh              # Main management script (run this!)
├── llama_env.sh                # Environment configuration
├── scripts/
│   └── setup/                  # Platform-specific setup scripts
│       ├── common_setup.sh     # Shared functionality
│       ├── mac_metal.sh        # macOS Metal setup
│       ├── linux_cpu.sh        # Linux CPU setup
│       └── linux_gpu.sh        # Linux GPU setup
├── llama/                      # Local llama.cpp installation
│   ├── build_metal/            # macOS Metal build
│   ├── build_gpu/              # Linux GPU build  
│   └── build_cpu/              # Linux CPU build
└── models/                     # Downloaded model files
    └── Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
```

## Quick Start

**Auto-detect and install:**
```bash
./setup_llama.sh --detect
```

**Or specify platform:**
```bash
./setup_llama.sh mac_metal     # macOS
./setup_llama.sh linux_cpu     # Linux CPU
./setup_llama.sh linux_gpu     # Linux GPU
```

## What Each Script Does

### **`setup_llama.sh` (Main Script)**
- **Platform Detection**: Automatically detects your OS and hardware
- **Argument Parsing**: Handles command-line options and platform selection
- **Prerequisites Check**: Ensures all required files are present
- **Script Execution**: Runs the appropriate platform-specific setup

### **`common_setup.sh` (Shared Functions)**
- **Python 3.13 Installation**: Sets up Python 3.13 as default
- **uv Package Manager**: Installs and configures uv for dependency management
- **Virtual Environment**: Creates `.venv` using uv
- **Repository Management**: Clones llama.cpp repository
- **Model Download**: Downloads the Llama model
- **Smoke Testing**: Runs basic functionality tests
- **Logging**: Provides colored, informative output

### **Platform-Specific Scripts**
- **`mac_metal.sh`**: macOS with Metal acceleration
- **`linux_cpu.sh`**: Linux CPU-only build
- **`linux_gpu.sh`**: Linux with NVIDIA CUDA support

## Environment Variables

The `llama_env.sh` file automatically sets up these environment variables:

### Base Directories
- `LLAMA_PROJECT_ROOT`: Current project directory
- `LLAMA_INSTALL_DIR`: Local llama.cpp installation directory
- `LLAMA_MODELS_DIR`: Models storage directory
- `LLAMA_BUILD_DIR`: Base build directory

### Executables
- `LLAMA_CLI`: Path to llama-cli executable
- `LLAMA_SERVER`: Path to llama-server executable

### Model
- `LLAMA_MODEL`: Path to the default model file

### Build Configuration
- `LLAMA_CMAKE_BUILD_TYPE`: Build type (Release/Debug)
- `LLAMA_CCACHE_ENABLED`: Whether ccache is enabled
- `LLAMA_FULL_BUILD_DIR`: Complete build directory path with platform suffix

### Platform Detection
The script automatically detects your platform and sets appropriate flags:
- **macOS**: Metal acceleration enabled
- **Linux with NVIDIA**: CUDA acceleration enabled  
- **Linux without NVIDIA**: CPU-only build

## Advanced Usage

### **Custom CUDA Architecture (Linux GPU)**
```bash
GGML_CUDA_ARCHITECTURES=86 ./setup_llama.sh linux-gpu
```

### **Verbose Output**
```bash
bash -x ./setup_llama.sh mac-metal
```

### **Check Current Setup**
```bash
source ./llama_env.sh
echo "LLama CLI: ${LLAMA_CLI}"
echo "Model: ${LLAMA_MODEL}"
```

## Benefits of the New Structure

1. **Maintainable**: Common code is factored out into shared functions
2. **Organized**: Clear separation of concerns and file organization
3. **User-Friendly**: Single command to run with auto-detection
4. **Professional**: Colored output, proper error handling, and logging
5. **Modular**: Easy to add new platforms or modify existing ones
6. **Consistent**: All platforms use the same setup process and environment

## Troubleshooting

### **Script Not Found**
```bash
# Make sure you're in the project root
pwd  # Should show: /path/to/llm_comp_graph
ls -la setup_llama.sh  # Should show the main script
```

### **Permission Denied**
```bash
chmod +x setup_llama.sh scripts/setup/*.sh
```

### **Environment Variables Not Set**
```bash
source ./llama_env.sh
```

### **Python Issues**
```bash
# Check Python version
python3.13 --version

# Check uv installation
uv --version
```

## Customization

### **Adding New Platforms**
1. Create a new script in `scripts/setup/`
2. Source `common_setup.sh` for shared functions
3. Add platform-specific build logic
4. Update `setup_llama.sh` validation

### **Modifying Environment**
Edit `llama_env.sh` to:
- Change model files
- Adjust build parameters
- Add custom flags
- Modify directory paths

### **Adding Dependencies**
Update `requirements.txt` and the scripts will automatically install them using `uv`.

## Migration from Old Scripts

If you were using the old scripts directly:

**Old way:**
```bash
./mac_metal.sh
./linux_cpu.sh
./linux_gpu.sh
```

**New way:**
```bash
./setup_llama.sh --detect
# or
./setup_llama.sh mac-metal
```

The new structure provides the same functionality with better organization, maintainability, and user experience.
