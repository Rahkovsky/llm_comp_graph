# Scripts Directory

This directory contains all the automation and utility scripts for the Llama.cpp project.

## Directory Structure

```
scripts/
├── README.md                   # This file
├── setup/                      # Setup and installation scripts
│   ├── common_setup.sh        # Shared setup functions
│   ├── mac_metal.sh           # macOS Metal setup
│   ├── linux_cpu.sh           # Linux CPU setup
│   └── linux_gpu.sh           # Linux GPU setup
├── download/                   # SEC filing download scripts
│   ├── __init__.py            # Package initialization
│   ├── download_10K.py        # SEC 10-K document downloader (specific companies)
│   ├── download_10Q.py        # SEC 10-Q document downloader
│   ├── download_all_companies_10k.py  # Download 10-K from all public companies in a year
│   └── download_2024.py       # Simple CLI for downloading all 2024 10-K data
├── generate_questions.py       # Question generation script
└── create_llama_index.py      # Create LlamaIndex for downloaded documents
```

## Setup Scripts

The setup scripts are organized to provide a clean, maintainable structure:

- **`common_setup.sh`**: Contains all shared functionality used by platform-specific scripts
- **Platform scripts**: Handle OS-specific build configurations and dependencies
- **Main script**: `setup_llama.sh` in the project root manages the entire setup process

## Usage

**From the project root:**
```bash
# Auto-detect platform
./setup_llama.sh --detect

# Specify platform
./setup_llama.sh mac-metal
./setup_llama.sh linux-cpu
./setup_llama.sh linux-gpu
```

**Direct execution (advanced users):**
```bash
# Run platform-specific setup directly
./scripts/setup/mac_metal.sh
./scripts/setup/linux_cpu.sh
./scripts/setup/linux_gpu.sh
```

## Adding New Platforms

To add support for a new platform:

1. Create a new script in `scripts/setup/`
2. Source `common_setup.sh` for shared functions
3. Implement platform-specific build logic
4. Update the main `setup_llama.sh` script validation

## Script Dependencies

All setup scripts depend on:
- `llama_env.sh` (environment configuration)
- `common_setup.sh` (shared functions)
- `requirements.txt` (Python dependencies)

## Maintenance

- Common functionality should be added to `common_setup.sh`
- Platform-specific logic belongs in individual platform scripts
- The main script (`setup_llama.sh`) should remain focused on orchestration


### Download data

# Download 10-K for Apple (AAPL)
python scripts/download/download_10K.py --ticker AAPL --years 2020-2024

# Download 10-K for multiple companies
python scripts/download/download_10K.py --ticker AAPL,MSFT,GOOGL --years 2022-2025 --output ./data

# Download with custom output directory
python scripts/download/download_10K.py --ticker TSLA --years 2023 --output ./data

# Download 10-K filings from ALL public companies in a specific year
python scripts/download/download_all_companies_10k.py --year 2024

# Download with limits and resume capability
python scripts/download/download_all_companies_10k.py --year 2024 --max-companies 100 --start-from AAPL

# Skip companies that already have filings
python scripts/download/download_all_companies_10k.py --year 2024 --skip-existing

# 🚀 SIMPLE CLI for 2024 data (recommended)
c

# Test mode (100 companies)
python scripts/download/download_2024.py --test

# Resume from specific company
python scripts/download/download_2024.py --resume AAPL

# Skip existing files
python scripts/download/download_2024.py --skip-existing

# Custom progress file
python scripts/download/download_2024.py --progress-file my_2024_progress.json

### Create LlamaIndex

# Create LlamaIndex for all downloaded documents
python scripts/create_llama_index.py

# Create index for specific directory
python scripts/create_llama_index.py --input-dir data/input/10K/ALL_COMPANIES/2024

# Custom output directory
python scripts/create_llama_index.py --output-dir llama_index_2024

# Custom chunk size
python scripts/create_llama_index.py --chunk-size 2048 --chunk-overlap 100

# Include only specific file types
python scripts/create_llama_index.py --file-pattern "*.plain.txt"

### Search LLAMA Index

# Show index statistics
python scripts/search_llama_index.py --stats

# List all companies
python scripts/search_llama_index.py --list-companies

# Search by ticker
python scripts/search_llama_index.py --ticker AAPL

# Search by date
python scripts/search_llama_index.py --date 2024-10-28

# Search by file size range
python scripts/search_llama_index.py --size-range 5 10

# Search by word count range
python scripts/search_llama_index.py --word-range 10000 50000

# Show detailed results
python scripts/search_llama_index.py --ticker AAPL --details

### Generate Questions

```
python scripts/generate_questions.py \
  --glob 'data/input/10K/MSFT/*/*.plain.txt' \
  --out out/questions_10k_msft.txt \
  --per-chunk 6
  ```
