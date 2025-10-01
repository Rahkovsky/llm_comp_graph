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
│   └── download_10K.py        # SEC 10-K document downloader (specific companies)
├── generate_questions.py       # Question generation script
├── create_llama_index.py      # Create LlamaIndex for downloaded documents
└── search_llama_index.py      # Query and search the vector index
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

## Key Features & Improvements

### Production-Ready Indexing (`create_llama_index.py`)
- **Optimized for 10-K Documents**: Enhanced chunking strategy with paragraph preservation
- **HNSW Tuning**: Configurable parameters for optimal recall/performance balance
- **Robust Error Handling**: Network resilience with retry logic and exponential backoff
- **Embedding Alignment**: Explicit normalization for cosine similarity search
- **Comprehensive CLI**: Full control over all indexing parameters

### Advanced Search (`search_llama_index.py`)
- **MMR Diversity Selection**: Reduces redundancy and improves coverage
- **Metadata Filtering**: Filter by ticker, year, and other metadata fields
- **Cross-Encoder Reranking**: Optional reranking for improved precision
- **Local LLM Summarization**: Context-aware executive summaries
- **Performance Monitoring**: Detailed timing and retrieval statistics

### Network Resilience
- **Retry Logic**: Automatic retry with exponential backoff for API calls
- **Rate Limit Handling**: Respects SEC API limits to prevent IP blocking
- **Graceful Degradation**: Continues processing even if some documents fail

## Maintenance

- Common functionality should be added to `common_setup.sh`
- Platform-specific logic belongs in individual platform scripts
- The main script (`setup_llama.sh`) should remain focused on orchestration


### Download data

# Download 10-K for Apple (AAPL)
python scripts/download/download_10K.py --tickers AAPL --year 2024

# Download 10-K for multiple companies
python scripts/download/download_10K.py --tickers AAPL MSFT GOOGL --year 2024 --output-dir ./data/input/10K

# Download from a tickers file
python scripts/download/download_10K.py --tickers-file tickers.txt --year 2024

# Download 10-K filings from ALL public companies in a specific year
python scripts/download/download_10K.py --all-companies --year 2024

### Create LlamaIndex

# Create LlamaIndex for all downloaded documents (optimized for 10-K)
python scripts/create_llama_index.py

# Create index for specific directory
python scripts/create_llama_index.py --input-dir data/input/10K/ALL_COMPANIES/2024

# Custom output directory
python scripts/create_llama_index.py --output-dir data/llama_index_2024

# Custom chunk size (optimized for financial documents)
python scripts/create_llama_index.py --chunk-size 900 --chunk-overlap 160

# HNSW tuning for optimal recall/performance balance
python scripts/create_llama_index.py \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 150 \
  --hnsw-max-neighbors 32

# Include only specific file types
python scripts/create_llama_index.py --file-pattern "*.plain.txt"

# Rebuild existing index (clears old data)
python scripts/create_llama_index.py --rebuild

# Metadata-only mode (no vector index)
python scripts/create_llama_index.py --metadata-only

### Search LLAMA Index

# Simple query (required: --query)
python scripts/search_llama_index.py --index-dir data/llama_index --query "revenue recognition"

# Top-K and extractive summary
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --collection-name 10k_documents \
  --query "AI will take over the world" \
  --top-k 8 --max-chars 800 --summarize

# Local LLAMA executive summary with context control
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "What the risk and opprotunities from social media companies in a new attention economy? How to reach consumers?" \
  --top-k 8 --max-chars 800 \
  --llm-summary --llm-n-ctx 16384

# Query rewriting for improved recall
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "What are the main risks?" \
  --rewrite-queries --n-rewritten-queries 3 \
  --top-k 8 --llm-summary

# Advanced search with metadata filters
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "supply chain disruptions" \
  --filter-ticker AAPL MSFT GOOGL \
  --filter-year 2024 2023 \
  --top-k 10

# MMR diversity selection (default: enabled)
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "risk factors" \
  --mmr --mmr-lambda 0.5 \
  --top-k 8

# Optional cross-encoder reranking for top results
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "revenue recognition policy" \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-top-n 100 \
  --top-k 5

# Specify embedding backend/model (must match index creation)
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --embedding-backend local-sbert \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --query "supply chain disruptions"

### Generate Questions

```
python scripts/generate_questions.py \
  --glob 'data/input/10K/MSFT/*/*.plain.txt' \
  --out out/questions_10k_msft.txt \
  --per-chunk 6
  ```
