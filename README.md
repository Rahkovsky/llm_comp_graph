# LLM Computational Graph

This repository locally runs LLAMA models and provides tools for financial document analysis using SEC 10-K filings.

## Installation

### Quick Setup
```bash
# Auto-detect your platform and install
./setup_llama.sh --detect

# Or specify platform manually
./setup_llama.sh mac_metal     # macOS with Metal
./setup_llama.sh linux_cpu     # Linux CPU-only
./setup_llama.sh linux_gpu     # Linux with NVIDIA GPU
```

### Manual Setup (uv)
```bash
# 1. Install Python 3.13 and uv
# 2. Create and activate a virtual environment
uv venv --python 3.13
source .venv/bin/activate

# 3. Install dependencies and the package in editable mode
uv pip install -r requirements.txt
uv pip install -e .

# 4. Configure SEC User-Agent (required by SEC)
cat > .env << 'EOF'
SEC_USER_NAME="Your Name"
SEC_USER_EMAIL="your.email@example.com"
EOF

# 5. Build llama.cpp (see LLAMA_SETUP.md for details)
```

## Usage Examples

### Download SEC 10-K Filings

Download 10-K filings for specific companies:

```bash
# Activate virtual environment
source .venv/bin/activate

# Download 10-K for Apple (AAPL)
python scripts/download/download_10K.py --tickers AAPL --year 2024 --verbose

# Download 10-K for multiple companies


# Download with text extraction (enabled by default)
python scripts/download/download_10K.py --tickers MSFT --year 2024 --verbose
```

### Generate Questions from Documents

```bash
# Generate questions from downloaded filings
python scripts/generate_questions.py --input ./sec_data --output ./questions
```

### Build a LlamaIndex (Chroma) and Search It

```bash
# Activate virtual environment
source .venv/bin/activate

# 1) Build the index with optimized settings for 10-K documents
python scripts/create_llama_index.py \
  --input-dir data/input/10K \
  --output-dir data/llama_index \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --chunk-size 900 \
  --chunk-overlap 160

# HNSW tuning for optimal recall/performance balance:
#   --hnsw-ef-construction 200    # Higher = better recall, slower build
#   --hnsw-ef-search 150          # Higher = better recall, slower queries
#   --hnsw-max-neighbors 32       # Higher = denser graph, better recall

# Other options:
#   --metadata-only               # create only metadata_index.json (no vector index)
#   --max-files 500               # index a subset for quick tests
#   --rebuild                     # wipe existing chroma_db and index dirs first
#   --collection-name NAME        # change Chroma collection name (default: 10k_documents)
#   --resolve-company-names       # enrich metadata with company_name via SEC (slower)

# 2) Query the index with advanced search options
python scripts/search_llama_index.py \
  --index-dir data/llama_index \
  --query "revenue recognition policy" \
  --top-k 8 \
  --summarize --summary-sentences 7

# Advanced search options:
#   --llm-summary                 # Use local LLAMA for executive summary
#   --llm-n-ctx 16384            # Context window for LLAMA (tokens)
#   --rewrite-queries             # Use local LLM to rewrite queries for better recall
#   --n-rewritten-queries 3      # Number of rewritten queries to generate
#   --query-rewrite-temp 0.3     # Temperature for query rewriting
#   --filter-ticker AAPL MSFT    # Filter by specific tickers
#   --filter-year 2024 2023      # Filter by filing years
#   --mmr                         # Enable MMR diversity selection (default: on)
#   --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2  # Optional reranking

# Embedding model options (must match index creation):
#   --embedding-model BAAI/bge-base-en-v1.5   # default; use same at query time
#                         intfloat/e5-large-v2
#                         BAAI/bge-m3         # 1024 ctx; chunking auto-caps to model max
```

## Project Structure

```
llm_comp_graph/
├── setup_llama.sh              # Main setup script
├── llama_env.sh                # Environment configuration
├── scripts/
│   ├── setup/                  # Platform-specific setup scripts
│   ├── download/               # Download scripts
│   │   └── download_10K.py     # SEC 10-K downloader
│   ├── generate_questions.py   # Question generation
│   ├── create_llama_index.py   # Build vector index
│   └── search_llama_index.py   # Query the index
├── llama/                      # Local llama.cpp installation
├── models/                     # Downloaded models
└── .venv/                      # Python virtual environment
```

## Development Plan

- [x] **Local LLAMA Setup** - Complete
- [x] **10-K Downloader** - Available
- [x] **Vector Database Creation** - Complete with HNSW tuning
- [x] **RAG System for Business Questions** - Available with advanced search
- [ ] **LLM Fine-tuning on 10-K Data** - Planned

## Documentation

- **Setup Guide**: [LLAMA_SETUP.md](LLAMA_SETUP.md) - Detailed installation instructions
- **Scripts Guide**: [scripts/README.md](scripts/README.md) - Script documentation

## Requirements

- Python 3.13+
- 8GB+ RAM (16GB+ recommended)
- macOS with Metal or Linux with CUDA/CPU support
- Internet connection for model and document downloads
