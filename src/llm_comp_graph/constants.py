#!/usr/bin/env python3
"""Constants shared across SEC filing download scripts."""

from llm_comp_graph.utils.env_config import get_sec_user_agent
import os
from pathlib import Path


# SEC API URLs
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_TXT_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}.txt"


DEFAULT_UA = get_sec_user_agent()

# Output directories
OUTDIR_10K = "data/input/10K"
OUTDIR_10Q = "data/input/10Q"

# Form types
FORM_TYPES_10K = ["10-K", "10-K/A"]
FORM_TYPES_10Q = ["10-Q"]

# HTTP settings
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 3
HTTP_RETRY_DELAY = 0.5

# File settings
DEFAULT_SLEEP = 0.25

# Llama.cpp paths
# Define relative paths first, then compute absolute once
LLAMA_CLI_REL = "llama/build_metal/bin/llama-cli"  # macOS Metal default
LLAMA_MODEL_REL = "models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"


def _find_project_root() -> str:
    """Locate project root by walking up until a marker file is found.

    Falls back to the repository root (three levels up from this file) if markers are not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
            return str(parent)
    # Fallback: repo root is typically three levels up: src/llm_comp_graph/constants.py → repo
    try:
        return str(here.parents[2])
    except Exception:
        return str(here.parent)


PROJECT_ROOT = _find_project_root()
LLAMA_CLI = os.path.join(PROJECT_ROOT, LLAMA_CLI_REL)
LLAMA_MODEL = os.path.join(PROJECT_ROOT, LLAMA_MODEL_REL)

# Indexing/Search defaults
INDEX_DIR = "data/llama_index"
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_COLLECTION_NAME = "10k_documents"
DEFAULT_LLM_CTX = 16384  # maximum size of the prompt + generation
