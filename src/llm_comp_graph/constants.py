#!/usr/bin/env python3
"""Constants shared across SEC filing download scripts."""

from llm_comp_graph.utils.env_config import get_sec_user_agent
import os


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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CLI = os.path.join(PROJECT_ROOT, LLAMA_CLI_REL)
LLAMA_MODEL = os.path.join(PROJECT_ROOT, LLAMA_MODEL_REL)
