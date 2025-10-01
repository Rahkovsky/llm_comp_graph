#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Create a LlamaIndex for downloaded 10-K documents with content ingestion.

This unified script merges the simple and advanced versions, adds robust CLI options,
metadata extraction, idempotent Chroma handling, and multiple embedding backends.
"""

import argparse
import json
import os
import shutil
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llm_comp_graph.utils.env_config import get_sec_user_agent
from llm_comp_graph.utils.logging_config import setup_logging
from transformers import AutoConfig, AutoTokenizer
from llm_comp_graph.constants import (
    OUTDIR_10K,
    INDEX_DIR as DEFAULT_INDEX_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_COLLECTION_NAME,
)


"""This build assumes modern LlamaIndex packages are installed (see requirements)."""

logger = setup_logging(module_name=__name__)


company_name_cache: Dict[str, str] = {}

# Defaults for chunking heuristics (optimized for 10-K documents)
DEFAULT_CHUNK_SIZE = 900
DEFAULT_MIN_CHUNK = 64
DEFAULT_MODEL_MARGIN = 32
DEFAULT_MIN_OVERLAP = 32
DEFAULT_OVERLAP_RATIO = 0.18

# HNSW tuning parameters for optimal recall/performance balance
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 150
DEFAULT_HNSW_MAX_NEIGHBORS = 32


def _create_robust_session() -> requests.Session:
    """Create a requests session with retry strategy for network resilience."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def normalize_cik(cik: str) -> str:
    digits = re.sub(r"\D", "", cik or "")
    if not digits:
        return ""
    if len(digits) > 10:
        return digits[:10]
    return digits.zfill(10)


def fetch_company_name_from_sec(cik: str) -> Optional[str]:
    norm = normalize_cik(cik)
    if not norm:
        return None
    url = f"https://data.sec.gov/submissions/CIK{norm}.json"
    try:
        try:
            ua = get_sec_user_agent()
        except Exception:
            ua = "llm-comp-graph/1.0 (contact@example.com)"
        headers = {"User-Agent": ua}
        session = _create_robust_session()
        resp = session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()  # Raises HTTPError for 4xx/5xx
        data = resp.json()
        name = data.get("name") or data.get("entityType")
        return name.strip() if isinstance(name, str) and name.strip() else None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch company name for CIK {cik}: {e}")
        pass
    return None


def resolve_company_name(cik: str) -> Optional[str]:
    key = normalize_cik(cik)
    if not key:
        return None
    if key in company_name_cache:
        return company_name_cache[key]
    name = fetch_company_name_from_sec(key)
    if name:
        company_name_cache[key] = name
        return name
    return None


def get_file_metadata(
    file_path: Path, *, resolve_names: bool = False
) -> Dict[str, str]:
    """Extract metadata from 10-K file path."""
    filename = file_path.stem
    parts = filename.split("_")

    ticker = parts[0].upper() if parts else "UNKNOWN"
    cik = "UNKNOWN"
    date_str = "UNKNOWN"
    accession = "UNKNOWN"

    # Support modern SEC accession pattern (e.g., AMZN_0001018724-25-000004)
    if len(parts) >= 2:
        rest = parts[1]
        m = re.match(r"(?P<cik>\d{10})-(?P<yy>\d{2})-(?P<seq>\d{6})$", rest)
        if m:
            cik = m.group("cik")
            accession = f"{m.group('cik')}-{m.group('yy')}-{m.group('seq')}"
            date_str = f"20{m.group('yy')}"  # year-only if full date not present
        else:
            cik = parts[1]

    # Legacy pattern: TICKER_CIK_DATE_ACCESSION...
    if len(parts) >= 3:
        date_str = parts[2]
    if len(parts) >= 4:
        accession = "_".join(parts[3:])

    # Resolve company name via SEC submissions (cached) if opted-in
    company_name = (
        resolve_company_name(cik)
        if resolve_names and cik not in ("", "UNKNOWN")
        else None
    )

    try:
        relative_path = str(file_path.relative_to(Path.cwd()))
    except Exception:
        # Fallback if file is outside CWD
        relative_path = str(file_path)

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "filing_date": date_str,
        "accession": accession,
        "file_path": str(file_path),
        "relative_path": relative_path,
        "company_name": company_name or "UNKNOWN",
    }


"""LLAMA model checks removed; backend not supported in this build."""


def create_documents_from_files(
    base_dir: str,
    file_pattern: str = "*.txt",
    max_files: Optional[int] = None,
    *,
    resolve_names: bool = False,
) -> List[Any]:
    """Create LlamaIndex Document objects from files (single-pass), with metadata."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.error(f"Directory not found: {base_dir}")
        return []

    logger.info(f"Reading documents from: {base_dir}")
    logger.info(f"Looking for files matching: {file_pattern}")

    documents: List[Any] = []
    processed_files = 0

    for file_path in base_path.rglob(file_pattern):
        if not file_path.is_file():
            continue
        if max_files is not None and len(documents) >= max_files:
            break

        processed_files += 1
        if processed_files % 50 == 0:
            logger.info(f"Processed {processed_files} files...")

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            metadata = get_file_metadata(file_path, resolve_names=resolve_names)
            try:
                doc = Document(text=content, metadata=metadata)  # type: ignore[call-arg]
            except Exception:
                doc = {"text": content, "metadata": metadata}
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    logger.info(f"Successfully created {len(documents)} Document objects")
    return documents


def _instructions(model_name: str):
    n = model_name.lower()
    if "bge-m3" in n:
        return None, None  # M3: no instructions
    if "bge" in n:
        return (
            "Represent this sentence for searching relevant passages: ",
            None,
        )  # BGE v1.5: query-only
    if "e5" in n:
        return "query: ", "passage: "  # E5: query/passages prefixes
    return None, None


def _embedder_max_len(model_name: str) -> int:
    # Prefer tokenizer, then config; fall back to 512 for BERT-ish models.
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model_max_length = getattr(tok, "model_max_length", None)
        if isinstance(model_max_length, int) and 16 <= model_max_length < 100_000:
            return model_max_length
    except Exception:
        pass
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        max_position_embeddings = getattr(cfg, "max_position_embeddings", None)
        if isinstance(max_position_embeddings, int) and max_position_embeddings >= 16:
            return max_position_embeddings
    except Exception:
        pass
    return 512


def create_vector_index(
    documents: List[Any],
    output_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    collection_name: str = "10k_documents",
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    hnsw_max_neighbors: int = DEFAULT_HNSW_MAX_NEIGHBORS,
):
    assert documents, "No documents to index"
    os.makedirs(output_dir, exist_ok=True)

    # Embeddings (with correct per-model instructions and explicit normalization)
    q_instr, t_instr = _instructions(embedding_model_name)
    cache_folder = str(Path(output_dir) / "models" / "embeddings")
    embed = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        cache_folder=cache_folder,
        query_instruction=q_instr,
        text_instruction=t_instr,
        normalize=True,  # Explicit normalization for cosine similarity alignment
    )

    # Cap chunking to embedder capacity (avoid truncation at embed time)
    model_max = _embedder_max_len(embedding_model_name)
    eff_chunk = max(
        DEFAULT_MIN_CHUNK, min(chunk_size, model_max - DEFAULT_MODEL_MARGIN)
    )
    eff_overlap = max(
        DEFAULT_MIN_OVERLAP,
        min(int(DEFAULT_OVERLAP_RATIO * eff_chunk), eff_chunk // 2),
    )

    Settings.embed_model = embed
    # Optimized chunking for 10-K documents: smaller chunks with more overlap
    Settings.node_parser = SentenceSplitter(
        chunk_size=eff_chunk,
        chunk_overlap=eff_overlap,
        paragraph_separator="\n\n",  # Preserve paragraph structure
        secondary_chunking_regex="[.!?]\\s+",  # Split on sentence boundaries
    )

    # Chroma (with explicit cosine space and HNSW tuning)
    db = chromadb.PersistentClient(path=os.path.join(output_dir, "chroma_db"))
    col = db.get_or_create_collection(
        collection_name,
        metadata={
            "hnsw:space": "cosine",  # Align with normalized embeddings
        },
    )
    vs = ChromaVectorStore(chroma_collection=col)
    sc = StorageContext.from_defaults(vector_store=vs)

    # Build & persist
    index = VectorStoreIndex.from_documents(
        documents, storage_context=sc, show_progress=True
    )
    index.storage_context.persist(persist_dir=os.path.join(output_dir, "index"))

    # Sidecar with effective settings (what was actually used)
    with open(
        os.path.join(output_dir, "embedding_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "backend": "huggingface",
                "model_name": embedding_model_name,
                "normalize": True,
                "metric": "cosine",
                "chunk_size": eff_chunk,
                "chunk_overlap": eff_overlap,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search": hnsw_ef_search,
                "hnsw_max_neighbors": hnsw_max_neighbors,
                "collection_name": collection_name,
                "docs_count": len(documents),
                "created_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    return index


def create_metadata_index(documents: List[Any]) -> Dict[str, Any]:
    """Create a metadata index for quick lookups."""
    logger.info("Creating metadata index...")

    companies: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        metadata: Dict[str, Any]
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})
        ticker = str(metadata.get("ticker", "UNKNOWN"))
        if ticker not in companies:
            companies[ticker] = {
                "ticker": ticker,
                "cik": metadata.get("cik", "UNKNOWN"),
                "total_filings": 0,
                "filing_dates": [],
                "accessions": [],
                "file_paths": [],
            }

        company = companies[ticker]
        company["total_filings"] += 1
        company["filing_dates"].append(metadata.get("filing_date", "UNKNOWN"))
        company["accessions"].append(metadata.get("accession", "UNKNOWN"))
        company["file_paths"].append(metadata.get("file_path", ""))

    sorted_companies = sorted(
        companies.values(), key=lambda x: x["total_filings"], reverse=True
    )

    return {
        "companies": sorted_companies,
        "total_companies": len(sorted_companies),
        "total_documents": len(documents),
        "created_at": datetime.now().isoformat(),
    }


def save_metadata_index(metadata_index: Dict[str, Any], output_file: str) -> None:
    """Save metadata index to JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata_index, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata index saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving metadata index: {e}")


def stream_and_index(
    input_dir: str,
    file_pattern: str,
    output_dir: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    collection_name: str,
    resolve_names: bool,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    hnsw_max_neighbors: int = DEFAULT_HNSW_MAX_NEIGHBORS,
    max_files: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Stream files → chunk → embed → insert, to avoid loading the full corpus in RAM.

    Returns a lightweight list of metadata dicts for building the metadata index.
    """
    logger.info("Streaming ingestion started")

    # Setup embedder and splitter
    q_instr, t_instr = _instructions(embedding_model_name)
    cache_folder = str(Path(output_dir) / "models" / "embeddings")
    embed = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        cache_folder=cache_folder,
        query_instruction=q_instr,
        text_instruction=t_instr,
        normalize=True,  # Explicit normalization for cosine similarity alignment
    )

    model_max = _embedder_max_len(embedding_model_name)
    eff_chunk = max(
        DEFAULT_MIN_CHUNK, min(chunk_size, model_max - DEFAULT_MODEL_MARGIN)
    )
    # Use user-provided chunk_overlap, but ensure it's within reasonable bounds
    eff_overlap = max(
        DEFAULT_MIN_OVERLAP,
        min(chunk_overlap, eff_chunk // 2),  # Don't exceed half of chunk size
    )
    Settings.embed_model = embed
    # Optimized chunking for 10-K documents
    splitter = SentenceSplitter(
        chunk_size=eff_chunk,
        chunk_overlap=eff_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[.!?]\\s+",
    )
    Settings.node_parser = splitter
    logger.info(
        f"Effective chunking -> chunk_size={eff_chunk}, chunk_overlap={eff_overlap} (model_max={model_max})"
    )

    # Vector store / index
    db = chromadb.PersistentClient(path=os.path.join(output_dir, "chroma_db"))
    col = db.get_or_create_collection(
        collection_name,
        metadata={
            "hnsw:space": "cosine",
        },
        configuration={
            "hnsw:ef_construction": hnsw_ef_construction,
            "hnsw:ef_search": hnsw_ef_search,
            "hnsw:max_neighbors": hnsw_max_neighbors,
        },
    )
    vs = ChromaVectorStore(chroma_collection=col)
    sc = StorageContext.from_defaults(vector_store=vs)
    # Create a new empty index that we can insert nodes into
    index = VectorStoreIndex(nodes=[], storage_context=sc)

    base_path = Path(input_dir)
    if not base_path.exists():
        logger.error(f"Directory not found: {input_dir}")
        return []

    meta_docs: List[Dict[str, Any]] = []
    processed = 0
    for file_path in base_path.rglob(file_pattern):
        if not file_path.is_file():
            continue
        if max_files is not None and processed >= max_files:
            break
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} files (streaming)...")

        try:
            logger.info(f"Processing file: {file_path}")
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                logger.info(f"Skipping empty file: {file_path}")
                continue
            logger.info(f"Read {len(content)} characters from {file_path}")

            metadata = get_file_metadata(file_path, resolve_names=resolve_names)
            meta_docs.append({"metadata": metadata})
            logger.info(f"Created metadata: {metadata}")

            doc = Document(text=content, metadata=metadata)  # type: ignore[call-arg]
            logger.info("Created document, splitting into nodes...")
            nodes = splitter.get_nodes_from_documents([doc])
            logger.info(f"Created {len(nodes)} nodes, inserting into index...")
            index.insert_nodes(nodes)
            logger.info(f"Successfully inserted {len(nodes)} nodes for {file_path}")
        except Exception as e:
            logger.warning(f"Error streaming {file_path}: {e}")
            import traceback

            logger.warning(f"Traceback: {traceback.format_exc()}")

    index.storage_context.persist(persist_dir=os.path.join(output_dir, "index"))
    logger.info("Streaming ingestion finished and persisted")

    return meta_docs


def main():
    parser = argparse.ArgumentParser(
        description="Create a LlamaIndex for 10-K documents with content ingestion (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create index for all downloaded documents
  python scripts/create_llama_index.py

  # Create index for specific directory
  python scripts/create_llama_index.py --input-dir data/input/10K

  # Custom output directory
  python scripts/create_llama_index.py --output-dir llama_index_2024

  # Custom chunk size
  python scripts/create_llama_index.py --chunk-size 2048 --chunk-overlap 100

  # Include only specific file types
  python scripts/create_llama_index.py --file-pattern "*.plain.txt"

  # Use local sentence-transformers embeddings
  python scripts/create_llama_index.py --embedding-backend local-sbert --embedding-model sentence-transformers/all-MiniLM-L6-v2

  # Metadata-only mode (no vector index)
  python scripts/create_llama_index.py --metadata-only
        """,
    )

    parser.add_argument(
        "--input-dir",
        default=OUTDIR_10K,
        help="Input directory containing 10-K documents (auto-detects if empty)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_INDEX_DIR,
        help="Output directory for the index",
    )
    parser.add_argument(
        "--file-pattern", default="*.txt", help="File pattern to include"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=160,
        help="Chunk overlap for text splitting",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=DEFAULT_HNSW_EF_CONSTRUCTION,
        help="HNSW ef_construction parameter (higher = better recall, slower build)",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=DEFAULT_HNSW_EF_SEARCH,
        help="HNSW ef_search parameter (higher = better recall, slower queries)",
    )
    parser.add_argument(
        "--hnsw-max-neighbors",
        type=int,
        default=DEFAULT_HNSW_MAX_NEIGHBORS,
        help="HNSW max_neighbors parameter (higher = denser graph, better recall)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit number of files to ingest"
    )
    parser.add_argument(
        "--resolve-company-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch company_name from SEC by CIK during ingestion (default: enabled)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Create only metadata index (skip vector index)",
    )
    # Backend selection removed; we assume a local HuggingFace embed model
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model id/path (default: {DEFAULT_EMBED_MODEL}; also supports intfloat/e5-large-v2)",
    )
    # LLAMA-specific flags removed in this build (local-sbert only)
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector DB and index (clears existing data)",
    )

    args = parser.parse_args()

    # All required packages are assumed present; no runtime check.
    logger.info("Creating LlamaIndex for 10-K Documents")
    logger.info(f"Input Directory: {args.input_dir}")
    logger.info(f"File Pattern: {args.file_pattern}")
    logger.info(f"Output Directory: {args.output_dir}")
    if not args.metadata_only:
        logger.info(f"Chunk Size: {args.chunk_size}")
        logger.info(f"Chunk Overlap: {args.chunk_overlap}")
        logger.info(f"HNSW ef_construction: {args.hnsw_ef_construction}")
        logger.info(f"HNSW ef_search: {args.hnsw_ef_search}")
        logger.info(f"HNSW max_neighbors: {args.hnsw_max_neighbors}")
        logger.info("Embedding: HuggingFace local model")

    # Auto-detect input directory if default doesn't exist or is empty
    start_time = time.time()
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        # Try centralized fallback
        if os.path.isdir(OUTDIR_10K):
            logger.info(f"Input dir '{input_dir}' not found. Using '{OUTDIR_10K}'")
            input_dir = OUTDIR_10K
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Rebuild cleanup must happen BEFORE indexing, not after
    if args.rebuild and not args.metadata_only:
        chroma_dir = os.path.join(args.output_dir, "chroma_db")
        index_dir = os.path.join(args.output_dir, "index")
        if os.path.isdir(chroma_dir):
            logger.info("Rebuild: removing existing Chroma DB directory...")
            shutil.rmtree(chroma_dir, ignore_errors=True)
        if os.path.isdir(index_dir):
            logger.info("Rebuild: removing existing index directory...")
            shutil.rmtree(index_dir, ignore_errors=True)

    processed_count = 0
    if args.metadata_only:
        # Metadata-only path (reads texts to build metadata index)
        documents = create_documents_from_files(
            input_dir,
            args.file_pattern,
            max_files=args.max_files,
            resolve_names=args.resolve_company_names,
        )
        if not documents:
            logger.error(
                "No documents found. Check your input directory and file pattern."
            )
            return
        metadata_index = create_metadata_index(documents)
        metadata_file = os.path.join(args.output_dir, "metadata_index.json")
        save_metadata_index(metadata_index, metadata_file)
        processed_count = len(documents)
    else:
        # Streaming index build; collect lightweight metadata
        metadata_docs = stream_and_index(
            input_dir,
            args.file_pattern,
            args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model_name=args.embedding_model or "BAAI/bge-base-en-v1.5",
            collection_name=args.collection_name,
            resolve_names=args.resolve_company_names,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            hnsw_max_neighbors=args.hnsw_max_neighbors,
            max_files=args.max_files,
        )
        if not metadata_docs:
            logger.error(
                "No documents found. Check your input directory and file pattern."
            )
            return
        metadata_index = create_metadata_index(metadata_docs)
        metadata_file = os.path.join(args.output_dir, "metadata_index.json")
        save_metadata_index(metadata_index, metadata_file)
        processed_count = len(metadata_docs)

    # Vector index already built in streaming path; performance stats
    elapsed_time = time.time() - start_time
    logger.info(f"Index creation completed in {elapsed_time:.2f} seconds")
    if not args.metadata_only:
        logger.info("Vector index ready for semantic search and RAG")
    # Report processed count
    logger.info(f"Documents processed: {processed_count}")


if __name__ == "__main__":
    main()
