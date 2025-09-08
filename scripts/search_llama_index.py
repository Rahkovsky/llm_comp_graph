#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Search and summarize content from a persisted LlamaIndex/Chroma index.

- Connects to an existing Chroma collection under --index-dir/chroma_db
- Reconstructs a VectorStoreIndex and performs similarity search
- Optionally produces a lightweight extractive summary (no external LLM required)

Examples:
  # Simple query
  python scripts/search_llama_index.py --index-dir data/llama_index --query "risk factors for AAPL"

  # Top-10 and extractive summary
  python scripts/search_llama_index.py --index-dir data/llama_index --query "revenue recognition" --top-k 10 --summarize --summary-sentences 7

  # Specify collection and local embedding backend
  python scripts/search_llama_index.py --index-dir data/llama_index --collection-name 10k_documents \
    --embedding-backend local-sbert --embedding-model BAAI/bge-base-en-v1.5 --query "supply chain"

  # Local LLAMA executive summary instead of extractive
  python scripts/search_llama_index.py --index-dir data/llama_index --collection-name 10k_documents \
    --query "attention economy" --top-k 8 --max-chars 800 --llm-summary
"""

import argparse
import os
import logging
import contextlib
import io
import atexit
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llm_comp_graph.constants import LLAMA_MODEL  # local llama.cpp model path
from llama_cpp import Llama  # local llama.cpp Python API

# Silence tokenizers parallelism warning when process forks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Silence LlamaIndex info logs (e.g., MockLLM notice)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)


def build_embed_model(backend: str, model_name: Optional[str]) -> Any:
    """Create an embedding model consistent with the index's embedding configuration.

    Note: For llama.cpp GGUF embeddings, many models lack an embedding head.
    This utility defaults to a local sentence-transformers backend.
    """
    if backend == "local-sbert":
        name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model_name=name, cache_folder="models/embeddings")
    if backend == "none":
        return None
    raise ValueError(f"Unknown embedding backend '{backend}'")


_llama: Optional[Llama] = None
_llama_cfg: Tuple[int, int, int, int] = (0, 0, 0, 0)


def _get_llama(
    n_ctx: int = 16384,
    n_gpu_layers: int = -1,
    n_threads: Optional[int] = None,
    n_batch: int = 512,
) -> Llama:
    global _llama, _llama_cfg
    if n_threads is None:
        try:
            n_threads = max(1, os.cpu_count() or 1)
        except Exception:
            n_threads = 4
    cfg = (n_ctx, n_gpu_layers, n_threads, n_batch)
    if _llama is None or _llama_cfg != cfg:
        # Suppress noisy Metal init logs during model load
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _llama = Llama(
                model_path=LLAMA_MODEL,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                n_batch=n_batch,
                embedding=False,
                verbose=False,
            )
        _llama_cfg = cfg
    return _llama


def _llm_summarize_with_llama(
    excerpts: List[str],
    *,
    companies: Optional[List[str]] = None,
    question: Optional[str] = None,
    system_prompt: Optional[str] = None,
    n_ctx: int = 16384,
    max_tokens: int = 512,
    temperature: float = 0.2,
    n_gpu_layers: int = -1,
) -> str:
    # Cap input to fit in context (rough estimate: 4 chars/token). Reserve ~256 tokens for system/user wrappers.
    reserved = max_tokens + 256
    max_input_tokens = max(512, n_ctx - reserved)
    max_input_chars = max_input_tokens * 4
    joined = "\n\n".join(excerpts)[:max_input_chars]
    company_hint = (
        "\nCompanies covered: " + ", ".join(sorted(set(companies or [])))
        if companies
        else ""
    )
    sys_txt = system_prompt or (
        "You are a precise financial analyst. Summarize evidence from SEC 10-K excerpts. "
        "Be concise, factual, and avoid speculation. Output only the summary, no preamble."
    )
    task_txt = (
        "Write an executive summary (6-10 sentences) that synthesizes cross-company risks and themes; "
        "avoid focusing on a single company; avoid repetition; generalize beyond any one brand."
    )
    if question:
        task_txt = f"Question: {question}\n" + task_txt
    prompt = (
        "system\n"
        + sys_txt
        + "\n"
        + "user\nExcerpts:\n"
        + joined[:12000]
        + company_hint
        + "\n\nTask: "
        + task_txt
        + "\nassistant\n"
    )
    llama = _get_llama(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    try:
        result = llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.15,
        )
        choices = result.get("choices", [])  # type: ignore[assignment]
        if choices:
            raw = str(choices[0].get("text", "")).strip()
            # Simple de-duplication of repeated sentences
            parts = [p.strip() for p in raw.split(". ") if p.strip()]
            seen = set()
            dedup: List[str] = []
            for p in parts:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    dedup.append(p)
            return (". ".join(dedup)).rstrip(".")
        return ""
    except Exception as e:
        return f"[LLM error: {e}]"


# Ensure clean teardown of llama-cpp model to avoid shutdown tracebacks
def _shutdown_llama() -> None:
    global _llama
    try:
        if _llama is not None:
            _llama.close()
    except Exception:
        pass
    _llama = None


atexit.register(_shutdown_llama)


def load_index(index_dir: str, collection_name: str, embed_model: Any) -> Any:
    """Reconstruct a VectorStoreIndex from a Chroma collection."""
    os.makedirs(index_dir, exist_ok=True)
    db_path = os.path.join(index_dir, "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Disable any default LLM resolution and set embedding/splitting
    Settings.llm = None
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter()
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )


def extractive_summary(texts: List[str], max_sentences: int = 5) -> str:
    """Very lightweight extractive summary using term-frequency scoring of sentences."""
    all_text = "\n".join(texts)  # Concatenate texts
    sentences = re.split(r"(?<=[.!?])\s+", all_text)  # Basic sentence split
    if not sentences:
        return ""

    words = re.findall(r"\w+", all_text.lower())  # Tokenize and compute term frequency
    if not words:
        return "\n".join(sentences[:max_sentences])
    freq: Dict[str, int] = Counter(words)

    # Score sentences by TF sum; prefer medium-length sentences
    scored: List[Tuple[float, str]] = []
    for sent in sentences:
        tokens = re.findall(r"\w+", sent.lower())
        if not tokens:
            continue
        score = sum(freq.get(tok, 0) for tok in tokens) / (1 + abs(len(tokens) - 20))
        scored.append((score, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    selected = {s for _, s in top}  # Preserve original order among selected
    ordered = [s for s in sentences if s in selected]
    return " ".join(ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Query and summarize a persisted LlamaIndex (Chroma-backed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--index-dir", default="llama_index", help="Directory where index was persisted"
    )
    parser.add_argument(
        "--collection-name", default="10k_documents", help="Chroma collection name"
    )
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=5000,
        help="Max characters to print from each result",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Produce extractive summary from retrieved chunks",
    )
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        help="Use local llama.cpp to generate an executive summary",
    )
    parser.add_argument(
        "--summary-sentences",
        type=int,
        default=5,
        help="Number of sentences in extractive summary",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Optional explicit question to guide LLM summary",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the default LLM system prompt",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["local-sbert", "none"],
        default="local-sbert",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name/path for local-sbert backend",
    )
    # llama.cpp embedding backend removed in this build

    args = parser.parse_args()

    embed_model = build_embed_model(args.embedding_backend, args.embedding_model)
    index = load_index(args.index_dir, args.collection_name, embed_model)

    print("🔎 Running query...")
    retriever = index.as_retriever(similarity_top_k=args.top_k)
    nodes = retriever.retrieve(args.query)

    print("\n📚 Top results:")  # Display sources
    texts_for_summary: List[str] = []
    companies_hint: List[str] = []
    for i, node in enumerate(nodes, start=1):
        meta = dict(getattr(node, "metadata", {}) or {})
        file_path = meta.get("file_path") or meta.get("relative_path") or "<unknown>"
        comp = str(meta.get("ticker") or meta.get("company_name") or "").strip()
        if comp:
            companies_hint.append(comp)
        score = getattr(node, "score", None)
        text = getattr(node, "text", "")

        print(f"  {i:2d}. {file_path}  score={score!s}")
        excerpt = (text or "").strip().replace("\n", " ")
        if len(excerpt) > args.max_chars:
            excerpt = excerpt[: args.max_chars] + "..."
        if excerpt:
            print(f"      {excerpt}")
        texts_for_summary.append(text)

    if args.llm_summary and texts_for_summary:
        print("\n📝 LLM executive summary (local llama):")
        summary = _llm_summarize_with_llama(
            texts_for_summary,
            companies=companies_hint,
            question=args.question,
            system_prompt=args.system_prompt,
            n_ctx=16384,
            max_tokens=512,
            temperature=0.2,
            n_gpu_layers=-1,
        )
        print(summary)
    elif args.summarize and texts_for_summary:
        print("\n📝 Extractive summary:")
        summary = extractive_summary(
            texts_for_summary, max_sentences=args.summary_sentences
        )
        print(summary)


if __name__ == "__main__":
    main()
