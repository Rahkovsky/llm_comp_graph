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
from llm_comp_graph.constants import (
    LLAMA_MODEL,
    INDEX_DIR as DEFAULT_INDEX_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_LLM_CTX,
)
from llama_cpp import Llama  # local llama.cpp Python API

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # Silence tokenizers parallelism warning, real parallelization is handled by Rust
logging.getLogger("llama_index").setLevel(
    logging.WARNING
)  # Silence LlamaIndex info logs (e.g., MockLLM notice)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)


def build_embed_model(backend: str, model_name: Optional[str]) -> Any:
    """Create an embedding model consistent with the index's embedding configuration."""
    if backend == "local-sbert":
        name = model_name or DEFAULT_EMBED_MODEL
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
    """Create or reuse a cached llama.cpp model instance.

    - n_ctx: Max context tokens for prompt + generation.
    - n_gpu_layers: Layers to offload to GPU (-1 = auto, as many as fit).
    - n_threads: CPU threads for CPU-resident ops; None auto-detects.
    - n_batch: Prompt processing batch size (tokens per step).

    Returns the cached Llama instance configured for LLAMA_MODEL.
    """
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


def _rewrite_query_with_llama(
    original_query: str,
    *,
    n_queries: int = 3,
    n_ctx: int = DEFAULT_LLM_CTX,
    max_tokens: int = 200,
    temperature: float = 0.3,
    n_gpu_layers: int = -1,
) -> List[str]:
    """Use local LLM to rewrite/expand user query into multiple optimized queries.

    This improves recall by generating queries that are more likely to match
    the specific language and terminology used in 10-K financial documents.
    """
    prompt = f"""You are a financial analyst expert. Rewrite the user's query into {n_queries} different, specific queries that would be effective for searching SEC 10-K filings. Each query should use precise financial terminology and be optimized for finding relevant information.

Original query: "{original_query}"

Generate {n_queries} rewritten queries, one per line. Each query should:
- Use specific financial/legal terminology from 10-K filings
- Focus on different aspects of the original question
- Be concise but comprehensive
- Avoid vague terms like "key" or "important"

Rewritten queries:"""

    llama = _get_llama(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    try:
        result = llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.1,
        )
        choices = result.get("choices", [])
        if choices:
            raw_text = str(choices[0].get("text", "")).strip()
            # Parse the generated queries
            queries = [q.strip() for q in raw_text.split("\n") if q.strip()]
            # Filter out empty or too-short queries
            queries = [
                q
                for q in queries
                if len(q) > 10 and not q.startswith(("Query", "1.", "2.", "3."))
            ]
            # Limit to requested number and add original if needed
            queries = queries[:n_queries]
            if len(queries) < n_queries:
                queries.append(original_query)  # Fallback to original
            return queries
    except Exception as e:
        print(f"[Query rewriting failed: {e}, using original query]")

    return [original_query]


def _fuse_query_results(
    all_nodes: List[Any], *, top_k: int, fusion_method: str = "rrf"
) -> List[Any]:
    """Fuse results from multiple queries using Reciprocal Rank Fusion (RRF).

    RRF combines rankings from multiple queries by scoring each document
    based on its rank across all queries. Higher scores = better overall relevance.
    """
    if not all_nodes:
        return []

    # Group nodes by their text content (deduplication)
    seen_texts = set()
    unique_nodes = []
    for node in all_nodes:
        text = getattr(node, "text", "")
        if text not in seen_texts:
            seen_texts.add(text)
            unique_nodes.append(node)

    if len(unique_nodes) <= top_k:
        return unique_nodes

    # Simple RRF scoring: each node gets 1/(rank + 60) points per query
    # We'll use a simplified approach since we don't have explicit ranks
    node_scores = {}
    for i, node in enumerate(unique_nodes):
        text = getattr(node, "text", "")
        # Simple scoring: earlier appearance = higher score
        score = 1.0 / (i + 60)  # RRF constant of 60
        if text in node_scores:
            node_scores[text] += score
        else:
            node_scores[text] = score

    # Sort by combined score and return top_k
    scored_nodes: List[Tuple[float, Any]] = [
        (node_scores.get(getattr(node, "text", ""), 0), node) for node in unique_nodes
    ]
    scored_nodes.sort(key=lambda x: x[0], reverse=True)

    return [node for _, node in scored_nodes[:top_k]]


def _llm_summarize_with_llama(
    excerpts: List[str],
    *,
    companies: Optional[List[str]] = None,
    question: Optional[str] = None,
    system_prompt: Optional[str] = None,
    n_ctx: int = DEFAULT_LLM_CTX,
    max_tokens: int = 512,
    temperature: float = 0.2,
    n_gpu_layers: int = -1,
) -> str:
    # Preflight: estimate token usage and enforce only n_ctx to avoid OOM.
    # Approx: 1 token ≈ 4 chars; reserve ~256 tokens for wrappers.
    reserved = max_tokens + 256
    est_tokens = sum(max(1, len(x) // 4) for x in excerpts)
    max_allowed = max(512, n_ctx - reserved)
    if est_tokens > max_allowed:
        raise ValueError(
            f"LLM input too large: est_tokens={est_tokens} > allowed={max_allowed} (n_ctx={n_ctx}, reserved={reserved}). "
            "Reduce --top-k, increase n_ctx, or lower chunk size during indexing."
        )
    joined = "\n\n".join(excerpts)
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
        "avoid focusing on a single company; avoid repetition; generalize beyond any one brand; try to be detailed and specific, avoid generalities;"
    )
    if question:
        task_txt = f"Question: {question}\n" + task_txt
    prompt = (
        "system\n"
        + sys_txt
        + "\n"
        + "user\nExcerpts:\n"
        + joined
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
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory where index was persisted",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name",
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
        "--llm-n-ctx",
        type=int,
        default=DEFAULT_LLM_CTX,
        help="Context window for local llama.cpp (tokens)",
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
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model name/path for local-sbert backend",
    )
    parser.add_argument(
        "--rewrite-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use local LLM to rewrite/expand queries for better recall",
    )
    parser.add_argument(
        "--n-rewritten-queries",
        type=int,
        default=3,
        help="Number of rewritten queries to generate (default: 3)",
    )
    parser.add_argument(
        "--query-rewrite-temp",
        type=float,
        default=0.3,
        help="Temperature for query rewriting (default: 0.3)",
    )
    # llama.cpp embedding backend removed in this build

    args = parser.parse_args()

    embed_model = build_embed_model(args.embedding_backend, args.embedding_model)
    index = load_index(args.index_dir, args.collection_name, embed_model)

    # Query rewriting for improved recall
    if args.rewrite_queries:
        print("🔄 Rewriting queries for better recall...")
        rewritten_queries = _rewrite_query_with_llama(
            args.query,
            n_queries=args.n_rewritten_queries,
            n_ctx=args.llm_n_ctx,
            temperature=args.query_rewrite_temp,
        )
        print(f"📝 Generated {len(rewritten_queries)} queries:")
        for i, q in enumerate(rewritten_queries, 1):
            print(f"  {i}. {q}")

        # Retrieve results for each rewritten query
        retriever = index.as_retriever(
            similarity_top_k=args.top_k * 2
        )  # Get more per query
        all_nodes = []
        for query in rewritten_queries:
            query_nodes = retriever.retrieve(query)
            all_nodes.extend(query_nodes)

        # Fuse results using RRF
        nodes = _fuse_query_results(all_nodes, top_k=args.top_k)
        print(f"🔗 Fused {len(all_nodes)} results into {len(nodes)} unique documents")
    else:
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
            n_ctx=args.llm_n_ctx,
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
