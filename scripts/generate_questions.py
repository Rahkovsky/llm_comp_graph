#!/usr/bin/env python3
import argparse
import glob
import os
import re
import subprocess
import time
from typing import List, Tuple, Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
from llm_comp_graph.constants import OUTDIR_10K, LLAMA_CLI, LLAMA_MODEL


def check_bins():
    if not os.path.isfile(LLAMA_CLI):
        raise SystemExit(
            f"llama-cli not found at {LLAMA_CLI}. Check constants.py for correct path."
        )
    if not os.path.isfile(LLAMA_MODEL):
        raise SystemExit(
            f"Model not found at {LLAMA_MODEL}. Check constants.py for correct path."
        )

    print(f"Using LLama CLI: {LLAMA_CLI}")
    print(f"Using Model: {LLAMA_MODEL}")


def chat_prompt(system: str, user: str) -> str:
    return "system\n" + system + "\nuser\n" + user + "\nassistant\n"


def generate_qa_on_chunk(
    chunk_text: str, num_questions: int = 2, company_hint: str | None = None
) -> List[Tuple[str, str]]:
    """Generate question-answer pairs from a text chunk."""
    company_context = (
        f"Company context: ticker {company_hint}. Anchor questions and answers to this company only.\n"
        if company_hint
        else ""
    )

    system = f"""You are a financial analyst creating Q&A pairs from SEC filings.
Generate {num_questions} clear, factual question-answer pairs based on the provided text.
Each Q&A must be specific, grounded in the text, and avoid vague generalities.
Favor concrete details: named products, segments, metrics, dates, percentages, model names.
Do NOT invent facts. Use only what appears in the text.

Strict rules:
- Questions must NOT be yes/no questions.
- Answers must be concise (≤ 50 words).

Format strictly as:
Q: [clear, specific question]
A: [concise factual answer ≤ 50 words]

{company_context}FOCUS ONLY on the company's BUSINESS OPERATIONS discussed in the text, including:
- Products and services explicitly mentioned in the text
- Business strategy and mission
- Financial performance and results
- Market position and competitive advantages
- Technology innovations and R&D
- Customer segments and markets
- Partnerships and acquisitions

AVOID bureaucratic/form-filling questions about:
- Filing dates, form numbers, page numbers
- SEC compliance details
- Administrative procedures
- Generic regulatory requirements

IMPORTANT: Generate {num_questions} Q&A pairs."""

    user = f"Text: {chunk_text[:500]}\n\nGenerate {num_questions} Q&A pairs:"

    response = run_llama(chat_prompt(system, user), n=256, temp=0.1)

    if not response:
        print("      [DEBUG] No LLM response")
        return []

    print(f"      [DEBUG] Raw response: {response[:200]}...")

    # Parse Q&A pairs from response
    qa_pairs: List[Tuple[str, str]] = []
    lines = response.strip().split("\n")
    current_q = None

    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            current_q = line[2:].strip()
            print(f"      [DEBUG] Found question: {current_q[:50]}...")
        elif line.startswith("A:") and current_q:
            answer = line[2:].strip()
            print(f"      [DEBUG] Found answer: {answer[:50]}...")
            if (
                current_q and answer and len(answer) > 10
            ):  # Filter out very short answers
                qa_pairs.append((current_q, answer))
                print("      [DEBUG] Added Q&A pair")
            else:
                print(
                    f"      [DEBUG] Rejected answer (len={len(answer) if answer else 0})"
                )
            current_q = None

        print(f"      [DEBUG] Total Q&A pairs found: {len(qa_pairs)}")

    # Accept any valid Q&A pairs (1 or more)
    if len(qa_pairs) >= 1:
        print(f"      [DEBUG] Returning {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    else:
        print("      [DEBUG] No valid pairs found, returning empty")
        return []


def _looks_generic_answer(answer: str) -> bool:
    """Heuristic check for overly generic answers."""
    if not answer:
        return True
    normalized = answer.lower().strip()
    if len(normalized) < 15:
        return True
    generic_phrases = [
        "help customers",
        "solve challenges",
        "drive growth",
        "improve productivity",
        "enable organizations",
        "support innovation",
        "cloud-based solutions",
        "respond to cyberthreats",
        "ai solutions",
        "industry-leading",
        "at scale",
        "end-to-end",
        "across industries",
        "across every industry",
    ]
    if any(p in normalized for p in generic_phrases):
        # May still be acceptable if it contains specifics below
        pass

    # Look for specificity signals
    has_number = bool(re.search(r"\b(\d{4}|\d+%|\$?\d+[.,]?\d*)\b", answer))
    has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", answer))
    # Capitalized proper nouns beyond sentence start (very rough)
    # Recognize a broader set of common proper nouns across companies; still non-exhaustive
    has_proper = bool(
        re.search(
            r"\b(?:Azure|Windows|Office|Dynamics|LinkedIn|Xbox|Nuance|Copilot|Fabric|Surface|Maia|Cobalt|iPhone|iPad|Mac|Vision Pro|Apple Watch|Apple TV|Siri|A17|M3|Prime|AWS|Kindle|Alexa|Echo|NVIDIA|GeForce|RTX|CUDA|DGX|H100|GH200|Instinct|Ryzen|EPYC)\b",
            answer,
        )
    )

    specificity_score = sum([has_number, has_acronym, has_proper])
    if specificity_score == 0 and any(p in normalized for p in generic_phrases):
        return True
    return False


def refine_qa_pair_with_llm(
    context: str, question: str, answer: str, company_hint: str | None = None
) -> Tuple[str, str] | None:
    """Use the local model to rewrite a Q&A to be specific and grounded.

    Returns (q, a) if to keep, or None if should reject.
    """
    company_context = (
        f"Company context: ticker {company_hint}. Keep the Q&A anchored to this company only.\n"
        if company_hint
        else ""
    )

    system = (
        "You are a strict Q&A reviewer. Improve specificity of Q&A using only the provided text.\n"
        "Requirements:\n"
        "- Questions must NOT be yes/no style; rewrite into specific fact-based questions.\n"
        "- Keep only if the answer is concrete and grounded in the text.\n"
        "- Prefer explicit entities: product names, segments, metrics, dates, model names.\n"
        "- If the answer is generic/vague and cannot be grounded with specifics from the text, REJECT.\n"
        "- Do NOT invent facts.\n"
        f"{company_context}"
        "Output strictly in this format:\n"
        "KEEP\nQ: <rewritten or original>\nA: <concise answer ≤ 50 words>\n"
        "or\nREJECT\n"
    )
    user = (
        f"Text:\n{context[:800]}\n\n"
        f"Original Q:\n{question}\n"
        f"Original A:\n{answer}\n\n"
        "Review and output as specified."
    )
    resp = run_llama(chat_prompt(system, user), n=256, temp=0.0)
    if not resp:
        return None

    text = resp.strip()
    if text.startswith("REJECT"):
        return None
    # Try to parse KEEP block
    # Accept small leading text until KEEP
    m_keep = re.search(r"KEEP[\s\S]*?Q:\s*(.+?)\s*\nA:\s*(.+)", text)
    if not m_keep:
        return None
    new_q = m_keep.group(1).strip()
    new_a = m_keep.group(2).strip()

    # Basic sanity checks
    if not new_q or not new_a:
        return None
    if _looks_generic_answer(new_a) or _is_yes_no_question(new_q):
        return None
    # Enforce short answers
    if len(new_a.split()) > 50:
        # Soft trim on sentence boundary
        sentences = re.split(r"(?<=[.!?])\s+", new_a)
        new_a = sentences[0].strip() if sentences else new_a[:200].strip()
    return (new_q, new_a)


def _looks_generic_question(question: str) -> bool:
    """Heuristic check for overly generic questions."""
    if not question:
        return True
    q = question.lower().strip()
    if len(q) < 10:
        return True
    generic_starts = [
        "what is ",
        "what are ",
        "what does ",
        "what do ",
        "what type",
        "what kinds",
        "what is the purpose",
        "what is the goal",
        "what is the focus",
    ]
    if any(q.startswith(gs) for gs in generic_starts):
        # Still allow if it names a specific product/entity
        has_entity = bool(
            re.search(
                r"\b(?:azure|copilot|fabric|nuance|xbox|windows 11|maia|cobalt|dynamics 365|power platform|iphone|ipad|mac|vision pro|apple watch|siri|prime|aws|kindle|alexa|echo|nvidia|geforce|rtx|cuda|dgx|h100|gh200|instinct|ryzen|epyc)\b",
                q,
            )
        )
        if not has_entity:
            return True
    return False


def _is_yes_no_question(question: str) -> bool:
    """Detect if a question is yes/no style."""
    if not question:
        return False
    q = question.strip().lower()
    # Common yes/no starts
    yn_starts = [
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "will ",
        "has ",
        "have ",
        "had ",
        "isn't ",
        "aren't ",
        "doesn't ",
        "can't ",
        "won't ",
        "haven't ",
        "hasn't ",
    ]
    return any(q.startswith(s) for s in yn_starts)


def run_llama(prompt: str, n: int = 256, temp: float = 0.2, ctx: int = 2048) -> str:
    cmd: List[str] = [
        LLAMA_CLI,
        "-m",
        LLAMA_MODEL,
        "-n",
        str(n),
        "-ngl",
        "999",
        "-c",
        str(ctx),
        "--repeat_penalty",
        "1.1",
        "--temp",
        str(temp),
        "-p",
        prompt,
        "--no-conversation",
        "--no-warmup",
    ]
    try:
        # Reduced timeout to prevent hanging
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
        return out.stdout
    except subprocess.TimeoutExpired:
        print("[WARN] LLM timeout for chunk, skipping...")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LLM failed: {e}")
        return ""


def chunk(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """Split text into overlapping chunks with better boundary handling."""
    if len(text) <= max_chars:
        return [text]

    blocks: List[str] = []
    i = 0
    chunk_count = 0
    max_chunks = 50  # Limit total chunks to prevent memory issues

    while i < len(text) and chunk_count < max_chunks:
        j = min(len(text), i + max_chars)

        # Try to find a good break point (sentence/paragraph end)
        if j < len(text):
            # Look for sentence endings within the last 200 chars
            for k in range(j, max(i + max_chars - 200, i), -1):
                if text[k] in ".!?":
                    j = k + 1
                    break
            # If no sentence end, look for paragraph breaks
            if j == min(len(text), i + max_chars):
                for k in range(j, max(i + max_chars - 100, i), -1):
                    if text[k : k + 2] == "\n\n":
                        j = k + 2
                        break

        chunk_text = text[i:j].strip()
        if chunk_text and len(chunk_text) > 100:  # Only add substantial chunks
            blocks.append(chunk_text)
            chunk_count += 1

        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break

    print(f"  [CHUNK] Created {len(blocks)} chunks from {len(text)} chars")
    return blocks


def validate_text_quality(text: str) -> bool:
    """Check if extracted text is readable and not just metadata."""
    if not text or len(text) < 100:
        return False

    # Check for excessive metadata/URLs
    url_count = len(re.findall(r"http[s]?://", text))
    if url_count > len(text) / 1000:  # More than 1 URL per 1000 chars
        return False

    # Check for readable content (words, sentences)
    words = re.findall(r"\b\w+\b", text)
    sentences = re.findall(r"[.!?]+", text)

    if len(words) < 50 or len(sentences) < 3:
        return False

    return True


def _get_mem_mb() -> float | None:
    """Get current process memory in MB if psutil is available."""
    try:
        if psutil is None:  # type: ignore
            return None
        proc = psutil.Process()  # type: ignore
        return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return None


def analyze_content_richness(text: str) -> Dict[str, float | int | bool | str]:
    """Analyze text content to determine if it's rich enough for question generation. Ask question about specific details that go beyond the general overview. The question should really show deep understanding of the business. If there is no such information, feel free to return False."""
    if not text or len(text) < 100:
        return {"rich_enough": False, "reason": "Text too short", "score": 0.0}

    # Count meaningful content indicators
    sentences = re.findall(r"[.!?]+", text)
    words = re.findall(r"\b\w+\b", text)

    # Look for business-specific content
    business_keywords = [
        "revenue",
        "profit",
        "income",
        "sales",
        "growth",
        "market",
        "customer",
        "product",
        "service",
        "technology",
        "innovation",
        "strategy",
        "performance",
        "financial",
        "operating",
        "business",
        "development",
        "research",
        "acquisition",
        "partnership",
        "competition",
        "investment",
        "expansion",
        "launch",
        "release",
    ]

    keyword_matches = sum(
        1 for keyword in business_keywords if keyword.lower() in text.lower()
    )

    # Calculate content richness score
    sentence_count = len(sentences)
    word_count = len(words)
    keyword_density = keyword_matches / max(word_count, 1)

    # Score based on multiple factors
    score = 0.0
    score += min(sentence_count / 10.0, 1.0) * 0.4  # Sentence density (max 40%)
    score += min(keyword_density * 100, 1.0) * 0.4  # Business keyword density (max 40%)
    score += min(word_count / 500.0, 1.0) * 0.2  # Overall content length (max 20%)

    # Determine if content is rich enough
    rich_enough = score >= 0.3  # Threshold for generating questions

    reason = f"Score: {score:.2f} (sentences: {sentence_count}, keywords: {keyword_matches}, words: {word_count})"

    return {
        "rich_enough": bool(rich_enough),
        "reason": reason,
        "score": float(score),
        "sentence_count": int(sentence_count),
        "keyword_count": int(keyword_matches),
        "word_count": int(word_count),
    }


def _derive_company_display_name(full_text: str, ticker: str | None) -> str | None:
    """Attempt to derive a human-readable company name from the filing text.

    Returns e.g. "Microsoft Corporation (MSFT)" or just the ticker if name not found.
    """
    try:
        hay = full_text[:5000]
        # 1) Look for the SEC standard marker
        m = re.search(
            r"Exact name of registrant as specified in its charter\s*([\w\s.,&'\-]{3,120})",
            hay,
            flags=re.IGNORECASE,
        )
        name = None
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r"\s+", " ", candidate)
            if 3 <= len(candidate) <= 120:
                name = candidate

        # 2) Heuristic: First company-like phrase in header
        if not name:
            m2 = re.search(
                r"\b([A-Z][A-Za-z0-9&.,'\- ]{2,80})\s+(Corporation|Corp\.|Company|Inc\.|Incorporated|Holdings|PLC|Limited|Ltd\.)\b",
                hay,
            )
            if m2:
                name = (m2.group(0)).strip()

        if not name and ticker:
            return f"{ticker}"
        if name and ticker:
            return f"{name} ({ticker})"
        return name
    except Exception:
        return ticker


def determine_optimal_question_count(text: str, max_questions: int = 5) -> int:
    """Determine how many questions to generate based on content richness."""
    analysis = analyze_content_richness(text)

    if not analysis["rich_enough"]:
        return 0

    # Base question count on content score
    base_count = max(1, int(analysis["score"] * max_questions))

    # Adjust based on content length and keyword density
    if int(analysis["sentence_count"]) >= 15 and int(analysis["keyword_count"]) >= 8:
        question_count = min(base_count + 1, max_questions)
    elif int(analysis["sentence_count"]) >= 10 and int(analysis["keyword_count"]) >= 5:
        question_count = base_count
    else:
        question_count = max(1, base_count - 1)

    return question_count


def cleanup_memory():
    """Force garbage collection to free memory."""
    import gc

    gc.collect()
    if hasattr(gc, "garbage"):
        gc.garbage.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default=f"{OUTDIR_10K}/*/*/*.plain.txt",
        help="Glob for plain text 10-K files",
    )
    ap.add_argument(
        "--tickers",
        action="append",
        default=[],
        help="Comma-separated list of stock tickers to include (can be repeated)",
    )
    ap.add_argument(
        "--companies",
        action="append",
        default=[],
        help="Alias for --tickers (comma-separated).",
    )
    ap.add_argument("--out", default="out/qa_10k.txt", help="Where to save Q&A pairs")
    ap.add_argument(
        "--save-provenance",
        default="",
        help="Optional path to save provenance JSON with ticker/file/chunk for each Q&A",
    )
    ap.add_argument(
        "--per-chunk",
        type=int,
        default=2,
        help="Number of Q&A pairs to generate per text chunk",
    )
    ap.add_argument(
        "--max-files", type=int, default=10, help="Maximum files to process"
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="List matched input files and exit (no generation)",
    )
    args = ap.parse_args()

    check_bins()
    # Build list of files, optionally filtered by tickers
    ticker_inputs: List[str] = []
    tickers_groups: List[str] = []
    try:
        for g in args.tickers:  # type: ignore[attr-defined]
            if g:
                tickers_groups.append(str(g))
    except Exception:
        pass
    try:
        for g in args.companies:  # type: ignore[attr-defined]
            if g:
                tickers_groups.append(str(g))
    except Exception:
        pass

    for group in tickers_groups:
        parts: List[str] = [p for p in group.split(",")]
        for p in parts:
            p_clean = p.strip().upper()
            if p_clean:
                ticker_inputs.append(p_clean)

    ticker_set = sorted(set(ticker_inputs))

    if ticker_set:
        print(f"Using ticker filter: {', '.join(ticker_set)}")
        files_set: set[str] = set()
        for t in ticker_set:
            patterns = [
                f"{OUTDIR_10K}/{t}/*/*.plain.txt",
                f"{OUTDIR_10K}/{t}/*.plain.txt",
                f"{OUTDIR_10K}/{t}/*/*.txt",
                f"{OUTDIR_10K}/{t}/*.txt",
            ]
            for g in patterns:
                for fp in glob.glob(g):
                    files_set.add(fp)
        files = sorted(files_set)
    else:
        files = sorted(glob.glob(args.glob))
    if not files:
        if ticker_set:
            raise SystemExit(f"No files matched for tickers: {', '.join(ticker_set)}")
        raise SystemExit(f"No files matched: {args.glob}")

    print(f"Found {len(files)} files to process")
    if args.list_only:
        # Show a preview of matched files and exit
        for idx, fp in enumerate(files[:10], 1):
            print(f"  [{idx}] {fp}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
        return
    if len(files) > args.max_files:
        print(f"Limiting to first {args.max_files} files")
        files = files[: args.max_files]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Memory monitoring
    mem = _get_mem_mb()
    if mem is not None:
        print(f"[START] Initial memory: {mem:.1f} MB")

    all_qa_pairs: List[Tuple[str, str, str]] = []  # (ticker, question, answer)
    provenance: List[Dict[str, str]] = []

    for i, fp in enumerate(files):
        print(f"Processing file {i + 1}/{len(files)}: {os.path.basename(fp)}")
        # Infer ticker from directory name: data/input/10K/<TICKER>/<file>
        try:
            ticker = os.path.basename(os.path.dirname(fp)).upper()
        except Exception:
            ticker = None  # type: ignore
        # Derive friendly company display (Name + ticker) from text if possible

        # Monitor memory usage
        mem = _get_mem_mb()
        if mem is not None:
            print(f"  [MEMORY] Current usage: {mem:.1f} MB")

        try:
            # Try multiple encodings to handle different file formats
            txt = None
            for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(fp, "r", encoding=encoding) as f:
                        txt = f.read()
                    print(f"  [ENCODING] Successfully read with {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if txt is None:
                print("  [ERROR] Could not read file with any encoding")
                continue

            # Clean corrupted characters from the text
            # Replace non-breaking spaces and other problematic characters
            txt = re.sub(r"\xc2\xa0", " ", txt)  # Non-breaking space
            txt = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", txt
            )  # Control characters
            txt = re.sub(r"\s+", " ", txt)  # Normalize whitespace

            # More aggressive cleaning for problematic bytes
            txt = txt.encode("utf-8", errors="ignore").decode("utf-8")
            # Remove any remaining invalid characters
            txt = "".join(char for char in txt if ord(char) < 0x10000)

            # Ensure the text is completely clean before processing
            txt = txt.encode("ascii", errors="ignore").decode("ascii")

            if not validate_text_quality(txt):
                print(f"  [SKIP] Poor text quality: {len(txt)} chars")
                continue

            print(f"  [OK] Text length: {len(txt)} chars (cleaned)")
            company_display = None
            try:
                company_display = _derive_company_display_name(txt, ticker)
            except Exception:
                company_display = ticker
            chunks = chunk(
                txt, max_chars=500, overlap=50
            )  # Dramatically reduced chunk size
            print(f"  [OK] Split into {len(chunks)} chunks")

            for j, ch in enumerate(chunks):
                if not validate_text_quality(ch):
                    continue

                print(f"    Processing chunk {j + 1}/{len(chunks)}...")

                # Analyze content richness and determine optimal question count
                content_analysis = analyze_content_richness(ch)
                optimal_questions = determine_optimal_question_count(ch, args.per_chunk)

                print(f"      [ANALYSIS] {content_analysis['reason']}")

                if optimal_questions == 0:
                    print("      [SKIP] Content not rich enough for questions")
                    continue

                print(f"      [QUESTIONS] Will generate {optimal_questions} questions")
                qa_pairs = generate_qa_on_chunk(
                    ch, optimal_questions, company_hint=company_display
                )

                refined_pairs: List[Tuple[str, str]] = []
                for q, a in qa_pairs:
                    if _is_yes_no_question(q):
                        refined = refine_qa_pair_with_llm(
                            ch, q, a, company_hint=company_display
                        )
                        if refined:
                            refined_pairs.append(refined)
                        else:
                            print("      [FILTER] Rejected yes/no style Q after review")
                        continue

                    if _looks_generic_question(q) or _looks_generic_answer(a):
                        refined = refine_qa_pair_with_llm(
                            ch, q, a, company_hint=company_display
                        )
                        if refined:
                            refined_pairs.append(refined)
                        else:
                            print("      [FILTER] Rejected generic Q&A after review")
                    else:
                        refined_pairs.append((q, a))

                if refined_pairs:
                    for qk, ak in refined_pairs:
                        all_qa_pairs.append((ticker if ticker else "", qk, ak))
                        if company_display:
                            provenance.append(
                                {
                                    "ticker": ticker or "",
                                    "file": fp,
                                    "company": company_display,
                                    "chunk": ch,
                                    "question": qk,
                                    "answer": ak,
                                }
                            )
                    print(
                        f"    ✓ Chunk {j + 1}: Kept {len(refined_pairs)} refined Q&A pairs"
                    )
                else:
                    print(
                        f"    ⚠ Chunk {j + 1}: All generated Q&A pairs were rejected as too generic"
                    )

                # Memory check after each chunk
                mem = _get_mem_mb()
                if mem is not None:
                    print(f"    [MEMORY] After chunk {j + 1}: {mem:.1f} MB")

                time.sleep(0.1)  # Reduced rate limiting

        except Exception as e:
            print(f"  [ERROR] Failed to process {fp}: {e}")
            continue
        finally:
            # Clean up memory after each file
            cleanup_memory()
            mem = _get_mem_mb()
            if mem is not None:
                print(f"  [MEMORY] After cleanup: {mem:.1f} MB")

    # Final dedup
    seen: set[str] = set()
    final_qa: List[Tuple[str, str]] = []
    for tk, q, a in all_qa_pairs:
        base = re.sub(r"[^\w]+", " ", q.lower()).strip()
        k = f"{tk}:{base}" if tk else base
        if k and k not in seen:
            seen.add(k)
            final_qa.append((q, a))

    with open(args.out, "w", encoding="utf-8") as f:
        for q, a in final_qa:
            f.write(f"Q: {q}\nA: {a}\n\n")

    print(f"\nDone! Wrote {len(final_qa)} unique Q&A pairs -> {args.out}")

    if args.save_provenance:
        try:
            import json as _json

            outp = args.save_provenance
            os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
            with open(outp, "w", encoding="utf-8") as pf:
                _json.dump(provenance, pf, indent=2, ensure_ascii=False)
            print(f"Saved provenance for {len(provenance)} items -> {outp}")
        except Exception as e:
            print(f"[WARN] Failed to save provenance: {e}")


if __name__ == "__main__":
    main()
