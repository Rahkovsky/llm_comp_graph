#!/usr/bin/env python3
import argparse
import glob
import os
import re
import subprocess
import time
from typing import List, Tuple, Dict, cast
import psutil
from llm_comp_graph.constants import OUTDIR_10K

# Import constants
from constants import LLAMA_CLI, LLAMA_MODEL  # type: ignore[reportMissingImports]


def check_bins():
    if not os.path.isfile(cast(str, LLAMA_CLI)):
        raise SystemExit(
            f"llama-cli not found at {LLAMA_CLI}. Check constants.py for correct path."
        )
    if not os.path.isfile(cast(str, LLAMA_MODEL)):
        raise SystemExit(
            f"Model not found at {LLAMA_MODEL}. Check constants.py for correct path."
        )

    print(f"Using LLama CLI: {LLAMA_CLI}")
    print(f"Using Model: {LLAMA_MODEL}")


def chat_prompt(system: str, user: str) -> str:
    return "system\n" + system + "\nuser\n" + user + "\nassistant\n"


def generate_qa_on_chunk(
    chunk_text: str, num_questions: int = 2
) -> List[Tuple[str, str]]:
    """Generate question-answer pairs from a text chunk."""
    system = f"""You are a financial analyst creating Q&A pairs from SEC filings.
Generate {num_questions} clear, factual question-answer pairs based on the provided text.
Each Q&A should be concise and accurate. Format as:
Q: [clear question]
A: [concise factual answer]

FOCUS ONLY on Microsoft's BUSINESS OPERATIONS, including:
- Products and services (Windows, Office, Azure, AI, etc.)
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


def analyze_content_richness(text: str) -> Dict[str, float | int | bool | str]:
    """Analyze text content to determine if it's rich enough for question generation."""
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
    ap.add_argument("--out", default="out/qa_10k.txt", help="Where to save Q&A pairs")
    ap.add_argument(
        "--per-chunk",
        type=int,
        default=2,
        help="Number of Q&A pairs to generate per text chunk",
    )
    ap.add_argument(
        "--max-files", type=int, default=10, help="Maximum files to process"
    )
    args = ap.parse_args()

    check_bins()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    print(f"Found {len(files)} files to process")
    if len(files) > args.max_files:
        print(f"Limiting to first {args.max_files} files")
        files = files[: args.max_files]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Memory monitoring
    process = psutil.Process()
    print(f"[START] Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    all_qa_pairs: List[Tuple[str, str]] = []

    for i, fp in enumerate(files):
        print(f"Processing file {i + 1}/{len(files)}: {os.path.basename(fp)}")

        # Monitor memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  [MEMORY] Current usage: {memory_mb:.1f} MB")

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
                qa_pairs = generate_qa_on_chunk(ch, optimal_questions)

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    print(f"    ✓ Chunk {j + 1}: Generated {len(qa_pairs)} Q&A pairs")
                else:
                    print(f"    ⚠ Chunk {j + 1}: No Q&A pairs generated")

                # Memory check after each chunk
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"    [MEMORY] After chunk {j + 1}: {memory_mb:.1f} MB")

                time.sleep(0.1)  # Reduced rate limiting

        except Exception as e:
            print(f"  [ERROR] Failed to process {fp}: {e}")
            continue
        finally:
            # Clean up memory after each file
            cleanup_memory()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"  [MEMORY] After cleanup: {memory_mb:.1f} MB")

    # Final dedup
    seen: set[str] = set()
    final_qa: List[Tuple[str, str]] = []
    for q, a in all_qa_pairs:
        k = re.sub(r"[^\w]+", " ", q.lower()).strip()
        if k and k not in seen:
            seen.add(k)
            final_qa.append((q, a))

    with open(args.out, "w", encoding="utf-8") as f:
        for q, a in final_qa:
            f.write(f"Q: {q}\nA: {a}\n\n")

    print(f"\nDone! Wrote {len(final_qa)} unique Q&A pairs -> {args.out}")


if __name__ == "__main__":
    main()
