#!/usr/bin/env python3
"""Test untrained model performance on generated Q&A pairs."""

import argparse
import json
import os
import time
import subprocess
from typing import List, Dict, Tuple

from constants import LLAMA_CLI, LLAMA_MODEL


def check_bins():
    """Check if required binaries exist."""
    if not os.path.exists(LLAMA_CLI):
        raise SystemExit(f"llama-cli not found at {LLAMA_CLI}. Set $LLAMA_CLI.")
    if not os.path.exists(LLAMA_MODEL):
        raise SystemExit(f"Model not found at {LLAMA_MODEL}. Set $LLAMA_MODEL.")


def run_llama(prompt: str, n: int = 256, temp: float = 0.2, ctx: int = 2048) -> str:
    """Run llama-cli with given prompt."""
    cmd = [
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
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
        return out.stdout.strip()
    except subprocess.TimeoutExpired:
        print("[WARN] LLM timeout, skipping...")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LLM failed: {e}")
        return ""


def simple_prompt(question: str) -> str:
    """Create a simple, direct prompt without system instructions."""
    return f"Question: {question}\nAnswer:"


def load_qa_pairs(qa_file: str) -> List[Tuple[str, str]]:
    """Load Q&A pairs from the generated file."""
    qa_pairs = []
    current_q = None

    try:
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Q: "):
                    current_q = line[3:].strip()
                elif line.startswith("A: ") and current_q:
                    answer = line[3:].strip()
                    if current_q and answer:
                        qa_pairs.append((current_q, answer))
                    current_q = None
    except Exception as e:
        print(f"[ERROR] Failed to load Q&A file: {e}")
        return []

    return qa_pairs


def test_model_on_questions(
    qa_pairs: List[Tuple[str, str]], max_questions: int = None, delay: float = 0.5
) -> List[Dict]:
    """Test the model on a list of questions."""
    results = []

    if max_questions:
        qa_pairs = qa_pairs[:max_questions]

    print(f"Testing model on {len(qa_pairs)} questions...")

    for i, (question, expected_answer) in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] Testing: {question[:60]}...")

        # Create simple prompt without system instructions
        prompt = simple_prompt(question)

        # Get model response
        start_time = time.time()
        model_response = run_llama(prompt, n=128, temp=0.1)
        response_time = time.time() - start_time

        if not model_response:
            print("    ⚠ No response from model")
            continue

        # Store results
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "model_response": model_response,
            "response_time": response_time,
            "timestamp": time.time(),
        }
        results.append(result)

        print(f"    ✓ Response: {model_response[:80]}...")
        print(f"    ⏱ Time: {response_time:.2f}s")

        # Rate limiting
        time.sleep(delay)

    return results


def save_results(results: List[Dict], output_file: str):
    """Save test results to JSON file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} test results to: {output_file}")


def main():
    ap = argparse.ArgumentParser(description="Test untrained model on Q&A pairs")
    ap.add_argument("--qa-file", required=True, help="Path to Q&A pairs file")
    ap.add_argument(
        "--out", default="out/model_test_results.json", help="Output JSON file"
    )
    ap.add_argument(
        "--max-questions", type=int, default=50, help="Maximum questions to test"
    )
    ap.add_argument(
        "--delay", type=float, default=0.5, help="Delay between questions (seconds)"
    )

    args = ap.parse_args()

    # Check requirements
    check_bins()

    if not os.path.exists(args.qa_file):
        raise SystemExit(f"Q&A file not found: {args.qa_file}")

    print(f"Using LLama CLI: {LLAMA_CLI}")
    print(f"Using Model: {LLAMA_MODEL}")
    print(f"Loading Q&A pairs from: {args.qa_file}")

    # Load Q&A pairs
    qa_pairs = load_qa_pairs(args.qa_file)
    if not qa_pairs:
        raise SystemExit("No Q&A pairs found in file")

    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Test model
    results = test_model_on_questions(
        qa_pairs, max_questions=args.max_questions, delay=args.delay
    )

    if not results:
        raise SystemExit("No test results generated")

    # Save results
    save_results(results, args.out)

    # Summary
    total_time = sum(r["response_time"] for r in results)
    avg_time = total_time / len(results) if results else 0

    print("\n=== TEST SUMMARY ===")
    print(f"Questions tested: {len(results)}")
    print(f"Total response time: {total_time:.2f}s")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
