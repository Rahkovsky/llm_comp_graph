#!/usr/bin/env python3
"""Assess correctness of model answers against expected answers using LLM evaluation."""

import argparse
import json
import re
import subprocess
import os
from typing import Dict, List
from difflib import SequenceMatcher
import numpy as np

from constants import LLAMA_CLI, LLAMA_MODEL


def check_bins():
    """Check if required binaries exist."""
    if not os.path.exists(LLAMA_CLI):
        raise SystemExit(f"llama-cli not found at {LLAMA_CLI}. Set $LLAMA_CLI.")
    if not os.path.exists(LLAMA_MODEL):
        raise SystemExit(f"Model not found at {LLAMA_MODEL}. Set $LLAMA_MODEL.")


def run_llama(prompt: str, n: int = 128, temp: float = 0.1, ctx: int = 2048) -> str:
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


def evaluate_answer_with_llm(
    question: str, expected_answer: str, actual_answer: str
) -> Dict:
    """Use LLM to evaluate answer correctness."""
    prompt = f"""Evaluate if the actual answer is correct based on the expected answer.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Rate the correctness from 0-100 and provide a brief explanation.
Format your response as:
Score: [0-100]
Explanation: [brief explanation]

Response:"""

    response = run_llama(prompt, n=128, temp=0.1)

    if not response:
        return {"score": 0, "explanation": "LLM evaluation failed"}

    # Parse the response
    score = 0
    explanation = "No explanation provided"

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Score:"):
            try:
                score_text = line.split(":", 1)[1].strip()
                score = int(score_text)
                score = max(0, min(100, score))  # Clamp to 0-100
            except (ValueError, IndexError):
                score = 0
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return {"score": score, "explanation": explanation, "llm_evaluated": True}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using sequence matcher."""
    return SequenceMatcher(None, text1, text2).ratio()


def extract_key_facts(text: str) -> List[str]:
    """Extract key facts from text (dates, numbers, names, etc.)."""
    facts = []

    # Extract dates (various formats)
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # YYYY-MM-DD
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # MM-DD-YYYY
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",  # Month DD, YYYY
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, text, re.IGNORECASE)
        facts.extend(dates)

    # Extract numbers (including currency)
    number_patterns = [
        r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Currency
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",  # Numbers with commas/decimals
        r"\b\d+%\b",  # Percentages
    ]

    for pattern in number_patterns:
        numbers = re.findall(pattern, text)
        facts.extend(numbers)

    # Extract company names and tickers
    company_patterns = [
        r"\b[A-Z]{2,5}\b",  # Potential ticker symbols
        r"\b(?:Microsoft|MSFT|Apple|AAPL|Google|GOOGL|Amazon|AMZN)\b",  # Common companies
    ]

    for pattern in company_patterns:
        companies = re.findall(pattern, text, re.IGNORECASE)
        facts.extend(companies)

    return list(set(facts))  # Remove duplicates


def assess_answer_correctness(question: str, expected: str, actual: str) -> Dict:
    """Assess the correctness of an answer using LLM evaluation."""
    # Get LLM-based evaluation
    llm_eval = evaluate_answer_with_llm(question, expected, actual)

    # Also calculate traditional metrics for comparison
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    similarity = calculate_similarity(expected_norm, actual_norm)

    # Extract key facts
    expected_facts = extract_key_facts(expected)
    actual_facts = extract_key_facts(actual)

    # Calculate fact accuracy
    fact_accuracy = 0.0
    if expected_facts:
        correct_facts = sum(1 for fact in expected_facts if fact in actual_facts)
        fact_accuracy = correct_facts / len(expected_facts)

    # Determine grade based on LLM score
    score = llm_eval["score"]
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "llm_score": score,
        "llm_explanation": llm_eval["explanation"],
        "similarity": similarity,
        "fact_accuracy": fact_accuracy,
        "grade": grade,
        "expected_facts": expected_facts,
        "actual_facts": actual_facts,
        "correct_facts": [f for f in expected_facts if f in actual_facts],
        "missing_facts": [f for f in expected_facts if f not in actual_facts],
        "extra_facts": [f for f in actual_facts if f not in expected_facts],
    }


def analyze_results(test_results: List[Dict]) -> Dict:
    """Analyze overall test results."""
    if not test_results:
        return {}

    # Calculate metrics for each answer
    assessments = []
    for result in test_results:
        assessment = assess_answer_correctness(
            result["question"], result["expected_answer"], result["model_response"]
        )
        assessments.append(assessment)

    # Aggregate statistics
    similarities = [a["similarity"] for a in assessments]
    fact_accuracies = [
        a["fact_accuracy"] for a in assessments if a["fact_accuracy"] > 0
    ]
    llm_scores = [a["llm_score"] for a in assessments]

    grade_counts = {}
    for assessment in assessments:
        grade = assessment["grade"]
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    # Calculate response time statistics
    response_times = [r["response_time"] for r in test_results if "response_time" in r]

    return {
        "total_questions": len(test_results),
        "average_similarity": np.mean(similarities) if similarities else 0,
        "median_similarity": np.median(similarities) if similarities else 0,
        "average_fact_accuracy": np.mean(fact_accuracies) if fact_accuracies else 0,
        "average_llm_score": np.mean(llm_scores) if llm_scores else 0,
        "median_llm_score": np.median(llm_scores) if llm_scores else 0,
        "grade_distribution": grade_counts,
        "response_time_stats": {
            "average": np.mean(response_times) if response_times else 0,
            "median": np.median(response_times) if response_times else 0,
            "min": np.min(response_times) if response_times else 0,
            "max": np.max(response_times) if response_times else 0,
        },
        "detailed_assessments": assessments,
    }


def print_summary(analysis: Dict):
    """Print a summary of the analysis."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE ASSESSMENT")
    print("=" * 60)

    print("\n📊 OVERALL STATISTICS:")
    print(f"   Total Questions: {analysis['total_questions']}")
    print(f"   Average LLM Score: {analysis['average_llm_score']:.1f}/100")
    print(f"   Median LLM Score: {analysis['median_llm_score']:.1f}/100")
    print(f"   Average Similarity: {analysis['average_similarity']:.3f}")
    print(f"   Median Similarity: {analysis['median_similarity']:.3f}")
    print(f"   Average Fact Accuracy: {analysis['average_fact_accuracy']:.3f}")

    print("\n📈 GRADE DISTRIBUTION:")
    for grade in ["A", "B", "C", "D", "F"]:
        count = analysis["grade_distribution"].get(grade, 0)
        percentage = (count / analysis["total_questions"]) * 100
        print(f"   {grade}: {count} ({percentage:.1f}%)")

    print("\n⏱ RESPONSE TIME STATISTICS:")
    rt = analysis["response_time_stats"]
    print(f"   Average: {rt['average']:.2f}s")
    print(f"   Median: {rt['median']:.2f}s")
    print(f"   Range: {rt['min']:.2f}s - {rt['max']:.2f}s")

    # Show some examples
    print("\n🔍 SAMPLE ASSESSMENTS:")
    detailed = analysis["detailed_assessments"]
    for i, assessment in enumerate(detailed[:3]):  # Show first 3
        print(f"\n   Question {i + 1}:")
        print(f"   Grade: {assessment['grade']}")
        print(f"   LLM Score: {assessment['llm_score']}/100")
        print(f"   LLM Explanation: {assessment['llm_explanation'][:100]}...")
        print(f"   Similarity: {assessment['similarity']:.3f}")
        print(f"   Fact Accuracy: {assessment['fact_accuracy']:.3f}")
        if assessment["missing_facts"]:
            print(f"   Missing Facts: {', '.join(assessment['missing_facts'])}")


def save_analysis(analysis: Dict, output_file: str):
    """Save detailed analysis to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed analysis saved to: {output_file}")


def main():
    ap = argparse.ArgumentParser(description="Assess correctness of model answers")
    ap.add_argument(
        "--test-results", required=True, help="Path to test results JSON file"
    )
    ap.add_argument(
        "--out", default="out/answer_assessment.json", help="Output analysis file"
    )

    args = ap.parse_args()

    # Load test results
    try:
        with open(args.test_results, "r", encoding="utf-8") as f:
            test_results = json.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to load test results: {e}")

    print(f"Loaded {len(test_results)} test results from: {args.test_results}")

    # Analyze results
    analysis = analyze_results(test_results)

    if not analysis:
        raise SystemExit("No analysis results generated")

    # Print summary
    print_summary(analysis)

    # Save detailed analysis
    save_analysis(analysis, args.out)


if __name__ == "__main__":
    main()
