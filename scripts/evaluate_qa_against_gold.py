#!/usr/bin/env python3
"""Run model answers for questions and evaluate against gold answers.

Pipeline:
- Load Q/A pairs from a gold file (format: lines with `Q:` and `A:`).
- For each question, get model's answer using local llama.
- Score each model answer against gold using: LLM judge + lexical overlap + fact overlap.
- Save per-question scores and an overall summary.
"""

import argparse
import glob
import math
import json
import os
import re
import subprocess
import time
from typing import List, Tuple, Dict, Any, Iterable, Pattern, cast

from llm_comp_graph.constants import LLAMA_CLI, LLAMA_MODEL, OUTDIR_10K


def check_bins():
    if not os.path.exists(LLAMA_CLI):
        raise SystemExit(f"llama-cli not found at {LLAMA_CLI}.")
    if not os.path.exists(LLAMA_MODEL):
        raise SystemExit(f"Model not found at {LLAMA_MODEL}.")


def run_llama(prompt: str, n: int = 128, temp: float = 0.1, ctx: int = 2048) -> str:
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
        return ""


def run_llama_embedding(text: str, ctx: int = 2048) -> List[float]:
    """Get embedding vector from llama-cli. Returns [] on failure."""
    cmd = [
        LLAMA_CLI,
        "-m",
        LLAMA_MODEL,
        "--embedding",
        "-c",
        str(ctx),
        "-p",
        text,
        "--no-conversation",
        "--no-warmup",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
        s = out.stdout.strip()
        # Try to parse floats from the last bracketed list
        start = s.rfind("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = s[start + 1 : end].split(",")
            vec: List[float] = []
            for x in arr:
                try:
                    vec.append(float(x.strip()))
                except Exception:
                    continue
            return vec
    except Exception:
        return []
    return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def load_gold_qa(path: str) -> List[Tuple[str, str]]:
    qa_pairs: List[Tuple[str, str]] = []
    q = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q: "):
                candidate_q = line[3:].strip()
                # Skip prompt/template artifacts
                lower_q = candidate_q.lower()
                if (
                    candidate_q.startswith("[")
                    or candidate_q.startswith("<")
                    or "rewritten or original" in lower_q
                    or "clear, specific question" in lower_q
                ):
                    q = None
                else:
                    q = candidate_q
            elif line.startswith("A: ") and q:
                a = line[3:].strip()
                # Clean common artifacts from gold answers
                a = re.sub(r"\(Note:.*?\)", "", a, flags=re.IGNORECASE)
                a = re.sub(r"\s+user$", "", a, flags=re.IGNORECASE)
                a = re.sub(r"\s+", " ", a).strip()
                if q and a:
                    qa_pairs.append((q, a))
                q = None
    return qa_pairs


def load_provenance(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
            if isinstance(data, list):
                data_list = cast(List[Dict[str, Any]], data)
                out: List[Dict[str, str]] = []
                for item in data_list:
                    tmp: Dict[str, str] = {}
                    for k_raw, v_raw in item.items():
                        k_s: str = str(k_raw)
                        v_s: str = str(v_raw)
                        tmp[k_s] = v_s
                    out.append(tmp)
                return out
    except Exception:
        pass
    return []


def question_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def placebo_prompt(context: str, question: str) -> str:
    system = (
        "You must answer ONLY using the provided Context. Copy the minimal text span that answers the question. "
        "Do not paraphrase. If the answer is not present in Context, output exactly: Not found."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return f"system\n{system}\nuser\n{user}\nassistant\n"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_key_facts(text: str) -> List[str]:
    facts: List[str] = []
    # Numbers, percentages, currency, and years
    patterns = [
        r"\b\d{4}\b",  # year
        r"\b\d+%\b",  # percentage
        r"\$\d+(?:,\d{3})*(?:\.\d+)?",  # currency
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",  # general numbers
    ]
    for p in patterns:
        facts.extend(re.findall(p, text))

    # Acronyms (2-6 uppercase letters)
    facts.extend(re.findall(r"\b[A-Z]{2,6}\b", text))

    # Capitalized name phrases (2-5 tokens)
    name_phrases = re.findall(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b",
        text,
    )
    stop_phrases = {
        "United States",
        "Securities Exchange",
        "Annual Report",
        "Form 10",
        "Item Management",
    }
    for np in name_phrases:
        if np not in stop_phrases:
            facts.append(np)

    return sorted(set(facts))


def _read_text(fp: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            with open(fp, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def _iter_source_files(tickers: List[str] | None) -> Iterable[str]:
    if tickers:
        for t in sorted(set([t.upper() for t in tickers if t])):
            for pat in (
                f"{OUTDIR_10K}/{t}/*.txt",
                f"{OUTDIR_10K}/{t}/*/*.txt",
                f"{OUTDIR_10K}/{t}/*.plain.txt",
                f"{OUTDIR_10K}/{t}/*/*.plain.txt",
            ):
                for fp in glob.glob(pat):
                    yield fp
    else:
        for fp in glob.glob(f"{OUTDIR_10K}/*/*.txt") + glob.glob(
            f"{OUTDIR_10K}/*/*/*.txt"
        ):
            yield fp


def find_context_for_qa(
    question: str, gold: str, tickers: List[str] | None, ctx_chars: int
) -> str | None:
    """Find approximate context from source files using a fuzzy word-sequence match."""

    def build_patterns(text: str) -> List[Pattern[str]]:
        words = re.findall(r"\w+", text)
        words = [w for w in words if len(w) >= 2]
        if not words:
            return []
        # Try mid-slices with decreasing window sizes
        patterns: List[Pattern[str]] = []
        for win in (10, 8, 6, 5, 4):
            if len(words) < win:
                continue
            mid = len(words) // 2
            start = max(0, mid - win // 2)
            end = min(len(words), start + win)
            seq = words[start:end]
            pat = r"\b" + r"\W+".join(re.escape(w) for w in seq) + r"\b"
            try:
                patterns.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                continue
        # Fallback: first 5 words
        if len(words) >= 5:
            seq = words[:5]
            pat = r"\b" + r"\W+".join(re.escape(w) for w in seq) + r"\b"
            try:
                patterns.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                pass
        return patterns

    # Prefer patterns from gold; if too short, from question
    base_text = gold if len(gold) >= 20 else question
    base_text = re.sub(r"\s+", " ", base_text).strip()
    patterns: List[Pattern[str]] = build_patterns(base_text)
    if not patterns and base_text is not gold:
        patterns = build_patterns(gold)
    if not patterns:
        return None

    for fp in _iter_source_files(tickers):
        txt = _read_text(fp)
        if not txt:
            continue
        for pat in patterns:
            m = pat.search(txt)
            if m:
                lo = max(0, m.start() - ctx_chars // 2)
                hi = min(len(txt), m.end() + ctx_chars // 2)
                return txt[lo:hi]
    return None


def llm_judge(question: str, gold: str, pred: str) -> Dict[str, Any]:
    prompt = (
        "You are a strict evaluator. Compare the model answer to the gold answer.\n"
        "Rules:\n- Penalize generic or evasive answers.\n- Reward correct specific entities (names, metrics, dates).\n"
        'Respond with JSON ONLY, exactly one object on a single line: {"score": <0-100>, "explanation": "<= 60 words"}. No extra text.\n\n'
        f"Question: {question}\nGold: {gold}\nModel: {pred}\n\nJSON ONLY:"
    )

    def _extract_json_object(text: str) -> Dict[str, Any] | None:
        text = text.strip()
        # Fast path
        try:
            return json.loads(text)  # type: ignore[arg-type]
        except Exception:
            pass
        # Try substring between first '{' and last '}'
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                cand = text[start : end + 1]
                return json.loads(cand)
        except Exception:
            pass
        return None

    out = run_llama(prompt, n=160, temp=0.0)
    data = _extract_json_object(out or "")
    if data:
        try:
            score = int(data.get("score", 0))
        except Exception:
            score = 0
        expl = str(data.get("explanation", ""))
        return {"score": max(0, min(100, score)), "explanation": expl}
    return {"score": 0, "explanation": "judge parse failure"}


def score_answer(
    question: str,
    gold: str,
    pred: str,
    use_embeddings: bool = False,
    emb_weight: float = 0.6,
) -> Dict[str, Any]:
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(pred)

    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())
    overlap = len(gold_tokens & pred_tokens) / max(1, len(gold_tokens))

    gold_facts = extract_key_facts(gold)
    pred_facts = extract_key_facts(pred)
    fact_acc = (
        sum(1 for f in gold_facts if f in pred_facts) / len(gold_facts)
        if gold_facts
        else 0.0
    )

    exact_match = gold_norm == pred_norm
    if exact_match:
        judged = {"score": 100, "explanation": "exact match"}
    else:
        judged = llm_judge(question, gold, pred)
        if (
            judged.get("explanation") == "judge parse failure"
            or int(judged.get("score", 0)) == 0
        ):
            # Fallback numeric score: blend semantic (if enabled) with lexical
            sem = 0.0
            if use_embeddings:
                g_trunc = gold[:512]
                p_trunc = pred[:512]
                gv = run_llama_embedding(g_trunc)
                pv = run_llama_embedding(p_trunc)
                sem = cosine_similarity(gv, pv)
            blended = (emb_weight * sem) + (
                (1.0 - emb_weight) * (0.7 * overlap + 0.3 * fact_acc)
            )
            fallback = int(round(100.0 * blended))
            judged = {"score": fallback, "explanation": "fallback semantic+lexical"}

    return {
        "overlap": overlap,
        "fact_accuracy": fact_acc,
        "llm_score": judged["score"],
        "llm_explanation": judged["explanation"],
        "gold_facts": gold_facts,
        "pred_facts": pred_facts,
        "exact_match": exact_match,
    }


def run_and_evaluate(
    qa_pairs: List[Tuple[str, str]],
    limit: int | None,
    delay: float,
    progress: bool,
    echo_gold: bool,
    placebo: bool,
    placebo_ctx_chars: int,
    placebo_tickers: List[str] | None,
    prov_map: Dict[str, Dict[str, str]] | None,
    use_embeddings: bool,
    emb_weight: float,
) -> List[Dict[str, Any]]:
    if limit is not None:
        qa_pairs = qa_pairs[:limit]

    total = len(qa_pairs)
    results: List[Dict[str, Any]] = []
    for i, (q, gold) in enumerate(qa_pairs, 1):
        if progress:
            head = (q[:80] + "…") if len(q) > 80 else q
            print(f"[{i}/{total}] Asking: {head}", flush=True)
        start = time.time()
        if echo_gold:
            pred = gold
        elif placebo:
            # Prefer provenance if available
            key = f"{q}\n{gold}"
            ctx = None
            if prov_map is not None:
                prov = prov_map.get(key)
                if prov and prov.get("chunk"):
                    ctx = prov.get("chunk")
            if not ctx:
                ctx = find_context_for_qa(q, gold, placebo_tickers, placebo_ctx_chars)
            if ctx:
                pred = run_llama(placebo_prompt(ctx, q), n=128, temp=0.0)
            else:
                pred = "Not found"
        else:
            pred = run_llama(question_prompt(q), n=160, temp=0.1)
        elapsed = time.time() - start
        pred = pred.strip()

        scores = score_answer(
            q, gold, pred, use_embeddings=use_embeddings, emb_weight=emb_weight
        )
        results.append(
            {
                "idx": i,
                "question": q,
                "gold": gold,
                "pred": pred,
                "elapsed_sec": elapsed,
                **scores,
            }
        )
        if progress:
            print(
                f"    ↳ llm_score={scores['llm_score']} overlap={scores['overlap']:.2f} facts={scores['fact_accuracy']:.2f} time={elapsed:.2f}s",
                flush=True,
            )
        time.sleep(delay)

    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    def avg(k: str) -> float:
        return float(sum(float(r[k]) for r in results) / len(results))

    return {
        "count": len(results),
        "avg_overlap": avg("overlap"),
        "avg_fact_accuracy": avg("fact_accuracy"),
        "avg_llm_score": avg("llm_score"),
        "min_llm_score": min(r["llm_score"] for r in results),
        "max_llm_score": max(r["llm_score"] for r in results),
        "exact_match_rate": float(
            sum(1 for r in results if r.get("exact_match")) / len(results)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate model answers vs. gold Q/A file")
    ap.add_argument("--qa-file", required=True, help="Path to gold Q/A file")
    ap.add_argument("--out", default="out/eval_results.json", help="Output JSON file")
    ap.add_argument("--limit", type=int, default=50, help="Limit number of questions")
    ap.add_argument("--delay", type=float, default=0.4, help="Delay between questions")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-question progress output",
    )
    ap.add_argument(
        "--echo-gold",
        action="store_true",
        help="Do not call the model; use the gold answer as the predicted answer (sanity check)",
    )
    ap.add_argument(
        "--placebo",
        action="store_true",
        help="Answer with exact chunk context when found (upper-bound stability test)",
    )
    ap.add_argument(
        "--placebo-ctx",
        type=int,
        default=600,
        help="Context window chars around the found answer (per match)",
    )
    ap.add_argument(
        "--placebo-tickers",
        type=str,
        default="",
        help="Comma-separated tickers to search for context (optional)",
    )
    ap.add_argument(
        "--provenance",
        type=str,
        default="",
        help="Optional path to provenance JSON saved by generator (preferred for placebo)",
    )
    ap.add_argument(
        "--use-emb",
        action="store_true",
        help="Use local llama embeddings for semantic fallback",
    )
    ap.add_argument(
        "--emb-weight",
        type=float,
        default=0.6,
        help="Weight of semantic similarity in fallback score [0-1]",
    )
    args = ap.parse_args()

    check_bins()

    qa_pairs = load_gold_qa(args.qa_file)
    if not qa_pairs:
        raise SystemExit("No Q/A pairs loaded from gold file")

    tickers = (
        [t.strip().upper() for t in args.placebo_tickers.split(",") if t.strip()]
        if args.placebo_tickers
        else None
    )
    prov_map: Dict[str, Dict[str, str]] = {}
    if args.provenance:
        prov_list = load_provenance(args.provenance)
        for item in prov_list:
            key = f"{item.get('question','')}\n{item.get('answer','')}"
            prov_map[key] = item

    results = run_and_evaluate(
        qa_pairs,
        args.limit,
        args.delay,
        progress=not args.no_progress,
        echo_gold=bool(args.echo_gold),
        placebo=bool(args.placebo),
        placebo_ctx_chars=int(args.placebo_ctx),
        placebo_tickers=tickers,
        prov_map=prov_map if prov_map else None,
        use_embeddings=bool(args.use_emb),
        emb_weight=float(args.emb_weight),
    )

    summary = summarize(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
