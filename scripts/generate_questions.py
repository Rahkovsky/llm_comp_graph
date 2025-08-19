#!/usr/bin/env python3
import argparse, glob, json, os, re, subprocess, textwrap, time
from typing import List

LLAMA_CLI = os.environ.get("LLAMA_CLI", os.path.expanduser("~/llama.cpp/build/bin/llama-cli"))
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", os.path.expanduser("~/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"))

def check_bins():
    if not os.path.isfile(LLAMA_CLI):
        raise SystemExit(f"llama-cli not found at {LLAMA_CLI}. Set $LLAMA_CLI.")
    if not os.path.isfile(LLAMA_MODEL):
        raise SystemExit(f"Model not found at {LLAMA_MODEL}. Set $LLAMA_MODEL.")

def chat_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system +
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + user +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

def run_llama(prompt: str, n: int = 512, temp: float = 0.2, ctx: int = 4096) -> str:
    cmd = [
        LLAMA_CLI, "-m", LLAMA_MODEL, "-n", str(n), "-ngl", "999",
        "-c", str(ctx), "--repeat_penalty", "1.1", "--temp", str(temp),
        "-p", prompt
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout

def chunk(text: str, max_chars: int = 6000, overlap: int = 500) -> List[str]:
    blocks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        blocks.append(text[i:j])
        i = j - overlap
        if i < 0: i = 0
    return blocks

def generate_questions_on_chunk(chunk_text: str, n_q: int = 6) -> List[str]:
    sys = "You generate factual, document-grounded questions. Avoid trivia; ask questions that the provided text can answer."
    usr = f"""From the text below, write {n_q} distinct, concise questions whose answers are directly supported by the text.
Return them as a bullet list, one per line, no numbering, no extra commentary.

TEXT:
{chunk_text[:6000]}
"""
    prompt = chat_prompt(sys, usr)
    raw = run_llama(prompt, n=384, temp=0.2)
    qs = []
    for line in raw.splitlines():
        line = re.sub(r"^[\-\*\u2022]\s*", "", line).strip()
        if len(line) > 5 and "assistant" not in line.lower():
            qs.append(line)
    # de-dup shallowly
    dedup = []
    seen = set()
    for q in qs:
        k = re.sub(r"[^\w]+", " ", q.lower()).strip()
        if k and k not in seen:
            seen.add(k); dedup.append(q)
    return dedup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/10k/AAPL/*/*.plain.txt", help="Glob for plain text 10-K files")
    ap.add_argument("--out", default="out/questions/questions_10k_aapl.txt", help="Where to save questions")
    ap.add_argument("--per-chunk", type=int, default=2, help="Questions per chunk")
    args = ap.parse_args()

    check_bins()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_qs: List[str] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        for ch in chunk(txt, max_chars=6000, overlap=800):
            qs = generate_questions_on_chunk(ch, n_q=args.per_chunk)
            all_qs.extend(qs)
            time.sleep(0.1)

    # final dedup
    seen = set(); final = []
    for q in all_qs:
        k = re.sub(r"[^\w]+", " ", q.lower()).strip()
        if k and k not in seen:
            seen.add(k); final.append(q)

    with open(args.out, "w", encoding="utf-8") as f:
        for q in final:
            f.write(q.rstrip() + "\n")

    print(f"Wrote {len(final)} questions -> {args.out}")

if __name__ == "__main__":
    main()
