#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import time
from html import unescape
from html.parser import HTMLParser
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_TXT_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}.txt"

DEFAULT_UA = "Ilya Rahkovsky ilya.rahkovsky@gmail.com"

# ---------- HTML -> text (stdlib) ----------
class _HTMLToText(HTMLParser):
    """Minimal, deterministic HTML->text stripper (keeps line breaks around block tags)."""
    _BLOCK_TAGS = {
        "p","div","section","article","header","footer","nav","ul","ol","li",
        "table","thead","tbody","tfoot","tr","th","td","h1","h2","h3","h4","h5","h6","br","hr","pre"
    }
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out: List[str] = []
        self.last_was_nl = True
    def handle_starttag(self, tag, attrs):
        if tag in self._BLOCK_TAGS:
            self._newline()
    def handle_endtag(self, tag):
        if tag in self._BLOCK_TAGS:
            self._newline()
    def handle_data(self, data):
        if not data:
            return
        text = unescape(data)
        # collapse internal runs of whitespace but keep newlines
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        # write
        self.out.append(text)
        self.last_was_nl = text.endswith("\n")
    def handle_entityref(self, name):
        self.out.append(unescape(f"&{name};"))
    def handle_charref(self, name):
        self.out.append(unescape(f"&#{name};"))
    def _newline(self):
        if not self.last_was_nl:
            self.out.append("\n")
            self.last_was_nl = True
    def get_text(self):
        # join and normalize multiple blank lines
        txt = "".join(self.out)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        return txt

def html_to_text(s: str) -> str:
    p = _HTMLToText()
    p.feed(s)
    return p.get_text()

# ---------- Combined filing parsing ----------
_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
_TYPE_RE = re.compile(r"<TYPE>\s*([^\r\n<]+)", re.IGNORECASE)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)

def extract_primary_10q_text(combined: str) -> Optional[str]:
    """
    From a combined SEC filing (SGML), find the 10-Q (or 10-Q/A) document's TEXT block.
    Return html-stripped plain text, or None if not found.
    """
    for m in _DOC_RE.finditer(combined):
        block = m.group(1)
        mtype = _TYPE_RE.search(block)
        if not mtype:
            continue
        doctype = mtype.group(1).strip().upper()
        if doctype not in {"10-Q", "10-Q/A"}:
            continue
        mtext = _TEXT_RE.search(block)
        if not mtext:
            # some rare filings omit <TEXT>—fall back to the whole block
            raw = block
        else:
            raw = mtext.group(1)
        # Heuristic: if looks like HTML/XML, strip tags; otherwise just unescape
        if "<" in raw and ">" in raw:
            return html_to_text(raw)
        return unescape(raw)
    return None

# ---------- SEC HTTP ----------
def http_get(url: str, headers: Dict[str, str], max_retries: int = 5, base_sleep: float = 0.5) -> requests.Response:
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_sleep * attempt); continue
        r.raise_for_status()
    r.raise_for_status()
    return r

def load_ticker_map(headers: Dict[str, str]) -> Dict[str, str]:
    r = http_get(SEC_TICKERS_URL, headers)
    data = r.json()
    out: Dict[str, str] = {}
    for _, rec in data.items():
        out[str(rec["ticker"]).strip().upper()] = str(rec["cik_str"]).zfill(10)
    return out

def fetch_submissions(cik_padded: str, headers: Dict[str, str]) -> dict:
    url = SEC_SUBMISSIONS_URL.format(cik_padded=cik_padded)
    r = http_get(url, headers)
    return r.json()

def iter_filings(subs: dict, form_types: Iterable[str],
                 start_date: Optional[dt.date], end_date: Optional[dt.date]
                 ) -> Iterator[Tuple[str, dt.date, str]]:
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accessions = recent.get("accessionNumber", []) or []
    for form, fdate, acc in zip(forms, dates, accessions):
        if form not in form_types:
            continue
        try:
            d = dt.date.fromisoformat(fdate)
        except Exception:
            continue
        if start_date and d < start_date: continue
        if end_date and d > end_date: continue
        yield (acc, d, form)

def build_txt_url(cik_padded: str, accession: str) -> str:
    cik_int = str(int(cik_padded))
    acc_nd = accession.replace("-", "")
    return SEC_ARCHIVES_TXT_TMPL.format(cik_int=cik_int, accession_nodash=acc_nd, accession=accession)

def sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in s)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk download SEC 10-Q filings and (optionally) extract plain text.")
    p.add_argument("--tickers", nargs="*", help="List of tickers, e.g. AAPL MSFT GOOGL")
    p.add_argument("--tickers-file", type=str, help="File with one ticker per line")
    p.add_argument("--forms", nargs="*", default=["10-Q"], help="Form types (default: 10-Q)")
    p.add_argument("--start-date", type=str, help="YYYY-MM-DD inclusive")
    p.add_argument("--end-date", type=str, help="YYYY-MM-DD inclusive")
    p.add_argument("--outdir", type=str, default="sec_10q", help="Output root directory")
    p.add_argument("--user-agent", type=str, default=DEFAULT_UA, help="SEC User-Agent")
    p.add_argument("--sleep", type=float, default=0.25, help="Sleep between requests (sec)")
    p.add_argument("--max-per-ticker", type=int, default=0, help="Limit per ticker (0=no limit)")
    p.add_argument("--extract-text", action="store_true", help="Also write plain-text file from 10-Q document")
    return p.parse_args()

def to_date(s: Optional[str]) -> Optional[dt.date]:
    return dt.date.fromisoformat(s) if s else None

def load_tickers(args: argparse.Namespace) -> List[str]:
    vals: List[str] = []
    if args.tickers:
        vals.extend(t.upper() for t in args.tickers if t.strip())
    if args.tickers_file:
        with open(args.tickers_file, "r", encoding="utf-8") as f:
            vals.extend(line.strip().upper() for line in f if line.strip())
    seen, uniq = set(), []
    for t in vals:
        if t not in seen:
            seen.add(t); uniq.append(t)
    if not uniq:
        raise SystemExit("No tickers provided.")
    return uniq

def main():
    args = parse_args()
    start_date, end_date = to_date(args.start_date), to_date(args.end_date)
    os.makedirs(args.outdir, exist_ok=True)

    headers = {
        "User-Agent": args.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    print("Loading ticker map…")
    tmap = load_ticker_map(headers)
    time.sleep(args.sleep)

    tickers = load_tickers(args)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Forms: {', '.join(args.forms)}")
    if start_date or end_date:
        print(f"Date range: {start_date or '…'} → {end_date or '…'}")
    print(f"Output root: {args.outdir}")

    total = 0
    for t in tickers:
        cik = tmap.get(t)
        if not cik:
            print(f"[WARN] No CIK for {t}, skipping.")
            continue

        try:
            subs = fetch_submissions(cik, headers)
        except requests.HTTPError as e:
            print(f"[ERROR] Submissions fetch failed for {t} ({cik}): {e}")
            time.sleep(args.sleep)
            continue

        count = 0
        for accession, fdate, form in iter_filings(subs, args.forms, start_date, end_date):
            url = build_txt_url(cik, accession)
            year = fdate.year
            out_dir_ticker = os.path.join(args.outdir, t, str(year))
            os.makedirs(out_dir_ticker, exist_ok=True)
            base = sanitize(f"{t}_{cik}_{fdate.isoformat()}_{accession}")
            raw_path = os.path.join(out_dir_ticker, base + ".txt")
            plain_path = os.path.join(out_dir_ticker, base + ".plain.txt")

            if os.path.exists(raw_path):
                print(f"[SKIP] Exists: {raw_path}")
            else:
                try:
                    r = http_get(url, headers)
                    with open(raw_path, "wb") as f:
                        f.write(r.content)
                    print(f"[OK] {t} {form} {fdate} {accession} -> {raw_path}")
                    total += 1
                except requests.HTTPError as e:
                    print(f"[ERROR] Download failed {t} {fdate} {accession}: {e}")
                    time.sleep(args.sleep)
                    continue

            if args.extract_text:
                try:
                    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                        combined = f.read()
                    text = extract_primary_10q_text(combined)
                    if text:
                        with open(plain_path, "w", encoding="utf-8") as f:
                            f.write(text + "\n")
                        print(f"[TXT ] Wrote plain text: {plain_path}")
                    else:
                        print(f"[WARN] No 10-Q TEXT block found in: {raw_path}")
                except Exception as e:
                    print(f"[ERROR] Text extraction failed for {raw_path}: {e}")

            time.sleep(args.sleep)
            count += 1
            if args.max_per_ticker and count >= args.max_per_ticker:
                break

        time.sleep(args.sleep)

    print(f"Done. Downloaded {total} filings under: {args.outdir}")

if __name__ == "__main__":
    main()
