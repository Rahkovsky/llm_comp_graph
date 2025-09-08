#!/usr/bin/env python3
"""Download 10-K filings for specified companies."""

import argparse
import sys
import time
from typing import List, Dict, Any
import requests

from llm_comp_graph.download.constants import DEFAULT_UA, FORM_TYPES, OUTPUT_BASE
from llm_comp_graph.download.core import download_and_extract_10k
from llm_comp_graph.utils.logging_config import setup_logging, get_logger


def get_all_companies() -> List[str]:
    """Get all public company tickers from SEC."""
    logger = get_logger(__name__)

    try:
        logger.info("Fetching company tickers from SEC")
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": DEFAULT_UA},
        )
        response.raise_for_status()
        companies_json: Dict[str, Any] = response.json()
        if not isinstance(companies_json, dict):
            logger.error("Unexpected tickers payload type from SEC")
            return ["AAPL", "MSFT"]

        tickers: List[str] = []
        for value in companies_json.values():
            if isinstance(value, dict):
                ticker_value = value.get("ticker")
                if isinstance(ticker_value, str):
                    t = ticker_value.strip()
                    if t:
                        tickers.append(t)

        logger.info(f"Found {len(tickers)} public companies")
        return tickers

    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        logger.warning("Falling back to sample companies")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Download 10-K filings")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", nargs="+", help="Ticker symbols")
    group.add_argument("--tickers-file", help="File with tickers (one per line)")
    group.add_argument(
        "--all-companies", action="store_true", help="All public companies"
    )
    parser.add_argument(
        "--forms",
        nargs="*",
        default=FORM_TYPES,
        help=f"Form types (default: {', '.join(FORM_TYPES)})",
    )
    parser.add_argument("--start-date", help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date filter (YYYY-MM-DD)")
    parser.add_argument(
        "--year", type=int, help="Specific year to download (e.g., 2024)"
    )
    parser.add_argument("--output-dir", default=OUTPUT_BASE, help="Output directory")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="User agent")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay between requests (default: 0.25)",
    )
    parser.add_argument(
        "--max-per-ticker", type=int, default=1, help="Max filings per ticker"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    setup_logging(
        verbose=args.verbose, module_name="sec_download"
    )  # Initialize logging
    logger = get_logger(__name__)

    # Get companies to process
    companies: List[str] = []
    if args.all_companies:
        companies = get_all_companies()
    elif args.tickers:
        companies = args.tickers
    elif args.tickers_file:
        with open(args.tickers_file, "r") as f:
            companies = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(companies)} companies")
    logger.info(f"Output directory: {args.output_dir}")

    # Process companies
    successful = failed = 0

    for i, ticker in enumerate(companies, 1):
        logger.info(f"[{i}/{len(companies)}] Processing {ticker}")

        result = download_and_extract_10k(
            ticker,
            args.user_agent,
            args.output_dir,
            args.forms,
            args.year,
            args.start_date,
            args.end_date,
        )
        if result > 0:
            successful += 1
            logger.info(f"Success: {ticker}")
        else:
            failed += 1
            logger.warning(f"Failed: {ticker}")

        # Sleep between requests; skip after the last company
        if i < len(companies):
            time.sleep(args.sleep)

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Total: {len(companies)}")
    logger.info(f"Success: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {(successful / len(companies) * 100):.1f}%")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
