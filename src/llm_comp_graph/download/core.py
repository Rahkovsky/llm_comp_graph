#!/usr/bin/env python3
"""Streamlined SEC filing downloads using edgartools."""

import os
import re
from edgar import Company, set_identity

from llm_comp_graph.download.constants import DEFAULT_UA, OUTPUT_BASE, FORM_TYPES
from llm_comp_graph.utils.logging_config import get_logger
from llm_comp_graph.utils.env_config import get_sec_user_info


from typing import Any, List, Optional, Tuple, cast


class SECExtractor:
    """Downloads and extracts 10-K/20-F filings."""

    def __init__(self, user_agent: str = DEFAULT_UA, output_dir: str = OUTPUT_BASE):
        self.output_dir = output_dir
        self.logger = get_logger(__name__)

        if "@" in user_agent:
            email = user_agent.split()[-1]
        else:
            _, email = get_sec_user_info()
        set_identity(email)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean up text formatting."""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _get_filing(
        self,
        ticker: str,
        forms: Optional[List[str]] = None,
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[object | None, str | None]:
        """Get filing, optionally filtered by forms, year, and date range."""
        co = Company(ticker)
        if not forms:
            forms = FORM_TYPES

        # Try each form type
        for form in forms:
            filings = cast(Any, co).get_filings(form=form)
            if filings:
                if year or start_date or end_date:
                    for filing in filings:  # Filter by date criteria
                        if getattr(filing, "filing_date", None):
                            if (
                                year and filing.filing_date.year != year
                            ):  # Check year filter
                                continue
                            if start_date:  # Check date range filters
                                from datetime import datetime

                                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                                if filing.filing_date < start:
                                    continue
                            if end_date:
                                from datetime import datetime

                                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                                if filing.filing_date > end:
                                    continue

                            return filing, form
                else:
                    return filings.latest(), form

        return None, None

    def download_and_extract(
        self,
        ticker: str,
        forms: Optional[List[str]] = None,
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Download and extract filing. Returns 1 for success, 0 for failure."""
        self.logger.info(f"Processing {ticker}")
        try:  # Get filing info (external API call - can fail)
            filing, form_type = self._get_filing(
                ticker, forms, year, start_date, end_date
            )
        except Exception as e:  # edgartools/network errors
            self.logger.error(f"Failed to fetch filing data for {ticker}: {e}")
            return 0

        if not filing:
            self.logger.warning(f"No 10-K or 20-F filings found for {ticker}")
            return 0

        # Check if file already exists
        ticker_dir = os.path.join(self.output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        accession_no = getattr(filing, "accession_no", None)
        if not accession_no:
            self.logger.warning(f"Missing accession number for {ticker}")
            return 0
        filename = f"{ticker}_{accession_no}.txt"
        filepath = os.path.join(ticker_dir, filename)

        if os.path.exists(filepath):
            self.logger.info(f"File already exists: {filename}")
            return 1

        # Extract text (external API call - can fail)
        self.logger.info(f"Extracting {form_type} for {ticker}")
        try:
            text_attr = getattr(filing, "text", None)
            if not callable(text_attr):
                self.logger.error(f"Filing.text() not available for {ticker}")
                return 0
            raw_text = text_attr()
            if not isinstance(raw_text, str):
                self.logger.error(f"Filing.text() returned non-string for {ticker}")
                return 0
            text = self._normalize_text(raw_text)
        except Exception as e:  # external I/O/parsing errors
            self.logger.error(f"Failed to extract text for {ticker}: {e}")
            return 0

        if len(text) < 1000:
            self.logger.warning(
                f"Insufficient content for {ticker} ({len(text)} chars)"
            )
            return 0

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        self.logger.info(f"Saved: {filename} ({len(text):,} chars)")
        return 1


def download_and_extract_10k(
    ticker: str,
    user_agent: str = DEFAULT_UA,
    output_dir: str = OUTPUT_BASE,
    forms: Optional[List[str]] = None,
    year: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1,
) -> int:
    """Download 10-K filing for a ticker."""
    extractor = SECExtractor(user_agent, output_dir)
    return extractor.download_and_extract(ticker, forms, year, start_date, end_date)
