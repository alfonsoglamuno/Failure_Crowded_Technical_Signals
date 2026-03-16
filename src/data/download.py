"""
Download OHLCV data for EURO STOXX 50 constituents via yfinance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_ticker_list(path: str | Path) -> list[str]:
    with open(path) as f:
        return yaml.safe_load(f)["tickers"]


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str | None,
    output_dir: str | Path,
    adjusted: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for each ticker and save as parquet.

    Returns a dict {ticker: DataFrame}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        out_path = output_dir / f"{ticker.replace('/', '_')}.parquet"
        if out_path.exists():
            logger.info("Cache hit: %s", ticker)
            results[ticker] = pd.read_parquet(out_path)
            continue

        logger.info("Downloading: %s", ticker)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=adjusted,
                progress=False,
            )
            if df.empty:
                logger.warning("No data returned for %s", ticker)
                continue
            df.index = pd.to_datetime(df.index)
            df.to_parquet(out_path)
            results[ticker] = df
        except Exception as exc:
            logger.error("Failed to download %s: %s", ticker, exc)

    return results
