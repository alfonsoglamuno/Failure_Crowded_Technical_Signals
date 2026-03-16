"""
Clean and align raw OHLCV data across the EURO STOXX 50 universe.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def clean_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Basic cleaning for a single ticker DataFrame."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    df = df.copy()
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    # Remove rows where Close is zero or NaN
    df = df[(df["Close"] > 0) & df["Close"].notna()]

    # Forward-fill volume gaps (weekends already dropped; handles split-adjust issues)
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill()

    return df


def build_panel(
    raw_data: dict[str, pd.DataFrame],
    min_history_days: int = 252,
) -> pd.DataFrame:
    """
    Combine all tickers into a long-format panel DataFrame.

    Returns columns: date, ticker, open, high, low, close, volume
    """
    frames = []
    for ticker, df in raw_data.items():
        try:
            df = clean_single(df, ticker)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            continue

        if len(df) < min_history_days:
            logger.warning(
                "Skipping %s: only %d rows (min %d)", ticker, len(df), min_history_days
            )
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df["ticker"] = ticker
        df.index.name = "date"
        frames.append(df.reset_index())

    if not frames:
        raise RuntimeError("No valid data after cleaning.")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel.sort_values(["ticker", "date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel
