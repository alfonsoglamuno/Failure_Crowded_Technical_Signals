"""
Alert detection for the trading agent.
Thin wrapper around the research project's alert engine, adapted for
real-time single-ticker use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow importing from parent research project
_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from src.alerts.engine import ALERT_REGISTRY


def detect_alerts(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Run all alerts on a single-ticker OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns [date, open, high, low, close, volume],
               sorted ascending. Should contain at least 200 rows.
        ticker: Yahoo Finance ticker string.

    Returns:
        DataFrame of today's alerts: [date, ticker, alert_name, direction]
    """
    df = ohlcv.copy().sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    data = {col: df[col] for col in ["open", "high", "low", "close", "volume"]}
    today = df.index[-1]

    events = []
    for alert_name, (direction, fn) in ALERT_REGISTRY.items():
        try:
            mask: pd.Series = fn(data)
            if mask.iloc[-1]:   # only today's bar
                events.append({
                    "date": today,
                    "ticker": ticker,
                    "alert_name": alert_name,
                    "direction": direction,
                })
        except Exception:
            continue

    return pd.DataFrame(events)


def detect_universe_alerts(universe_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run alert detection across all tickers.

    Args:
        universe_data: {yahoo_ticker: ohlcv_df}

    Returns:
        Combined events DataFrame.
    """
    all_events = []
    for ticker, ohlcv in universe_data.items():
        if len(ohlcv) < 50:
            continue
        events = detect_alerts(ohlcv, ticker)
        if not events.empty:
            all_events.append(events)

    if not all_events:
        return pd.DataFrame(columns=["date", "ticker", "alert_name", "direction"])

    combined = pd.concat(all_events, ignore_index=True)

    # Count simultaneous alerts per (date, ticker)
    count = (
        combined.groupby(["date", "ticker"])["alert_name"]
        .count()
        .rename("n_simultaneous_alerts")
        .reset_index()
    )
    combined = combined.merge(count, on=["date", "ticker"], how="left")
    return combined
