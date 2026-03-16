"""
Label construction for alert-failure prediction.

For each alert event on day t:
  - bullish alert failure  →  r(t+1 : t+h) < -theta
  - bearish alert failure  →  r(t+1 : t+h) > +theta
  - neutral alert: uses absolute reversal in either direction
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(
    panel: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """
    Add forward return columns to the panel.
    forward_ret_{h}d = close(t+h) / close(t) - 1
    """
    panel = panel.copy()
    for ticker, grp in panel.groupby("ticker"):
        idx = grp.index
        for h in horizons:
            fwd = grp["close"].shift(-h) / grp["close"] - 1
            panel.loc[idx, f"fwd_ret_{h}d"] = fwd
    return panel


def assign_labels(
    events: pd.DataFrame,
    panel_with_fwd: pd.DataFrame,
    horizons: list[int],
    theta: float = 0.005,
) -> pd.DataFrame:
    """
    Merge forward returns onto events and compute binary failure labels.

    Label = 1  →  alert failed (contrarian opportunity)
    Label = 0  →  alert worked (continuation)
    """
    fwd_cols = [f"fwd_ret_{h}d" for h in horizons]
    lookup = panel_with_fwd[["date", "ticker"] + fwd_cols].copy()

    events = events.merge(lookup, on=["date", "ticker"], how="left")

    for h in horizons:
        col = f"fwd_ret_{h}d"
        label_col = f"label_failure_{h}d"
        events[label_col] = np.nan

        bull_mask = events["direction"] == "bullish"
        bear_mask = events["direction"] == "bearish"
        neut_mask = events["direction"] == "neutral"

        # Bullish alert fails if price reverses down
        events.loc[bull_mask, label_col] = (
            events.loc[bull_mask, col] < -theta
        ).astype(float)

        # Bearish alert fails if price reverses up
        events.loc[bear_mask, label_col] = (
            events.loc[bear_mask, col] > theta
        ).astype(float)

        # Neutral alert: fails if magnitude of reversal exceeds theta
        events.loc[neut_mask, label_col] = (
            events.loc[neut_mask, col].abs() > theta
        ).astype(float)

    return events
