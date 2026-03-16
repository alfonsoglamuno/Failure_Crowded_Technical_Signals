"""
Strategy simulation comparing three approaches:
  1. Follow-the-alert    — trade in the signal's direction
  2. Blind inverse        — always fade the alert
  3. ML-filtered inverse  — fade only when model confidence >= threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _direction_sign(direction: str) -> int:
    return {"bullish": 1, "bearish": -1, "neutral": 0}.get(direction, 0)


def simulate_follow_alert(events_with_returns: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Returns a Series of trade-level returns for the follow-the-alert strategy.
    A bullish alert → long; bearish alert → short.
    """
    col = f"fwd_ret_{horizon}d"
    trades = events_with_returns.copy()
    trades["sign"] = trades["direction"].map(_direction_sign)
    # Neutral alerts are excluded
    trades = trades[trades["sign"] != 0]
    return (trades[col] * trades["sign"]).dropna().rename(f"follow_alert_h{horizon}")


def simulate_blind_inverse(events_with_returns: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Returns a Series of trade-level returns for always-fade strategy.
    A bullish alert → short; bearish alert → long.
    """
    col = f"fwd_ret_{horizon}d"
    trades = events_with_returns.copy()
    trades["sign"] = -trades["direction"].map(_direction_sign)
    trades = trades[trades["sign"] != 0]
    return (trades[col] * trades["sign"]).dropna().rename(f"blind_inverse_h{horizon}")


def simulate_ml_filtered(
    events_with_returns: pd.DataFrame,
    horizon: int,
    confidence_threshold: float = 0.6,
    proba_col: str = "failure_proba",
) -> pd.Series:
    """
    Returns a Series of trade-level returns for ML-filtered contrarian strategy.
    Only fades the alert when predicted failure probability >= threshold.
    """
    col = f"fwd_ret_{horizon}d"
    trades = events_with_returns.copy()

    if proba_col not in trades.columns:
        raise ValueError(f"Column '{proba_col}' not found. Run model prediction first.")

    # Only take contrarian trades with high confidence
    trades = trades[trades[proba_col] >= confidence_threshold]
    trades["sign"] = -trades["direction"].map(_direction_sign)
    trades = trades[trades["sign"] != 0]
    return (trades[col] * trades["sign"]).dropna().rename(f"ml_filtered_h{horizon}")


def compare_strategies(
    events_with_returns: pd.DataFrame,
    horizon: int,
    confidence_threshold: float = 0.6,
    proba_col: str = "failure_proba",
) -> pd.DataFrame:
    """
    Returns a DataFrame with trade returns for all three strategies side by side.
    """
    s1 = simulate_follow_alert(events_with_returns, horizon).reset_index(drop=True)
    s2 = simulate_blind_inverse(events_with_returns, horizon).reset_index(drop=True)
    try:
        s3 = simulate_ml_filtered(
            events_with_returns, horizon, confidence_threshold, proba_col
        ).reset_index(drop=True)
    except ValueError:
        s3 = pd.Series(dtype=float, name=f"ml_filtered_h{horizon}")

    return pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3})
