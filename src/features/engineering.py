"""
Feature engineering: builds the full feature matrix from panel data + alert events.

Feature blocks:
  A. Alert features
  B. Short-term price state
  C. Volatility state
  D. Volume and crowding state
  E. Regime features
  F. Calendar features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alerts.engine import _rsi, _ema, _atr, _volume_zscore


# ---------------------------------------------------------------------------
# Block B — short-term price state
# ---------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"ret_{w}d"] = df["close"].pct_change(w)
    df["ret_1d_lead"] = df["close"].pct_change(1).shift(-1)  # not used in features; useful for labels
    return df


def add_ma_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in [10, 20, 50, 100, 200]:
        ma = df["close"].rolling(w).mean()
        df[f"dist_ma{w}"] = (df["close"] - ma) / ma
    return df


def add_price_position_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    hi = df["high"].rolling(window).max()
    lo = df["low"].rolling(window).min()
    df[f"price_pos_{window}d"] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Block C — volatility state
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for w in windows:
        df[f"realvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    atr = _atr(df["high"], df["low"], df["close"])
    df["atr_14"] = atr
    df["atr_norm_14"] = atr / df["close"]

    df["candle_range_norm"] = (df["high"] - df["low"]) / df["close"]
    df["gap_size"] = (df["open"] / df["close"].shift(1) - 1).abs()
    return df


def add_volatility_regime(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    vol = np.log(df["close"] / df["close"].shift(1)).rolling(20).std()
    df["vol_regime_pct"] = vol.rolling(window).rank(pct=True)
    return df


# ---------------------------------------------------------------------------
# Block D — volume and crowding state
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"vol_zscore_{w}d"] = _volume_zscore(df["volume"], w)
    df["ret_vol_interaction"] = df["close"].pct_change().abs() * _volume_zscore(df["volume"])
    # Consecutive up/down sessions
    ret = df["close"].pct_change()
    df["consec_up"] = ret.gt(0).astype(int).groupby((ret.le(0)).cumsum()).cumcount()
    df["consec_down"] = ret.lt(0).astype(int).groupby((ret.ge(0)).cumsum()).cumcount()
    return df


# ---------------------------------------------------------------------------
# Block E — regime features (index-level)
# ---------------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame, index_close: pd.Series) -> pd.DataFrame:
    """
    df is a single-ticker DataFrame; index_close is the EURO STOXX 50 index series
    aligned to the same date index.
    """
    idx = index_close.reindex(df.index)
    df["index_ret_5d"] = idx.pct_change(5)
    df["index_vol_20d"] = np.log(idx / idx.shift(1)).rolling(20).std() * np.sqrt(252)
    # Simple trend: index above/below 50-day MA
    df["index_above_ma50"] = (idx > idx.rolling(50).mean()).astype(int)
    return df


# ---------------------------------------------------------------------------
# Block F — calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df.index)
    df["dow"] = dt.dayofweek
    df["month"] = dt.month
    df["is_month_end"] = dt.is_month_end.astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    return df


# ---------------------------------------------------------------------------
# Block A — alert-level features (merged after per-ticker features are built)
# ---------------------------------------------------------------------------

def add_alert_features(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each event row, count simultaneous alerts and add strength proxies.
    """
    count = (
        events.groupby(["date", "ticker"])["alert_name"]
        .count()
        .rename("n_simultaneous_alerts")
        .reset_index()
    )
    events = events.merge(count, on=["date", "ticker"], how="left")
    return events


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_features(
    panel: pd.DataFrame,
    return_windows: list[int] = (1, 3, 5, 10, 20),
    vol_windows: list[int] = (5, 10, 20),
    volume_windows: list[int] = (5, 20),
    index_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Apply all feature blocks to a panel DataFrame.
    Expects panel to have columns: date, ticker, open, high, low, close, volume.
    Returns panel with all features added.
    """
    frames = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date").copy()

        grp = add_return_features(grp, list(return_windows))
        grp = add_ma_distance_features(grp)
        grp = add_price_position_features(grp)
        grp = add_volatility_features(grp, list(vol_windows))
        grp = add_volatility_regime(grp)
        grp = add_volume_features(grp, list(volume_windows))
        grp = add_calendar_features(grp)

        if index_close is not None:
            grp = add_regime_features(grp, index_close)

        grp["ticker"] = ticker
        frames.append(grp.reset_index())

    return pd.concat(frames, ignore_index=True)
