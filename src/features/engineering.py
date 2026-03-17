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
        df[f"dist_ma{w}"] = (df["close"] - ma) / ma.replace(0, np.nan)
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

    # Commission-awareness: how many round-trips of commission fit inside one ATR.
    # IBKR round-trip ≈ 0.10% (2 × 0.05%). High ratio = stock moves dominate cost.
    # Model uses this to avoid illiquid/tight stocks where commission eats moves.
    _COMMISSION_RT = 0.001   # 0.10% round-trip (entry + exit, 0.05% each side)
    df["atr_vs_commission"] = df["atr_norm_14"] / _COMMISSION_RT
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
    # +1 so a single qualifying day = 1, not 0
    df["consec_up"] = ret.gt(0).astype(int).groupby((ret.le(0)).cumsum()).cumcount() + 1
    df["consec_down"] = ret.lt(0).astype(int).groupby((ret.ge(0)).cumsum()).cumcount() + 1
    # Zero out the count on non-qualifying days
    df.loc[ret.le(0), "consec_up"] = 0
    df.loc[ret.ge(0), "consec_down"] = 0
    return df


# ---------------------------------------------------------------------------
# Block E — regime features (index-level)
# ---------------------------------------------------------------------------

def _align_index(index_close: pd.Series, target_index: pd.Index) -> pd.Series:
    """
    Align index_close to target_index safely, handling timezone mismatches.
    yfinance may return tz-aware timestamps; ticker DataFrames are tz-naive.
    Uses forward-fill to cover weekends/holidays where index has no close.
    """
    src = index_close.copy()
    # Normalise to tz-naive midnight timestamps
    src.index = pd.to_datetime(src.index).normalize()
    if src.index.tz is not None:
        src.index = src.index.tz_localize(None)

    tgt = pd.to_datetime(target_index).normalize()
    if tgt.tz is not None:
        tgt = tgt.tz_localize(None)

    aligned = src.reindex(tgt, method="ffill")
    aligned.index = target_index   # restore original index for downstream .loc ops
    return aligned


def add_regime_features(df: pd.DataFrame, index_close: pd.Series) -> pd.DataFrame:
    """
    Add EURO STOXX 50 regime and correlation features to a single-ticker DataFrame.

    Features added:
      index_ret_1d/5d/20d      — index momentum at multiple horizons
      index_vol_20d             — index realised volatility (annualised)
      index_above_ma50/200      — trend regime flags
      index_corr_20d/60d        — rolling stock-to-index return correlation
      beta_20d/60d              — rolling market beta (cov/var)
      rel_strength_5d/20d       — stock return minus index return (outperformance)
      index_regime              — +1 bull / 0 neutral / -1 bear (20d return threshold ±2%)
    """
    idx = _align_index(index_close, df.index)

    df["index_ret_1d"]  = idx.pct_change(1)
    df["index_ret_5d"]  = idx.pct_change(5)
    df["index_ret_20d"] = idx.pct_change(20)
    df["index_vol_20d"] = (
        np.log(idx / idx.shift(1)).rolling(20).std() * np.sqrt(252)
    )

    # Trend regime flags
    df["index_above_ma50"]  = (idx > idx.rolling(50).mean()).astype(int)
    df["index_above_ma200"] = (idx > idx.rolling(200).mean()).astype(int)

    # Rolling stock-to-index correlation and beta
    stock_ret = df["close"].pct_change()
    idx_ret   = idx.pct_change()
    for w in [20, 60]:
        df[f"index_corr_{w}d"] = stock_ret.rolling(w).corr(idx_ret)
        cov = stock_ret.rolling(w).cov(idx_ret)
        var = idx_ret.rolling(w).var()
        df[f"beta_{w}d"] = cov / var.replace(0, np.nan)

    # Relative strength: positive = stock outperforming index
    df["rel_strength_5d"]  = stock_ret.rolling(5).sum()  - idx_ret.rolling(5).sum()
    df["rel_strength_20d"] = stock_ret.rolling(20).sum() - idx_ret.rolling(20).sum()

    # Index regime bucket: use np.where to avoid boolean Series alignment issues
    idx_20d = idx.pct_change(20)
    df["index_regime"] = np.where(
        idx_20d >  0.02,  1,
        np.where(idx_20d < -0.02, -1, 0)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Block F — calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df.index).normalize()
    if dt.tz is not None:
        dt = dt.tz_localize(None)
    df["dow"]            = dt.dayofweek
    df["month"]          = dt.month
    df["is_month_end"]   = dt.is_month_end.astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    # Session bucket: open (Mon-Tue morning) vs midday vs close (Thu-Fri)
    # Proxy for intraday regime — EOM/BOW have higher institutional flow
    df["is_week_start"]  = (dt.dayofweek == 0).astype(int)   # Monday
    df["is_week_end"]    = (dt.dayofweek == 4).astype(int)   # Friday
    return df


# ---------------------------------------------------------------------------
# Block H — intraday structure proxies from daily OHLCV
# ---------------------------------------------------------------------------
# These approximate microstructure signals without requiring tick/L2 data.
# Research shows: close-vs-range, open-to-close direction, gap/overnight drift,
# and relative volume by session are highly informative for intraday reversals.
# ---------------------------------------------------------------------------

def add_intraday_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday structure proxies computable from daily OHLCV.

    Features:
      close_vs_range       — 0=closed at low, 1=closed at high (order-flow proxy)
      open_to_close_ret    — signed intraday move (open → close)
      gap_pct              — signed overnight gap vs prior close
      gap_is_filled        — 1 if price moved back through the gap during the day
      vwap_distance        — (close - typical_price) / typical_price,
                             typical_price = (H+L+2C)/4
      vol_vs_dow_baseline  — volume vs 8-week same-day-of-week rolling average
                             (rough "relative volume at this time of day" proxy)
      rel_strength_1d      — 1-day return minus same-day sector/index return
                             (residualized move after controlling for macro)
      reversal_intrabar    — recovery fraction after a gap-down opening
                             (how much of the morning gap was recovered intraday)
    """
    daily_range = (df["high"] - df["low"]).replace(0, np.nan)

    # Where did price close within today's range? 0 = at low, 1 = at high.
    df["close_vs_range"] = (df["close"] - df["low"]) / daily_range

    # Signed intraday return: positive = buyers won the session.
    df["open_to_close_ret"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # Overnight gap: signed pct from yesterday's close to today's open.
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # Gap fill: did price trade back through the gap level during the day?
    prior_close = df["close"].shift(1)
    gap_up   = df["open"] > prior_close
    gap_down = df["open"] < prior_close
    df["gap_is_filled"] = np.where(
        gap_up,   (df["low"]  <= prior_close).astype(float),
        np.where(
            gap_down, (df["high"] >= prior_close).astype(float),
            0.0
        )
    )

    # VWAP proxy: typical price = (H + L + 2C) / 4
    # Close above typical price = buyers held into the close.
    typical_px = (df["high"] + df["low"] + 2 * df["close"]) / 4
    df["vwap_distance"] = (df["close"] - typical_px) / typical_px.replace(0, np.nan)

    # Relative volume vs same-day-of-week rolling baseline (40 sessions ≈ 8 weeks).
    # Approximates "is today's volume unusual for this time of day?"
    vol_baseline = df["volume"].rolling(40, min_periods=10).mean()
    df["vol_vs_dow_baseline"] = df["volume"] / vol_baseline.replace(0, np.nan)

    # Reversal intrabar: after a gap-down, how much did price recover?
    # Positive = gap-down that recovered → buying pressure absorbed selling.
    gap_down_mag = (-df["gap_pct"]).clip(lower=0)   # only gap-downs
    df["reversal_intrabar"] = np.where(
        gap_down_mag > 0,
        df["open_to_close_ret"] / gap_down_mag.replace(0, np.nan),
        0.0,
    )

    return df


# ---------------------------------------------------------------------------
# Block G — inter-stock correlation (peer crowding / herding)
# ---------------------------------------------------------------------------

def add_peer_correlation_feature(result: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    For each ticker × date, compute the average pairwise rolling return
    correlation with all other tickers in the universe.

    avg_peer_corr_20d:
      High  → stocks moving together = herding / crowded regime.
              Breakout/oversold signals in a herded market are more likely to
              be noise (everyone hitting the same screen) → model fades more.
      Low   → idiosyncratic move = genuinely stock-specific → signal is cleaner.

    This is a post-loop step because it requires all tickers simultaneously.
    """
    # Pivot into a wide returns matrix (date × ticker)
    df = result.copy()
    df["_ret"] = df.groupby("ticker")["close"].pct_change()
    ret_wide = df.pivot_table(index="date", columns="ticker", values="_ret")

    tickers = ret_wide.columns.tolist()
    if len(tickers) < 2:
        result["avg_peer_corr_20d"] = np.nan
        return result

    # For each ticker, compute rolling mean correlation with all peers.
    # We use pairwise rolling corr in a vectorised way: build a 3D structure.
    corr_cols = {}
    for t in tickers:
        others = [c for c in tickers if c != t]
        # Rolling corr of t with each peer, then average across peers
        peer_corrs = pd.concat(
            [ret_wide[t].rolling(window).corr(ret_wide[peer]) for peer in others],
            axis=1,
        )
        corr_cols[t] = peer_corrs.mean(axis=1)

    avg_corr_wide = pd.DataFrame(corr_cols)   # date × ticker
    avg_corr_melted = (
        avg_corr_wide.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="avg_peer_corr_20d")
    )
    avg_corr_melted["date"] = pd.to_datetime(avg_corr_melted["date"])
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(avg_corr_melted, on=["date", "ticker"], how="left")
    result = result.drop(columns=["_ret"], errors="ignore")
    return result


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
    Returns panel with all features added (~67 features when index_close provided).

    Pass index_close (EURO STOXX 50 Close series) to enable Block E:
    regime, correlation, beta, and relative-strength features.
    index_close may be tz-aware or tz-naive — alignment is handled internally.
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
        grp = add_intraday_structure_features(grp)

        if index_close is not None:
            grp = add_regime_features(grp, index_close)
            # 1-day residualized move: stock return minus index return
            idx = _align_index(index_close, grp.index)
            grp["rel_strength_1d"] = grp["close"].pct_change(1) - idx.pct_change(1)

        grp["ticker"] = ticker
        frames.append(grp.reset_index())

    result = pd.concat(frames, ignore_index=True)

    # Block G: inter-stock peer correlation — requires all tickers together
    result = add_peer_correlation_feature(result)

    return result
