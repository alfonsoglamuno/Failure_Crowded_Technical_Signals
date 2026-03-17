"""
Alert detection engine.

For each (ticker, date) pair, flags which technical alerts are active.
Alerts are classified as bullish (direction=+1) or bearish (direction=-1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).max()


def _rolling_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).min()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Alert functions — each returns a boolean Series (True = alert active)
# ---------------------------------------------------------------------------

def alert_breakout_high(close: pd.Series, window: int = 20) -> pd.Series:
    """Close above N-day rolling high (bullish)."""
    return close > _rolling_max(close.shift(1), window)


def alert_breakout_low(close: pd.Series, window: int = 20) -> pd.Series:
    """Close below N-day rolling low (bearish)."""
    return close < _rolling_min(close.shift(1), window)


def alert_ma_cross_bullish(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """Fast MA crosses above slow MA."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return (f > s) & (f.shift(1) <= s.shift(1))


def alert_ma_cross_bearish(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """Fast MA crosses below slow MA."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return (f < s) & (f.shift(1) >= s.shift(1))


def alert_rsi_overbought(close: pd.Series, window: int = 14, threshold: float = 70) -> pd.Series:
    """RSI crosses above overbought threshold (bearish)."""
    rsi = _rsi(close, window)
    return (rsi >= threshold) & (rsi.shift(1) < threshold)


def alert_rsi_oversold(close: pd.Series, window: int = 14, threshold: float = 30) -> pd.Series:
    """RSI crosses below oversold threshold (bullish reversal signal)."""
    rsi = _rsi(close, window)
    return (rsi <= threshold) & (rsi.shift(1) > threshold)


def alert_macd_bullish(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line crosses above signal line."""
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return (macd > sig) & (macd.shift(1) <= sig.shift(1))


def alert_macd_bearish(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line crosses below signal line."""
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return (macd < sig) & (macd.shift(1) >= sig.shift(1))


def alert_bb_breakout_up(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Close above upper Bollinger Band."""
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return close > mu + n_std * sigma


def alert_bb_breakout_down(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Close below lower Bollinger Band."""
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return close < mu - n_std * sigma


def alert_volume_spike(volume: pd.Series, window: int = 20, z_thresh: float = 2.0) -> pd.Series:
    """Volume z-score exceeds threshold."""
    return _volume_zscore(volume, window) >= z_thresh


def alert_extreme_return(close: pd.Series, threshold: float = 0.03) -> tuple[pd.Series, pd.Series]:
    """Returns (extreme_up, extreme_down) boolean series."""
    ret = close.pct_change()
    return ret >= threshold, ret <= -threshold


def alert_gap_up(open_: pd.Series, close: pd.Series, threshold: float = 0.01) -> pd.Series:
    """Open gaps up more than threshold vs prior close."""
    return (open_ / close.shift(1) - 1) >= threshold


def alert_gap_down(open_: pd.Series, close: pd.Series, threshold: float = 0.01) -> pd.Series:
    """Open gaps down more than threshold vs prior close."""
    return (open_ / close.shift(1) - 1) <= -threshold


def alert_atr_spike(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, multiplier: float = 2.0
) -> pd.Series:
    """Daily ATR exceeds multiplier x rolling mean ATR."""
    atr = _atr(high, low, close, window)
    return atr > multiplier * atr.rolling(window).mean().shift(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ALERT_REGISTRY: dict[str, tuple[str, "Callable"]] = {
    # name: (direction, function)
    "breakout_high_20":    ("bullish", lambda d: alert_breakout_high(d["close"], 20)),
    "breakout_high_50":    ("bullish", lambda d: alert_breakout_high(d["close"], 50)),
    "breakout_low_20":     ("bearish", lambda d: alert_breakout_low(d["close"], 20)),
    "breakout_low_50":     ("bearish", lambda d: alert_breakout_low(d["close"], 50)),
    "ma_cross_bull_10_50": ("bullish", lambda d: alert_ma_cross_bullish(d["close"], 10, 50)),
    "ma_cross_bull_20_100":("bullish", lambda d: alert_ma_cross_bullish(d["close"], 20, 100)),
    "ma_cross_bear_10_50": ("bearish", lambda d: alert_ma_cross_bearish(d["close"], 10, 50)),
    "ma_cross_bear_20_100":("bearish", lambda d: alert_ma_cross_bearish(d["close"], 20, 100)),
    "rsi_overbought":      ("bearish", lambda d: alert_rsi_overbought(d["close"])),
    "rsi_oversold":        ("bullish", lambda d: alert_rsi_oversold(d["close"])),
    "macd_bullish":        ("bullish", lambda d: alert_macd_bullish(d["close"])),
    "macd_bearish":        ("bearish", lambda d: alert_macd_bearish(d["close"])),
    "bb_breakout_up":      ("bullish", lambda d: alert_bb_breakout_up(d["close"])),
    "bb_breakout_down":    ("bearish", lambda d: alert_bb_breakout_down(d["close"])),
    "volume_spike":        ("neutral", lambda d: alert_volume_spike(d["volume"])),
    "atr_spike":           ("neutral", lambda d: alert_atr_spike(d["high"], d["low"], d["close"])),
    "gap_up":              ("bullish", lambda d: alert_gap_up(d["open"], d["close"])),
    "gap_down":            ("bearish", lambda d: alert_gap_down(d["open"], d["close"])),
}


def run_alert_engine(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all alerts to the panel and return an events DataFrame.

    Output columns: date, ticker, alert_name, direction
    """
    events = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")
        data = {col: grp[col] for col in ["open", "high", "low", "close", "volume"]}

        for alert_name, (direction, fn) in ALERT_REGISTRY.items():
            try:
                mask: pd.Series = fn(data)
            except Exception:
                continue

            active_dates = mask[mask].index
            for dt in active_dates:
                events.append(
                    {"date": dt, "ticker": ticker, "alert_name": alert_name, "direction": direction}
                )

    return pd.DataFrame(events)
