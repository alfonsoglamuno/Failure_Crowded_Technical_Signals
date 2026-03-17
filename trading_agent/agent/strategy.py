"""
Strategy engine — translates P(failure) scores into trade signals.

Decision logic (informed by literature):
  1. Regime gate (Lopez de Prado, 2018):
       When the market is trending in the same direction as the alert,
       the alert has a better chance of following through. We raise the
       effective fade threshold to avoid fading trends blindly.

  2. Crowding gate (Barber & Odean, 2008):
       The fade thesis is strongest when the alert is attention-driven —
       high volume, extreme price position, or multiple concurrent signals.
       Low-crowding signals are skipped even if P(failure) is high, because
       the failure probability may be regime-driven, not crowd-driven.

  3. Core decision:
       P(failure) >= effective_fade_threshold  → FADE (contrarian)
       P(failure) <= follow_threshold          → FOLLOW (momentum)
       between                                 → SKIP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    yahoo_ticker: str
    alert_name: str
    alert_direction: str     # bullish / bearish / neutral
    trade_direction: str     # BUY / SELL
    action: str              # FADE / FOLLOW
    failure_proba: float
    horizon_days: int
    conviction: float        # 0–1, distance from effective decision boundary
    crowding_score: float    # 0–1, Barber & Odean attention intensity
    regime_boost_applied: bool = False  # True when threshold was raised for regime


def make_signals(
    feature_df: pd.DataFrame,
    probas: pd.Series,
    fade_threshold: float = 0.65,
    follow_threshold: float = 0.35,
    horizon_days: int = 3,
    crowding_min_score: float = 0.30,
    regime_threshold_boost: float = 0.05,
) -> list[Signal]:
    """
    Convert model probabilities into trade signals.

    Parameters
    ----------
    crowding_min_score : float
        Minimum crowding score required to FADE.  Signals with lower crowding
        are skipped — the model may be flagging a regime shift, not a crowd
        exhaustion event (Barber & Odean gate).
    regime_threshold_boost : float
        Added to fade_threshold when the market is trending in the same
        direction as the alert.  Prevents fading strong institutional momentum
        (Lopez de Prado regime awareness).
    """
    from agent.explain import compute_crowding_score

    signals = []

    for idx, prob in probas.items():
        row       = feature_df.loc[idx]
        direction = row.get("_dir_raw", "neutral")
        ticker    = row.get("ticker", "")
        alert     = row.get("_alert_name_raw", "") or _infer_alert(row)

        if direction == "neutral":
            continue   # no directional trade possible

        # ── Crowding score (Barber & Odean attention conditions) ──────────────
        crowding = compute_crowding_score(row)

        # ── Regime-aware effective threshold (Lopez de Prado) ─────────────────
        # If the market is trending WITH the alert direction, raise our bar —
        # the alert might be correct (momentum), not failing.
        regime_boost = False
        index_above_200ma = float(row.get("index_above_200ma", -1.0))
        if index_above_200ma >= 0:   # feature is present
            market_uptrend   = (index_above_200ma == 1.0)
            market_downtrend = (index_above_200ma == 0.0)
            if (market_uptrend and direction == "bullish") or \
               (market_downtrend and direction == "bearish"):
                regime_boost = True

        eff_fade = fade_threshold + (regime_threshold_boost if regime_boost else 0.0)

        # ── Decision ──────────────────────────────────────────────────────────
        if prob >= eff_fade:
            # ── Crowding gate: skip low-attention FADE signals ─────────────
            if crowding < crowding_min_score:
                log.debug(
                    "SKIP FADE %s %s — P=%.3f but crowding=%.2f < %.2f "
                    "(no attention signal to fade)",
                    ticker, alert, prob, crowding, crowding_min_score,
                )
                continue

            action          = "FADE"
            trade_direction = "SELL" if direction == "bullish" else "BUY"
            conviction      = (prob - eff_fade) / (1.0 - eff_fade)

        elif prob <= follow_threshold:
            action          = "FOLLOW"
            trade_direction = "BUY" if direction == "bullish" else "SELL"
            conviction      = (follow_threshold - prob) / follow_threshold

        else:
            continue   # uncertain zone — no trade

        signals.append(Signal(
            ticker=ticker,
            yahoo_ticker=ticker,
            alert_name=alert,
            alert_direction=direction,
            trade_direction=trade_direction,
            action=action,
            failure_proba=float(prob),
            horizon_days=horizon_days,
            conviction=float(conviction),
            crowding_score=float(crowding),
            regime_boost_applied=regime_boost,
        ))

    # Sort by conviction descending (highest confidence first)
    signals.sort(key=lambda s: s.conviction, reverse=True)
    return signals


def _infer_alert(row: pd.Series) -> str:
    """Infer alert name from one-hot encoded columns."""
    for col in row.index:
        if col.startswith("alert_name_") and row[col] == 1:
            return col.replace("alert_name_", "")
    return "unknown"


def filter_signals(
    signals: list[Signal],
    max_trades: int = 5,
    allow_short: bool = True,
) -> list[Signal]:
    """
    Apply pre-trade filters:
    - Limit to max_trades per day (sorted by conviction, highest first)
    - Optionally block SELL signals (long-only mode)
    - No duplicate tickers
    """
    seen_tickers = set()
    filtered     = []

    for sig in signals:
        if sig.ticker in seen_tickers:
            continue
        if sig.trade_direction == "SELL" and not allow_short:
            log.info("Skipping SELL for %s (long-only mode)", sig.ticker)
            continue
        seen_tickers.add(sig.ticker)
        filtered.append(sig)
        if len(filtered) >= max_trades:
            break

    return filtered
