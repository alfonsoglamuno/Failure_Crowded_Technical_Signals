"""
Strategy engine — translates model scores into trade signals.

Core thesis (Barber & Odean, 2008 + Lopez de Prado, 2018):
  When a technical signal is widely noticed and acted upon (crowded),
  it tends to fail to follow through. We predict P(failure) and:
  1. Crowding gate: only FADE when attention is elevated
  2. Regime gate: raise fade threshold when market trends with alert
  3. Decision: P(failure) >= threshold → FADE; <= follow_threshold → FOLLOW

Optional alert_whitelist restricts which alert types are considered.
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
    failure_proba: float     # P(alert fails to follow through)
    horizon_days: int
    conviction: float        # distance from decision boundary
    crowding_score: float    # 0-1 attention intensity (Barber & Odean)
    regime_boost_applied: bool = False


def make_signals(
    feature_df: pd.DataFrame,
    probas: pd.Series,
    fade_threshold: float = 0.60,
    follow_threshold: float = 0.40,
    horizon_days: int = 1,
    crowding_min_score: float = 0.30,
    regime_threshold_boost: float = 0.05,
    alert_whitelist: list[str] | None = None,
) -> list[Signal]:
    """
    Convert model probabilities (P(failure)) into trade signals.

    Decision logic per row:
      1. Whitelist gate: skip if alert not in whitelist.
      2. Direction gate: skip neutral alerts.
      3. Crowding score: compute composite attention indicator (Barber & Odean).
      4. Regime boost: if market trends WITH alert direction → raise fade threshold.
      5. P(failure) >= eff_fade_threshold AND crowding >= crowding_min_score → FADE
         FADE bearish → BUY (contrarian long)
         FADE bullish → SELL (contrarian short, only if allow_short)
      6. P(failure) <= follow_threshold → FOLLOW (no crowding gate)
         FOLLOW bullish → BUY (momentum)
         FOLLOW bearish → SELL (momentum short, only if allow_short)
      7. Otherwise: SKIP (uncertain zone — no trade)
    """
    from agent.explain import compute_crowding_score

    signals = []

    for idx, prob in probas.items():
        row       = feature_df.loc[idx]
        direction = row.get("_dir_raw", "neutral")
        ticker    = row.get("ticker", "")
        alert     = row.get("_alert_name_raw", "") or _infer_alert(row)

        # ── 1. Alert whitelist gate ───────────────────────────────────────────
        if alert_whitelist and alert not in alert_whitelist:
            continue

        # ── 2. Direction gate ─────────────────────────────────────────────────
        if direction == "neutral":
            continue   # no directional trade possible

        # ── 3. Crowding score (Barber & Odean attention composite) ────────────
        crowding = compute_crowding_score(row)

        # ── 4. Regime boost: if market trends with the alert → raise threshold ─
        regime_boost = False
        eff_fade = fade_threshold
        above_200 = float(row.get("index_above_200ma", -1))
        if above_200 >= 0:
            # Market uptrend + bullish alert → hard to fade; raise threshold
            if above_200 == 1.0 and direction == "bullish":
                eff_fade = fade_threshold + regime_threshold_boost
                regime_boost = True
            # Market downtrend + bearish alert → hard to fade; raise threshold
            elif above_200 == 0.0 and direction == "bearish":
                eff_fade = fade_threshold + regime_threshold_boost
                regime_boost = True

        # ── 5. FADE decision: high P(failure) + crowding present ──────────────
        if prob >= eff_fade:
            if crowding < crowding_min_score:
                continue   # crowding gate not met — signal not crowded enough to fade

            action          = "FADE"
            trade_direction = "SELL" if direction == "bullish" else "BUY"
            conviction      = (prob - eff_fade) / (1.0 - eff_fade) if eff_fade < 1.0 else 0.0

        # ── 6. FOLLOW decision: low P(failure) ────────────────────────────────
        elif prob <= follow_threshold:
            action          = "FOLLOW"
            trade_direction = "BUY" if direction == "bullish" else "SELL"
            conviction      = (follow_threshold - prob) / follow_threshold if follow_threshold > 0 else 0.0

        # ── 7. Uncertain zone — skip ──────────────────────────────────────────
        else:
            continue

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
    max_trades: int = 50,
    allow_short: bool = True,
) -> list[Signal]:
    """
    Apply pre-trade filters:
    - Optionally block SELL signals (long-only mode)
    - No duplicate tickers
    - No hard top-N cap — position limits are enforced via max_open_positions
      and capital constraints instead.
    """
    seen_tickers  = set()
    filtered      = []
    blocked_short = 0

    for sig in signals:
        if sig.ticker in seen_tickers:
            continue
        if sig.trade_direction == "SELL" and not allow_short:
            log.debug("Skipping SELL for %s (long-only mode)", sig.ticker)
            blocked_short += 1
            continue
        seen_tickers.add(sig.ticker)
        filtered.append(sig)

    if blocked_short:
        log.info(
            "Long-only filter: %d SELL signal(s) blocked, %d BUY signal(s) pass",
            blocked_short, len(filtered),
        )

    return filtered
