"""
Trade Explainability
====================
Computes a human-readable rationale for every signal, grounded in:

  • Barber & Odean (2008) — attention-driven retail buying predicts reversals:
    volume surges, extreme price positions, and news events attract retail
    crowding that sophisticated participants can fade.

  • Lopez de Prado (2018) — meta-labeling: a second signal (crowding intensity)
    should confirm the primary model prediction before we bet on it.

Public API
----------
    compute_crowding_score(row)        → float  0-1
    build_explanation(model, row, ...) → str    human-readable narrative
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Human-readable feature labels ─────────────────────────────────────────────

_FEAT_LABELS: dict[str, str] = {
    "vol_ratio_20d":           "vol vs 20d avg",
    "vol_ratio_5d":            "vol vs 5d avg",
    "n_simultaneous_alerts":   "concurrent alerts",
    "atr_pct_20":              "ATR% regime",
    "atr_pct_10":              "ATR% 10d",
    "atr_pct_5":               "ATR% 5d",
    "vol_regime":              "vol regime flag",
    "ret_1d":                  "1d return",
    "ret_3d":                  "3d return",
    "ret_5d":                  "5d return",
    "ret_10d":                 "10d return",
    "ret_20d":                 "20d return",
    "ma_dist_20d":             "dist from 20d MA",
    "ma_dist_50d":             "dist from 50d MA",
    "ma_dist_200d":            "dist from 200d MA",
    "price_position_52w":      "52w price position",
    "index_above_200ma":       "mkt above 200MA",
    "index_ret_1d":            "mkt 1d return",
    "index_ret_3d":            "mkt 3d return",
    "dow_monday":              "Monday",
    "dow_friday":              "Friday",
}


# ── Crowding score ─────────────────────────────────────────────────────────────

def compute_crowding_score(row: pd.Series) -> float:
    """
    Returns a crowding / attention score in [0, 1].

    Components (Barber & Odean, 2008):
      • Volume anomaly  — retail attention proxy (0-0.40)
      • Simultaneous alerts — multiple crowd signals (0-0.30)
      • Price at 52w extreme — price attention zone (0-0.20)
      • Compressed volatility — ATR exhaustion → reversal likely (0-0.10)

    A score ≥ 0.30 means at least one strong attention indicator is present.
    """
    score = 0.0

    # ── Volume component ──────────────────────────────────────────────────────
    vol = _safe_float(row, "vol_ratio_20d")
    if vol >= 2.5:
        score += 0.40
    elif vol >= 2.0:
        score += 0.30
    elif vol >= 1.5:
        score += 0.20
    elif vol >= 1.2:
        score += 0.10

    # ── Concurrent alerts component ───────────────────────────────────────────
    n_alerts = int(_safe_float(row, "n_simultaneous_alerts", default=1.0))
    if n_alerts >= 3:
        score += 0.30
    elif n_alerts >= 2:
        score += 0.15

    # ── 52-week price extreme component ───────────────────────────────────────
    pos = _safe_float(row, "price_position_52w")
    if pos >= 0.90 or pos <= 0.10:
        score += 0.20
    elif pos >= 0.80 or pos <= 0.20:
        score += 0.10

    # ── Compressed volatility component ───────────────────────────────────────
    atr = _safe_float(row, "atr_pct_20")
    if atr < 0.010:
        score += 0.10
    elif atr < 0.015:
        score += 0.05

    return round(min(score, 1.0), 3)


# ── SHAP explanation ───────────────────────────────────────────────────────────

def build_explanation(
    model,
    feature_row: pd.Series,
    feature_cols: list[str],
    failure_proba: float,
    alert_name: str,
    alert_direction: str,
    action: str,
    trade_direction: str,
    crowding_score: float,
    top_n: int = 3,
) -> str:
    """
    Returns a one-line human-readable explanation of the trade rationale.

    Attempts SHAP TreeExplainer for feature attribution; falls back to
    model.feature_importances_ if shap is unavailable.
    """
    crowding_narrative = _crowding_narrative(feature_row, crowding_score)
    regime_narrative   = _regime_narrative(feature_row)
    drivers            = _driver_narrative(model, feature_row, feature_cols, top_n)

    return (
        f"{action} {trade_direction} | {alert_direction} '{alert_name}' | "
        f"P(fail)={failure_proba:.2f} crowd={crowding_score:.2f} | "
        f"{crowding_narrative} | {regime_narrative} | "
        f"drivers: [{drivers}]"
    )


def _driver_narrative(model, row: pd.Series, feature_cols: list[str], top_n: int) -> str:
    """Top contributing features — SHAP preferred, importance fallback."""
    try:
        import shap
        X = row[feature_cols].fillna(-1).values.reshape(1, -1)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # shap_values returns (1, n_features) for XGBoost binary
        if isinstance(sv, list):
            vals = sv[1][0]          # class-1 (failure)
        elif hasattr(sv, "values"):  # newer shap returns Explanation object
            v = sv.values
            vals = v[0] if v.ndim == 2 else v[0, :, 1]
        else:
            vals = sv[0]

        pairs = sorted(
            [(feature_cols[i], float(vals[i])) for i in range(len(feature_cols))],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

    except Exception:
        # Fallback: model feature importances (less precise, but always available)
        try:
            imps = model.feature_importances_
            pairs = sorted(
                [(feature_cols[i], float(imps[i])) for i in range(len(feature_cols))],
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
        except Exception:
            return "n/a"

    parts = []
    for fname, fval in pairs:
        label = _FEAT_LABELS.get(fname, fname)
        raw   = _safe_float(row, fname, default=np.nan)
        sign  = "+" if fval > 0 else "-"
        raw_s = f"{raw:.3f}" if not np.isnan(raw) else "?"
        parts.append(f"{label}={raw_s}({sign})")

    return ", ".join(parts)


def _crowding_narrative(row: pd.Series, score: float) -> str:
    """Short attention condition description."""
    parts = []

    vol = _safe_float(row, "vol_ratio_20d")
    if vol >= 2.0:
        parts.append(f"vol {vol:.1f}x avg (retail attention surge)")
    elif vol >= 1.5:
        parts.append(f"vol {vol:.1f}x avg (elevated)")

    n = int(_safe_float(row, "n_simultaneous_alerts", default=1.0))
    if n >= 2:
        parts.append(f"{n} concurrent alerts (crowded)")

    pos = _safe_float(row, "price_position_52w")
    if pos >= 0.90:
        parts.append("near 52w high (exhaustion)")
    elif pos <= 0.10:
        parts.append("near 52w low (oversold)")

    if not parts:
        return f"standard conditions (crowd={score:.2f})"
    return "; ".join(parts)


def _regime_narrative(row: pd.Series) -> str:
    """Market regime context."""
    above_200 = _safe_float(row, "index_above_200ma", default=-1.0)
    idx_ret   = _safe_float(row, "index_ret_3d", default=np.nan)

    if above_200 < 0:
        return "regime unknown"

    regime = "market uptrend" if above_200 == 1.0 else "market downtrend"
    if not np.isnan(idx_ret):
        regime += f" (mkt 3d={idx_ret:+.1%})"
    return regime


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default
