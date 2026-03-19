"""
Trade Explainability
====================
Computes a human-readable rationale for every signal, grounded in:

  • Barber & Odean (2008) — attention-driven retail buying predicts reversals:
    volume surges, extreme price positions, and news events attract retail
    crowding that sophisticated participants can fade.

  • Lopez de Prado (2018) — meta-labeling: model prediction (P(alert failure))
    drives the primary trade decision.

Public API
----------
    compute_crowding_score(row)              → float  0-1 composite crowding score
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


# ── Crowding score (Barber & Odean, 2008) ─────────────────────────────────────

def compute_crowding_score(row: pd.Series) -> float:
    """
    Composite attention/crowding score (Barber & Odean, 2008).

    Score = weighted sum of four attention indicators:
      1. vol_zscore_20d  : volume z-score vs 20-day mean (surge = crowd buying)
      2. price_pos_20d   : price position in 20-day range (extreme = crowd attention)
      3. atr_norm_14     : normalised ATR (volatility surge = crowd panic/excitement)
      4. n_simultaneous_alerts : concurrent alerts on same ticker (pile-on)

    Returns value in [0, 1]. Higher = more crowded/attentive market.
    """
    score = 0.0

    # 1. Volume surge — z-score of today's volume vs 20-day mean
    vol_z = float(row.get("vol_zscore_20d", -1))
    if vol_z >= 0:  # sentinel -1 = missing
        if vol_z >= 3.0:
            score += 0.40
        elif vol_z >= 2.0:
            score += 0.30
        elif vol_z >= 1.5:
            score += 0.20
        elif vol_z >= 1.0:
            score += 0.10

    # 2. Price position in 20-day range (0 = 20d low, 1 = 20d high)
    pos = float(row.get("price_pos_20d", -1))
    if pos >= 0:  # sentinel -1 = missing
        if pos >= 0.90 or pos <= 0.10:   # extreme = crowd attention
            score += 0.30
        elif pos >= 0.80 or pos <= 0.20:
            score += 0.20
        elif pos >= 0.70 or pos <= 0.30:
            score += 0.10

    # 3. Volatility surge (normalised ATR-14)
    atr = float(row.get("atr_norm_14", -1))
    if atr >= 0:  # sentinel -1 = missing
        if atr >= 0.04:    # >4% daily ATR = panic/excitement
            score += 0.20
        elif atr >= 0.025:
            score += 0.10

    # 4. Concurrent alerts on same ticker (pile-on)
    n_alerts = int(row.get("n_simultaneous_alerts", 0))
    if n_alerts >= 3:
        score += 0.10
    elif n_alerts >= 2:
        score += 0.05

    return min(score, 1.0)


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
    top_n: int = 3,
) -> str:
    """
    Returns a one-line human-readable explanation of the trade rationale.

    failure_proba : P(alert fails to follow through) — from the model.

    Attempts SHAP TreeExplainer for feature attribution; falls back to
    model.feature_importances_ if shap is unavailable.
    """
    crowding_narrative = _crowding_narrative(feature_row)
    regime_narrative   = _regime_narrative(feature_row)
    drivers            = _driver_narrative(model, feature_row, feature_cols, top_n)

    return (
        f"{action} {trade_direction} | {alert_direction} '{alert_name}' | "
        f"P(fail)={failure_proba:.2f} | "
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
            vals = sv[1][0]          # class-1 (profitable)
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


def _crowding_narrative(row: pd.Series) -> str:
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
        return "standard conditions"
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
