"""
Feature engineering for the trading agent.
Wraps the research project's feature pipeline for real-time use.
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from src.features.engineering import (
    add_return_features,
    add_ma_distance_features,
    add_price_position_features,
    add_volatility_features,
    add_volatility_regime,
    add_volume_features,
    add_calendar_features,
    add_regime_features,
)

log = logging.getLogger(__name__)


def build_features_for_ticker(
    ohlcv: pd.DataFrame,
    index_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build feature vector for a single ticker's OHLCV history.
    Returns DataFrame indexed by date with all feature columns.
    """
    df = ohlcv.copy().sort_values("date").set_index("date")

    df = add_return_features(df, [1, 3, 5, 10, 20])
    df = add_ma_distance_features(df)
    df = add_price_position_features(df)
    df = add_volatility_features(df, [5, 10, 20])
    df = add_volatility_regime(df)
    df = add_volume_features(df, [5, 20])
    df = add_calendar_features(df)

    if index_close is not None:
        df = add_regime_features(df, index_close)

    return df.reset_index()


def build_feature_row(
    events: pd.DataFrame,
    universe_data: dict[str, pd.DataFrame],
    feature_cols: list[str],
    index_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    For each event in today's alert list, build the full feature vector
    aligned with the model's expected feature columns.

    Returns a DataFrame with one row per event, columns = feature_cols.
    """
    rows = []

    for _, event in events.iterrows():
        ticker = event["ticker"]
        ohlcv = universe_data.get(ticker)
        if ohlcv is None or len(ohlcv) < 50:
            continue

        try:
            feat_df = build_features_for_ticker(ohlcv, index_close)
            today_row = feat_df.iloc[-1]
        except Exception as e:
            log.error("Feature build failed for %s: %s", ticker, e)
            continue

        row = event.to_dict()
        for col in feat_df.columns:
            if col not in row:
                row[col] = today_row[col] if col in today_row.index else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Preserve raw labels before one-hot encoding (used by signal logger)
    result["_dir_raw"] = result["direction"]
    result["_alert_name_raw"] = result["alert_name"]
    result = pd.get_dummies(result, columns=["direction", "alert_name"], drop_first=False)
    # Newer pandas returns bool dtype for dummies — XGBoost needs numeric
    bool_cols = result.select_dtypes(include="bool").columns
    result[bool_cols] = result[bool_cols].astype(int)

    # Add n_simultaneous_alerts if missing
    if "n_simultaneous_alerts" not in result.columns:
        result["n_simultaneous_alerts"] = 1

    # Align to model's expected feature columns
    for col in feature_cols:
        if col not in result.columns:
            result[col] = 0   # fill missing one-hot columns

    # Keep only model feature columns + metadata (no overlap)
    meta_cols = ["date", "ticker", "_dir_raw", "_alert_name_raw"]
    available_meta = [c for c in meta_cols if c in result.columns]
    feature_subset = [c for c in feature_cols if c in result.columns]
    # Guard against any remaining duplicate column names
    keep = available_meta + feature_subset
    result = result[keep].loc[:, ~result[keep].columns.duplicated()]

    return result.fillna(-1)   # -1 sentinel matches training


def load_feature_cols(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def save_feature_cols(cols: list[str], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cols, f)
