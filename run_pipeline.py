"""
Full pipeline runner — v2 (fixed probability mapping, alert_name feature, caching).
Usage: python run_pipeline.py
"""

import sys
import warnings
import logging
import random
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
np.random.seed(42)
random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import yaml
import pandas as pd
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
with open("configs/eurostoxx50_tickers.yaml") as f:
    tickers = yaml.safe_load(f)["tickers"]

for d in ["data/raw", "data/processed", "data/features", "results/models", "results/reports"]:
    Path(d).mkdir(parents=True, exist_ok=True)

HORIZONS   = cfg["labels"]["horizons"]
THETA      = cfg["labels"]["failure_threshold"]
MODEL_NAME = cfg["models"]["main"]

# ── Step 1: Load raw data ────────────────────────────────────────────────────
log.info("=== Step 1: Loading raw OHLCV ===")
raw = {}
for t in tickers:
    p = Path(f"data/raw/{t.replace('/', '_')}.parquet")
    if not p.exists():
        log.warning("Missing parquet for %s — skipping", t)
        continue
    df = pd.read_parquet(p)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    raw[t] = df
log.info("Loaded %d tickers", len(raw))

# ── Step 2: Build panel (cached) ─────────────────────────────────────────────
PANEL_PATH = Path("data/processed/panel.parquet")
if PANEL_PATH.exists():
    log.info("=== Step 2: Panel loaded from cache ===")
    panel = pd.read_parquet(PANEL_PATH)
else:
    log.info("=== Step 2: Building panel ===")
    from src.data.preprocess import build_panel
    panel = build_panel(raw, min_history_days=252)
    panel.to_parquet(PANEL_PATH, index=False)
log.info("Panel: %s  (%d tickers)", panel.shape, panel["ticker"].nunique())

# ── Step 3: Feature engineering (cached) ────────────────────────────────────
FEAT_PATH = Path("data/features/features.parquet")
if FEAT_PATH.exists():
    log.info("=== Step 3: Features loaded from cache ===")
    feat_panel = pd.read_parquet(FEAT_PATH)
else:
    log.info("=== Step 3: Feature engineering ===")
    from src.features.engineering import build_features
    from src.features.labels import compute_forward_returns

    try:
        idx_raw = yf.download("^STOXX50E", start=cfg["data"]["start_date"],
                              auto_adjust=True, progress=False)
        idx_raw.columns = [c[0] if isinstance(c, tuple) else c for c in idx_raw.columns]
        index_close = idx_raw["Close"].rename("index_close")
        index_close.index = pd.to_datetime(index_close.index)
        log.info("Index loaded: %d rows", len(index_close))
    except Exception as e:
        log.warning("Index download failed (%s), regime features skipped", e)
        index_close = None

    feat_panel = build_features(
        panel,
        return_windows=cfg["features"]["return_windows"],
        vol_windows=cfg["features"]["volatility_windows"],
        volume_windows=cfg["features"]["volume_windows"],
        index_close=index_close,
    )
    feat_panel = compute_forward_returns(feat_panel, horizons=HORIZONS)
    feat_panel.to_parquet(FEAT_PATH, index=False)
log.info("Feature panel: %s", feat_panel.shape)

# ── Step 4: Alert engine (cached) ────────────────────────────────────────────
EVENTS_PATH = Path("data/features/events_raw.parquet")
if EVENTS_PATH.exists():
    log.info("=== Step 4: Alerts loaded from cache ===")
    events = pd.read_parquet(EVENTS_PATH)
else:
    log.info("=== Step 4: Alert engine ===")
    from src.alerts.engine import run_alert_engine
    from src.features.engineering import add_alert_features
    events = run_alert_engine(panel)
    events = add_alert_features(events, panel)
    events.to_parquet(EVENTS_PATH, index=False)
log.info("Events: %d  Alerts: %d types", len(events), events["alert_name"].nunique())
log.info("Top alerts:\n%s", events["alert_name"].value_counts().head(8).to_string())

# ── Step 5: Labels ───────────────────────────────────────────────────────────
LABELED_PATH = Path("data/features/events_labeled.parquet")
if LABELED_PATH.exists():
    log.info("=== Step 5: Labels loaded from cache ===")
    labeled = pd.read_parquet(LABELED_PATH)
else:
    log.info("=== Step 5: Label construction ===")
    from src.features.labels import assign_labels
    labeled = assign_labels(events, feat_panel, horizons=HORIZONS, theta=THETA)
    labeled.to_parquet(LABELED_PATH, index=False)

labeled["date"] = pd.to_datetime(labeled["date"])
for h in HORIZONS:
    col = f"label_failure_{h}d"
    rate = labeled[col].mean()
    n = labeled[col].notna().sum()
    log.info("  h=%dd  failure_rate=%.3f  n=%d", h, rate, n)

# ── Step 6: Build feature matrix ─────────────────────────────────────────────
log.info("=== Step 6: Feature matrix ===")

EXCLUDE_COLS = {"date", "ticker", "open", "high", "low", "close", "volume",
                "ret_1d_lead", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d",
                "alert_name", "direction", "n_simultaneous_alerts",
                "_dir_raw", "direction_str"}

PRICE_FEATURE_COLS = [c for c in feat_panel.columns if c not in EXCLUDE_COLS]

def build_model_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge price features onto events, one-hot encode categoricals.
    Returns (merged_df, feature_col_list).
    """
    price_feats = feat_panel[["date", "ticker"] + PRICE_FEATURE_COLS].copy()
    price_feats["date"] = pd.to_datetime(price_feats["date"])
    price_feats = price_feats.drop_duplicates(["date", "ticker"])

    merged = df.merge(price_feats, on=["date", "ticker"], how="left", suffixes=("", "_feat"))

    # Store raw categoricals before encoding
    merged["_dir_raw"] = merged["direction"].copy()

    # One-hot encode alert_name and direction (most informative categorical features)
    merged = pd.get_dummies(merged, columns=["direction", "alert_name"], drop_first=False)

    feat_cols = (
        [c for c in merged.columns if c in PRICE_FEATURE_COLS]
        + [c for c in merged.columns if c.startswith("direction_") or c.startswith("alert_name_")]
        + ["n_simultaneous_alerts"]
    )
    # Remove any object-type columns that slipped through
    feat_cols = [c for c in feat_cols if merged[c].dtype != object]

    return merged, feat_cols

merged, feat_cols = build_model_df(labeled)
merged = merged.reset_index(drop=True)   # guarantee unique 0..N-1 index for proba mapping
log.info("Feature matrix: %d rows x %d features", len(merged), len(feat_cols))

# ── Step 7: Model training ────────────────────────────────────────────────────
log.info("=== Step 7: Walk-forward training ===")
from src.models.train import train_evaluate, WalkForwardConfig
from src.models.evaluate import ml_metrics

all_results = {}
for h in HORIZONS:
    label_col = f"label_failure_{h}d"
    valid = merged[label_col].notna()
    X = merged.loc[valid, feat_cols].fillna(-1)   # -1 sentinel: XGBoost handles missing natively
    y = merged.loc[valid, label_col]
    dates = merged.loc[valid, "date"]

    log.info("  Training h=%dd  samples=%d  pos_rate=%.3f", h, len(y), y.mean())

    wf_cfg = WalkForwardConfig(
        n_splits=cfg["validation"]["n_splits"],
        purge_days=h + cfg["validation"]["purge_days"],   # horizon-aware purge
        embargo_days=cfg["validation"]["embargo_days"],
    )
    results = train_evaluate(X, y, dates, model_name=MODEL_NAME, cfg=wf_cfg)
    metrics = ml_metrics(results["y_true"], results["y_pred_proba"])
    all_results[h] = {"results": results, "metrics": metrics}

    log.info("  h=%dd  ROC-AUC=%.4f  PR-AUC=%.4f  Top-decile-precision=%.4f",
             h, metrics["roc_auc"], metrics["pr_auc"], metrics["top_decile_precision"])

    if results["feature_importance"] is not None:
        top10 = results["feature_importance"].head(10)
        log.info("  Top-10 features (h=%dd):\n%s", h, top10.to_string())
        results["feature_importance"].to_csv(f"results/reports/feature_importance_h{h}d.csv",
                                             header=["importance"])

# ── Step 8: Backtest ──────────────────────────────────────────────────────────
log.info("=== Step 8: Backtest ===")
from src.backtest.strategy import compare_strategies
from src.models.evaluate import strategy_metrics

reports = []
for h in HORIZONS:
    res = all_results[h]["results"]
    label_col = f"label_failure_{h}d"
    valid = merged[label_col].notna()
    oos_df = merged[valid].copy()

    # ── Correct probability mapping: dict-map by original row index ────────
    proba_map = dict(zip(res["orig_idx"], res["y_pred_proba"]))
    oos_df["failure_proba"] = oos_df.index.map(proba_map)
    oos_df["direction"] = oos_df["_dir_raw"]

    # Only evaluate on rows that received OOS predictions
    oos_df = oos_df.dropna(subset=["failure_proba", f"fwd_ret_{h}d"])
    log.info("  h=%dd  OOS rows with predictions: %d", h, len(oos_df))

    strat = compare_strategies(
        oos_df, horizon=h,
        confidence_threshold=cfg["strategy"]["confidence_threshold"],
        proba_col="failure_proba",
    )

    row = {"horizon": h}
    for col in ["follow_alert", "blind_inverse", "ml_filtered"]:
        s = strat[col].dropna()
        if len(s) > 0:
            m = strategy_metrics(s, cost_bps=cfg["strategy"]["transaction_cost_bps"])
            row[f"{col}_sharpe"]   = round(m["sharpe"], 3)
            row[f"{col}_hit_rate"] = round(m["hit_rate"], 3)
            row[f"{col}_net_ret"]  = round(m["mean_net_ret"], 5)
            row[f"{col}_n_trades"] = m["n_trades"]
        else:
            row[f"{col}_sharpe"] = row[f"{col}_hit_rate"] = row[f"{col}_net_ret"] = row[f"{col}_n_trades"] = None
    reports.append(row)

# ── Save reports ──────────────────────────────────────────────────────────────
report_df = pd.DataFrame(reports)
report_df.to_csv("results/reports/strategy_comparison.csv", index=False)

ml_rows = []
for h, v in all_results.items():
    m = v["metrics"]
    ml_rows.append({"horizon": h, **{k: round(val, 4) if isinstance(val, float) else val
                                      for k, val in m.items()}})
ml_df = pd.DataFrame(ml_rows)
ml_df.to_csv("results/reports/ml_metrics.csv", index=False)

log.info("\n%s\nSTRATEGY COMPARISON\n%s", "="*60, "="*60)
log.info("\n%s", report_df.to_string(index=False))
log.info("\nML METRICS\n%s", ml_df.to_string(index=False))
log.info("=== Pipeline complete. Results in results/reports/ ===")
