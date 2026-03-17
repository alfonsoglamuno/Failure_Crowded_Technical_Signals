"""
Bootstrap script — fetches historical data, runs the full research pipeline,
and saves a trained XGBoost model ready for the agent.

Run once before starting the agent:
    python bootstrap_model.py --yfinance    # no IBKR required (recommended first run)
    python bootstrap_model.py --paper       # use IBKR paper port 4002
    python bootstrap_model.py --live        # use IBKR live port 4001
"""

import argparse
import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fetch_yfinance(tickers: list[str], days: int, cache_dir: str | None) -> dict:
    """Fetch OHLCV data via yfinance (no IBKR needed)."""
    import yfinance as yf
    from datetime import datetime, timedelta

    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    results = {}

    log.info("Fetching %d tickers from yfinance (start=%s)...", len(tickers), start)
    for ticker in tickers:
        if cache:
            cache_file = cache / f"{ticker.replace('/', '_')}.parquet"
            if cache_file.exists():
                age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if age < 1:
                    results[ticker] = pd.read_parquet(cache_file)
                    log.debug("Cache hit: %s", ticker)
                    continue
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if raw.empty:
                log.warning("No data for %s", ticker)
                continue
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            df = df[df["close"] > 0].reset_index()
            df.rename(columns={"index": "date", "Date": "date"}, inplace=True)
            if "date" not in df.columns:
                df.insert(0, "date", df.index)
            df["ticker"] = ticker
            if cache:
                df.to_parquet(cache_file, index=False)
            results[ticker] = df
            log.info("Fetched %s: %d rows", ticker, len(df))
        except Exception as e:
            log.error("Failed to fetch %s: %s", ticker, e)

    return results


def main(paper: bool = True, use_yfinance: bool = False):
    # ── Load config ─────────────────────────────────────────────────────────
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/ibkr_contracts.yaml") as f:
        contracts_cfg = yaml.safe_load(f)["contracts"]

    parent_tickers_file = cfg["universe"]["parent_tickers_file"]
    with open(parent_tickers_file) as f:
        tickers = yaml.safe_load(f)["tickers"]

    log.info("Bootstrap starting. Universe: %d tickers. yfinance=%s paper=%s",
             len(tickers), use_yfinance, paper)

    # ── Step 1: Fetch data ───────────────────────────────────────────────────
    cache_dir = cfg["data"]["cache_dir"]
    history_days = cfg["data"]["history_days"]

    if use_yfinance:
        universe_data = _fetch_yfinance(tickers, history_days, cache_dir)
    else:
        from agent.data_feed import IBKRFeed
        feed = IBKRFeed(cfg, paper=paper)
        ibkr_ok = False
        try:
            feed.connect()
            ibkr_ok = True
        except ConnectionError as e:
            port = cfg["ibkr"]["paper_port"] if paper else cfg["ibkr"]["live_port"]
            log.warning("IBKR connect failed (%s). Falling back to yfinance.", e)
            log.warning("(To use IBKR: ensure IB Gateway is running on port %d)", port)

        if ibkr_ok:
            log.info("Fetching %d days via IBKR for %d tickers...", history_days, len(tickers))
            universe_data = feed.fetch_universe(tickers, contracts_cfg,
                                                days=history_days, cache_dir=cache_dir)
            feed.disconnect()
        else:
            universe_data = _fetch_yfinance(tickers, history_days, cache_dir)

    log.info("Fetched data for %d tickers", len(universe_data))

    if not universe_data:
        log.error("No data fetched. Check connectivity and ticker list.")
        sys.exit(1)

    # ── Step 2: Build panel ──────────────────────────────────────────────────
    from src.data.preprocess import build_panel

    raw_dict = {}
    for ticker, df in universe_data.items():
        d = df.copy()
        # Normalise to lowercase first, then title-case OHLCV for build_panel
        d.columns = [c.lower() for c in d.columns]
        rename_map = {"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "volume": "Volume"}
        d = d.rename(columns=rename_map)
        if "date" in d.columns:
            d.index = pd.to_datetime(d["date"])
        else:
            d.index = pd.to_datetime(d.index)
        d["ticker"] = ticker
        raw_dict[ticker] = d

    panel = build_panel(raw_dict, min_history_days=cfg["model"]["min_history_days"])
    log.info("Panel: %s", panel.shape)

    # ── Step 3: Features + alerts + labels ──────────────────────────────────
    # Fetch index for regime features
    try:
        import yfinance as yf
        idx = yf.download("^STOXX50E", start="2010-01-01", auto_adjust=True, progress=False)
        idx.columns = [c[0] if isinstance(c, tuple) else c for c in idx.columns]
        index_close = idx["Close"]
        index_close.index = pd.to_datetime(index_close.index)
    except Exception:
        log.warning("Could not fetch index — regime features disabled")
        index_close = None

    from src.features.engineering import build_features
    from src.features.labels import compute_forward_returns, assign_labels
    from src.alerts.engine import run_alert_engine
    from src.features.engineering import add_alert_features

    feat_panel = build_features(panel, index_close=index_close)
    feat_panel = compute_forward_returns(feat_panel, horizons=[1, 3, 5])

    events = run_alert_engine(panel)
    events = add_alert_features(events, panel)
    labeled = assign_labels(events, feat_panel, horizons=[1, 3, 5], theta=0.005)

    log.info("Labeled events: %d  (failure_rate_3d=%.3f)",
             len(labeled), labeled["label_failure_3d"].mean())

    # ── Step 4: Build feature matrix ─────────────────────────────────────────
    EXCLUDE = {"date","ticker","open","high","low","close","volume",
               "ret_1d_lead","fwd_ret_1d","fwd_ret_3d","fwd_ret_5d",
               "alert_name","direction","n_simultaneous_alerts","_dir_raw"}

    price_feat_cols = [c for c in feat_panel.columns if c not in EXCLUDE]
    price_feats = feat_panel[["date","ticker"] + price_feat_cols].copy()
    price_feats["date"] = pd.to_datetime(price_feats["date"])

    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled["_dir_raw"] = labeled["direction"]
    labeled = pd.get_dummies(labeled, columns=["direction","alert_name"], drop_first=False)

    merged = labeled.merge(
        price_feats.drop_duplicates(["date","ticker"]),
        on=["date","ticker"], how="left", suffixes=("","_feat")
    )
    merged = merged.reset_index(drop=True)

    feat_cols = (
        [c for c in merged.columns if c in price_feat_cols]
        + [c for c in merged.columns if c.startswith("direction_") or c.startswith("alert_name_")]
        + ["n_simultaneous_alerts"]
    )
    feat_cols = [c for c in feat_cols if merged[c].dtype != object]

    label_col = "label_failure_3d"
    valid = merged[label_col].notna()
    X = merged.loc[valid, feat_cols].fillna(-1)
    y = merged.loc[valid, label_col]

    log.info("Training XGBoost: %d samples, %d features, pos_rate=%.3f",
             len(y), len(feat_cols), y.mean())

    # ── Step 5: Train model ──────────────────────────────────────────────────
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        scale_pos_weight=spw,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    log.info("Validation ROC-AUC: %.4f", auc)

    # ── Step 6: Save model and feature cols ──────────────────────────────────
    model_dir = Path(cfg["model"]["path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, cfg["model"]["path"])
    with open(cfg["model"]["feature_cols_path"], "w") as f:
        json.dump(feat_cols, f)

    log.info("Model saved: %s", cfg["model"]["path"])
    log.info("Feature cols saved: %s", cfg["model"]["feature_cols_path"])
    log.info("Bootstrap complete. Ready to run: python run_agent.py --paper")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap XGBoost model for the trading agent")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--yfinance", action="store_true",
                      help="Use yfinance (no IBKR required) — recommended for first run")
    mode.add_argument("--paper", action="store_true",
                      help="Use IBKR paper trading port (4002)")
    mode.add_argument("--live", action="store_true",
                      help="Use IBKR live trading port (4001)")
    args = parser.parse_args()

    if args.yfinance:
        main(use_yfinance=True)
    elif args.live:
        main(paper=False, use_yfinance=False)
    else:
        # Default: paper (with yfinance fallback on connect failure)
        main(paper=True, use_yfinance=False)
