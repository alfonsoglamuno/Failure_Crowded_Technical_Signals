"""
Bootstrap script — fetches historical data, runs the full research pipeline,
and saves trained XGBoost models ready for the agent.

Model variants are defined by two axes:
  horizon  : 1d / 3d / 5d  — forward return window used for labels
  mode     : longonly / both — which signal directions are included in training
               longonly  trains only on bearish alerts  (FADE → BUY)
               both      trains on all alert directions (FADE → BUY or SELL)

Six models are trained and saved:
  xgboost_h1d_longonly.joblib   feature_cols_h1d_longonly.json
  xgboost_h3d_longonly.joblib   feature_cols_h3d_longonly.json  ← default
  xgboost_h5d_longonly.joblib   feature_cols_h5d_longonly.json
  xgboost_h1d_both.joblib       feature_cols_h1d_both.json
  xgboost_h3d_both.joblib       feature_cols_h3d_both.json
  xgboost_h5d_both.joblib       feature_cols_h5d_both.json

After training, set model.variant in configs/config.yaml to select which
model the agent uses (e.g. "h3d_longonly" for intraday-ish long-only).

Usage:
    python bootstrap_model.py --yfinance             # train all 6 variants
    python bootstrap_model.py --yfinance --variant h3d_longonly  # one variant
    python bootstrap_model.py --paper                # IBKR paper port
    python bootstrap_model.py --live                 # IBKR live port
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

# All variants: (horizon_days, mode)
ALL_VARIANTS = [
    (1, "longonly"),
    (3, "longonly"),
    (5, "longonly"),
    (1, "both"),
    (3, "both"),
    (5, "both"),
]


def variant_name(horizon: int, mode: str) -> str:
    return f"h{horizon}d_{mode}"


def variant_paths(model_dir: Path, horizon: int, mode: str) -> tuple[Path, Path]:
    name = variant_name(horizon, mode)
    return (
        model_dir / f"xgboost_{name}.joblib",
        model_dir / f"feature_cols_{name}.json",
    )


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


def _build_feature_matrix(labeled: pd.DataFrame, feat_panel: pd.DataFrame,
                           horizon: int, mode: str) -> tuple:
    """
    Build (X, y, feat_cols) for a specific training variant.

    mode = "longonly"  → only train on bearish alerts (FADE → BUY).
                         These are the signals the long-only agent actually trades.
                         Training on matching examples gives better calibration.
    mode = "both"      → train on all signal directions. Use this model when
                         allow_short=True so it sees SELL-side failure events too.
    """
    EXCLUDE = {
        "date", "ticker", "open", "high", "low", "close", "volume",
        "ret_1d_lead", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d",
        "alert_name", "direction", "n_simultaneous_alerts", "_dir_raw",
    }

    price_feat_cols = [c for c in feat_panel.columns if c not in EXCLUDE]
    price_feats = feat_panel[["date", "ticker"] + price_feat_cols].copy()
    price_feats["date"] = pd.to_datetime(price_feats["date"])

    lab = labeled.copy()
    lab["date"] = pd.to_datetime(lab["date"])

    # Direction filter
    if mode == "longonly":
        # Keep only bearish alerts (oversold/breakdown) → FADE produces BUY entries.
        # Also keep neutral for context; exclude bullish-only (those are FOLLOW=BUY
        # or FADE=SELL which long-only never trades).
        lab = lab[lab["direction"].isin(["bearish", "neutral"])].copy()
        log.info("  [longonly] filtered to bearish+neutral: %d events", len(lab))

    lab["_dir_raw"] = lab["direction"]
    lab = pd.get_dummies(lab, columns=["direction", "alert_name"], drop_first=False)

    merged = lab.merge(
        price_feats.drop_duplicates(["date", "ticker"]),
        on=["date", "ticker"], how="left", suffixes=("", "_feat"),
    )
    merged = merged.reset_index(drop=True)

    feat_cols = (
        [c for c in merged.columns if c in price_feat_cols]
        + [c for c in merged.columns
           if c.startswith("direction_") or c.startswith("alert_name_")]
        + ["n_simultaneous_alerts"]
    )
    feat_cols = [c for c in feat_cols if merged[c].dtype != object]

    label_col = f"label_failure_{horizon}d"
    if label_col not in merged.columns:
        log.error("Label column %s not found — skipping", label_col)
        return None, None, None

    valid = merged[label_col].notna()
    X = merged.loc[valid, feat_cols].fillna(-1)
    y = merged.loc[valid, label_col]
    dates = pd.to_datetime(merged.loc[valid, "date"])

    return X, y, feat_cols, dates


def _make_sample_weights(dates: pd.Series, halflife_days: int = 252) -> np.ndarray:
    """
    Exponential recency weights: recent samples get weight 1.0, older samples decay.

    Half-life = number of calendar days at which weight halves.
    Default 252 ≈ 1 trading year — data one year old carries ~50% weight.

    Rationale (Jegadeesh & Titman 1993, Lo & MacKinlay 1988):
      Financial regimes shift over time; stale data can bias the model toward
      conditions that no longer hold. Exponential weighting is a principled
      way to keep long history (for rare events) while emphasising recent structure.
    """
    ts = pd.to_datetime(dates).values.astype("datetime64[D]").astype(float)
    age_days = ts.max() - ts        # 0 = most recent
    decay = np.log(2) / halflife_days
    weights = np.exp(-decay * age_days)
    return weights / weights.sum() * len(weights)   # normalise to mean = 1


def _train_one_variant(X, y, feat_cols: list, horizon: int, mode: str,
                       model_dir: Path, dates: pd.Series | None = None,
                       halflife_days: int = 252) -> None:
    """Train and save one XGBoost model variant."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, precision_score

    name = variant_name(horizon, mode)
    log.info("=" * 60)
    log.info("Training variant: %s  |  %d samples  |  %d features  |  pos=%.3f",
             name, len(y), len(feat_cols), y.mean())

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    # Recency sample weights — give more influence to recent data
    sample_weight = None
    if dates is not None:
        w = _make_sample_weights(dates, halflife_days=halflife_days)
        sample_weight = w[:split_idx]
        log.info("  Sample weights: halflife=%dd  min=%.3f  max=%.3f",
                 halflife_days, sample_weight.min(), sample_weight.max())

    # h1d: lighter model — horizon is very short, fewer trees needed to avoid overfit
    # h5d: deeper trees — more regime context captured at longer horizons
    max_depth    = {1: 4, 3: 5, 5: 6}.get(horizon, 5)
    n_estimators = {1: 600, 3: 1000, 5: 800}.get(horizon, 1000)

    model = XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        max_depth=max_depth,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    log.info("  ROC-AUC: %.4f  (best_iter=%d)", auc, model.best_iteration)

    for t in [0.55, 0.60, 0.65, 0.70]:
        pred = (val_proba >= t).astype(int)
        n = int(pred.sum())
        if n > 0:
            prec = precision_score(y_val, pred, zero_division=0)
            log.info("    thr=%.2f  signals=%4d  precision=%.3f", t, n, prec)

    model_path, cols_path = variant_paths(model_dir, horizon, mode)
    joblib.dump(model, model_path)
    with open(cols_path, "w") as f:
        json.dump(feat_cols, f)
    log.info("  Saved: %s", model_path.name)

    # Feature importance top-10
    importances = model.feature_importances_
    top10 = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:10]
    log.info("  Top-10 features: %s",
             "  ".join(f"{n}={v:.3f}" for n, v in top10))


def main(paper: bool = True, use_yfinance: bool = False,
         target_variant: str | None = None):
    # ── Load config ─────────────────────────────────────────────────────────
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/ibkr_contracts.yaml") as f:
        contracts_cfg = yaml.safe_load(f)["contracts"]

    parent_tickers_file = cfg["universe"]["parent_tickers_file"]
    with open(parent_tickers_file) as f:
        tickers = yaml.safe_load(f)["tickers"]

    # Determine which variants to train
    if target_variant:
        parts = target_variant.split("_", 1)
        if len(parts) != 2 or not parts[0].startswith("h"):
            log.error("Invalid variant format. Use e.g. 'h3d_longonly' or 'h1d_both'")
            sys.exit(1)
        horizon = int(parts[0][1:-1])
        mode = parts[1]
        variants_to_train = [(horizon, mode)]
    else:
        variants_to_train = ALL_VARIANTS

    log.info("Bootstrap starting. Universe: %d tickers. yfinance=%s  variants=%s",
             len(tickers), use_yfinance,
             [variant_name(h, m) for h, m in variants_to_train])

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

    # ── Step 3: Features + alerts + labels ───────────────────────────────────
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

    log.info("Building features (this may take 1-2 minutes for 2yr / 49 tickers)...")
    feat_panel = build_features(panel, index_close=index_close)
    feat_panel = compute_forward_returns(feat_panel, horizons=[1, 3, 5])

    events = run_alert_engine(panel)
    events = add_alert_features(events, panel)
    labeled = assign_labels(events, feat_panel, horizons=[1, 3, 5], theta=0.005)

    log.info("Labeled events: %d  failure_rate 1d=%.3f  3d=%.3f  5d=%.3f",
             len(labeled),
             labeled["label_failure_1d"].mean(),
             labeled["label_failure_3d"].mean(),
             labeled["label_failure_5d"].mean())

    # ── Step 4–5: Train all requested variants ────────────────────────────────
    model_dir = Path(cfg["model"]["path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    halflife = cfg["model"].get("sample_weight_halflife_days", 252)

    for horizon, mode in variants_to_train:
        X, y, feat_cols, dates = _build_feature_matrix(labeled, feat_panel, horizon, mode)
        if X is None:
            log.warning("Skipping variant %s — feature matrix build failed",
                        variant_name(horizon, mode))
            continue
        _train_one_variant(X, y, feat_cols, horizon, mode, model_dir,
                           dates=dates, halflife_days=halflife)

    # ── Step 6: Report summary ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("All variants trained. Files in %s:", model_dir)
    for f in sorted(model_dir.glob("xgboost_*.joblib")):
        log.info("  %s", f.name)

    # Show current active variant from config
    active = cfg["model"].get("variant", "h3d_longonly")
    active_path, active_cols = variant_paths(model_dir, *_parse_variant(active))
    if active_path.exists():
        log.info("")
        log.info("Active variant in config: %s", active)
        log.info("  model     : %s", active_path)
        log.info("  feat cols : %s", active_cols)
    else:
        log.warning("Active variant '%s' not found — update model.variant in config.yaml", active)

    log.info("")
    log.info("To change active model, set model.variant in configs/config.yaml, then:")
    log.info("  python run_agent.py --paper")


def _parse_variant(variant: str) -> tuple[int, str]:
    """Parse 'h3d_longonly' → (3, 'longonly')."""
    parts = variant.split("_", 1)
    return int(parts[0][1:-1]), parts[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap XGBoost models for the trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variants trained (horizon × direction mode):
  h1d_longonly  h3d_longonly*  h5d_longonly
  h1d_both      h3d_both       h5d_both

  *default active variant (set via model.variant in config.yaml)

longonly: train on bearish alerts only → FADE=BUY. Use when allow_short=false.
both:     train on all directions      → FADE=BUY or SELL. Use when allow_short=true.
""",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--yfinance", action="store_true",
                     help="Use yfinance (no IBKR required) — recommended")
    src.add_argument("--paper", action="store_true",
                     help="Use IBKR paper trading port (4002)")
    src.add_argument("--live", action="store_true",
                     help="Use IBKR live trading port (4001)")
    parser.add_argument(
        "--variant", metavar="VARIANT",
        help="Train only one variant e.g. h3d_longonly (default: train all 6)"
    )
    args = parser.parse_args()

    if args.yfinance:
        main(use_yfinance=True, target_variant=args.variant)
    elif args.live:
        main(paper=False, use_yfinance=False, target_variant=args.variant)
    else:
        main(paper=True, use_yfinance=False, target_variant=args.variant)
