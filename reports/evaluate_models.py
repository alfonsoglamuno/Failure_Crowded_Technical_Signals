"""
Model evaluation script — runs validation on all trained variants and
writes a summary report to reports/results/.

Core question answered:
    Given a visible chart alert, what is the probability this alert will *fail*
    (i.e. the expected move does not materialise) over the next 1-5 sessions?

For each variant (h1d/h3d/h5d x longonly/both), this script measures:
  - ROC-AUC: overall discrimination power
  - Precision at 0.55 / 0.60 / 0.65 / 0.70: how accurate are the high-confidence calls?
  - Recall at each threshold: fraction of true failures captured
  - Break-even analysis: minimum hit rate to cover commissions at default position size
  - Calibration: are predicted probabilities actually close to observed failure rates?

Usage:
    cd <repo_root>
    python reports/evaluate_models.py                  # evaluate all variants
    python reports/evaluate_models.py --variant h1d_longonly   # one variant only
    python reports/evaluate_models.py --output reports/results/latest.csv

The script always uses held-out data (last 20% of the chronological sample) —
never the training split — to avoid inflated estimates.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
AGENT = ROOT / "trading_agent"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

VARIANTS = [
    ("h1d_longonly", 1, "longonly"),
    ("h3d_longonly", 3, "longonly"),
    ("h5d_longonly", 5, "longonly"),
    ("h1d_both",     1, "both"),
    ("h3d_both",     3, "both"),
    ("h5d_both",     5, "both"),
]

THRESHOLDS = [0.55, 0.60, 0.65, 0.70]

# Break-even: minimum precision so that E[PnL] > 0 given SL/TP ratio.
# With SL=1.5%, TP=2.5%, commission ~0.10% round-trip:
#   win: +2.5% - 0.10% = +2.40%
#   loss: -1.5% - 0.10% = -1.60%
#   break_even = loss / (win + loss) = 1.60 / (2.40 + 1.60) = 40.0%
BREAKEVEN_PRECISION = 0.40


def _load_config() -> dict:
    with open(AGENT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _build_dataset(cfg: dict, horizon: int, mode: str) -> tuple:
    """Build the full (X, y, dates) feature matrix for a variant."""
    from src.data.preprocess import build_panel
    from src.features.engineering import build_features, add_alert_features
    from src.features.labels import compute_forward_returns, assign_labels
    from src.alerts.engine import run_alert_engine
    import yfinance as yf

    with open(cfg["universe"]["parent_tickers_file"]) as f:
        tickers = yaml.safe_load(f)["tickers"]

    cache_dir  = AGENT / cfg["data"]["cache_dir"]
    start_date = (datetime.now() - __import__("datetime").timedelta(
        days=cfg["data"]["history_days"] + 10)).strftime("%Y-%m-%d")

    # Fetch OHLCV
    results = {}
    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('/', '_')}.parquet"
        if cache_file.exists():
            results[ticker] = pd.read_parquet(cache_file)
            continue
        raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df[df["close"] > 0].reset_index()
        df.rename(columns={"index": "date", "Date": "date"}, inplace=True)
        df["ticker"] = ticker
        results[ticker] = df

    raw_dict = {}
    for ticker, df in results.items():
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"})
        d.index = pd.to_datetime(d.get("date", d.index))
        d["ticker"] = ticker
        raw_dict[ticker] = d

    panel = build_panel(raw_dict, min_history_days=cfg["model"]["min_history_days"])

    try:
        idx = yf.download("^STOXX50E", start="2010-01-01", auto_adjust=True, progress=False)
        idx.columns = [c[0] if isinstance(c, tuple) else c for c in idx.columns]
        index_close = idx["Close"]
        index_close.index = pd.to_datetime(index_close.index)
    except Exception:
        index_close = None

    feat_panel = build_features(panel, index_close=index_close)
    feat_panel = compute_forward_returns(feat_panel, horizons=[horizon])

    events = run_alert_engine(panel)
    events = add_alert_features(events, panel)
    labeled = assign_labels(events, feat_panel, horizons=[horizon], theta=0.005)

    EXCLUDE = {"date", "ticker", "open", "high", "low", "close", "volume",
               "ret_1d_lead", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d",
               "alert_name", "direction", "n_simultaneous_alerts", "_dir_raw"}

    feat_cols_all = [c for c in feat_panel.columns if c not in EXCLUDE]
    price_feats   = feat_panel[["date", "ticker"] + feat_cols_all].copy()
    price_feats["date"] = pd.to_datetime(price_feats["date"])

    lab = labeled.copy()
    lab["date"] = pd.to_datetime(lab["date"])

    if mode == "longonly":
        lab = lab[lab["direction"].isin(["bearish", "neutral"])].copy()

    lab["_dir_raw"] = lab["direction"]
    lab = pd.get_dummies(lab, columns=["direction", "alert_name"], drop_first=False)
    merged = lab.merge(
        price_feats.drop_duplicates(["date", "ticker"]),
        on=["date", "ticker"], how="left",
    )

    feat_cols = (
        [c for c in merged.columns if c in feat_cols_all]
        + [c for c in merged.columns if c.startswith("direction_") or c.startswith("alert_name_")]
        + ["n_simultaneous_alerts"]
    )
    feat_cols = [c for c in feat_cols if merged[c].dtype != object]

    label_col = f"label_failure_{horizon}d"
    if label_col not in merged.columns:
        return None, None, None, None

    valid  = merged[label_col].notna()
    X      = merged.loc[valid, feat_cols].fillna(-1)
    y      = merged.loc[valid, label_col]
    dates  = pd.to_datetime(merged.loc[valid, "date"])
    return X, y, dates, feat_cols


def evaluate_variant(name: str, horizon: int, mode: str, cfg: dict) -> dict | None:
    """Evaluate one variant on its chronological hold-out split."""
    import joblib
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    model_dir  = AGENT / cfg["model"]["path"].replace("\\", "/").rsplit("/", 1)[0]
    model_path = model_dir / f"xgboost_{name}.joblib"
    cols_path  = model_dir / f"feature_cols_{name}.json"

    if not model_path.exists():
        print(f"  [SKIP] {name}: model file not found at {model_path}")
        return None

    model    = joblib.load(model_path)
    with open(cols_path) as f:
        feat_cols_trained = json.load(f)

    print(f"  Building dataset for {name}...")
    X, y, dates, feat_cols_built = _build_dataset(cfg, horizon, mode)
    if X is None:
        print(f"  [SKIP] {name}: could not build dataset")
        return None

    # Align columns — use only features the model was trained on
    common = [c for c in feat_cols_trained if c in X.columns]
    X = X[common].fillna(-1)

    # Strict chronological split — last 20% is hold-out
    split = int(len(X) * 0.80)
    X_val, y_val = X.iloc[split:], y.iloc[split:]
    dates_val    = dates.iloc[split:]

    if len(y_val) < 50:
        print(f"  [SKIP] {name}: insufficient validation samples ({len(y_val)})")
        return None

    probas = model.predict_proba(X_val)[:, 1]
    auc    = roc_auc_score(y_val, probas)

    result = {
        "variant":           name,
        "horizon_days":      horizon,
        "mode":              mode,
        "n_val_samples":     len(y_val),
        "failure_rate":      float(y_val.mean()),
        "auc":               round(auc, 4),
        "val_period_start":  str(dates_val.min().date()),
        "val_period_end":    str(dates_val.max().date()),
    }

    for thr in THRESHOLDS:
        pred = (probas >= thr).astype(int)
        n    = int(pred.sum())
        if n > 5:
            prec   = float(precision_score(y_val, pred, zero_division=0))
            recall = float(recall_score(y_val, pred, zero_division=0))
            above_breakeven = prec >= BREAKEVEN_PRECISION
        else:
            prec = recall = float("nan")
            above_breakeven = False
        result[f"signals_at_{thr}"]        = n
        result[f"precision_at_{thr}"]      = round(prec, 4) if not np.isnan(prec) else None
        result[f"recall_at_{thr}"]         = round(recall, 4) if not np.isnan(recall) else None
        result[f"profitable_at_{thr}"]     = above_breakeven

    # Calibration: mean predicted proba vs observed failure rate per decile
    df_cal = pd.DataFrame({"proba": probas, "label": y_val.values})
    df_cal["decile"] = pd.qcut(df_cal["proba"], 10, labels=False, duplicates="drop")
    cal = df_cal.groupby("decile").agg(mean_pred=("proba", "mean"),
                                        obs_rate=("label", "mean")).reset_index()
    cal_error = float(((cal["mean_pred"] - cal["obs_rate"]).abs()).mean())
    result["calibration_mae"] = round(cal_error, 4)

    return result


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("  MODEL EVALUATION SUMMARY — Predicts when technical alerts FAIL and the reversal is tradeable")
    print("=" * 100)
    print(f"  Break-even precision (SL=1.5%, TP=2.5%, comm=0.10% rt): {BREAKEVEN_PRECISION:.0%}")
    print()

    header = (
        f"  {'Variant':<20s} {'AUC':>6s} {'N_val':>6s} {'Fail%':>6s} "
        f"{'P@0.55':>7s} {'P@0.60':>7s} {'P@0.65':>7s} {'P@0.70':>7s} "
        f"{'Cal_MAE':>8s}  Period"
    )
    print(header)
    print("  " + "-" * 98)

    for r in sorted(results, key=lambda x: (-x["auc"])):
        p = lambda k: f"{r.get(k, float('nan')):.1%}" if r.get(k) is not None else "  —  "
        profitable = lambda t: "*" if r.get(f"profitable_at_{t}") else " "

        print(
            f"  {r['variant']:<20s} {r['auc']:>6.3f} {r['n_val_samples']:>6d} "
            f"{r['failure_rate']:>5.1%}  "
            f"{p('precision_at_0.55'):>6s}{profitable(0.55)} "
            f"{p('precision_at_0.60'):>6s}{profitable(0.60)} "
            f"{p('precision_at_0.65'):>6s}{profitable(0.65)} "
            f"{p('precision_at_0.70'):>6s}{profitable(0.70)} "
            f"{r['calibration_mae']:>8.4f}  "
            f"{r['val_period_start']} / {r['val_period_end']}"
        )

    print()
    print("  * = precision >= break-even threshold")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model variants")
    parser.add_argument("--variant", metavar="VARIANT",
                        help="Evaluate a single variant (e.g. h1d_longonly)")
    parser.add_argument("--output", metavar="PATH",
                        help="Save results CSV to this path")
    args = parser.parse_args()

    cfg = _load_config()

    variants_to_run = VARIANTS
    if args.variant:
        variants_to_run = [(v, h, m) for v, h, m in VARIANTS if v == args.variant]
        if not variants_to_run:
            print(f"Unknown variant '{args.variant}'. Choose from: {[v for v,_,_ in VARIANTS]}")
            sys.exit(1)

    print(f"\nEvaluating {len(variants_to_run)} variant(s)...\n")
    results = []
    for name, horizon, mode in variants_to_run:
        print(f"[{name}]")
        r = evaluate_variant(name, horizon, mode, cfg)
        if r:
            results.append(r)

    if not results:
        print("No results — are models trained? Run: python trading_agent/bootstrap_model.py --yfinance")
        return

    print_summary(results)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path  = Path(args.output) if args.output else out_dir / f"eval_{timestamp}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
