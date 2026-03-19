"""
Standalone research script -- backtests all (alert_type, signal_mode, horizon)
combinations to find what actually works historically.

Signal modes tested:
  Long-only (all entries are BUY):
    follow_bullish : fire on bullish alert, go long (momentum)
    fade_bearish   : fire on bearish alert, go long expecting reversal (contrarian BUY)

  Short-enabled (entries are SELL / short position):
    follow_bearish : fire on bearish alert, go short (momentum short)
    fade_bullish   : fire on bullish alert, go short expecting reversal (contrarian SHORT)

For each combination the script computes in-sample (first 70% of dates) and
out-of-sample (last 30% of dates) statistics:
  n_trades, win_rate, mean_ret, std_ret, sharpe (annualized), total_ret, max_drawdown

Commission: 0.2% round-trip (0.002) deducted from every trade gross return.

Output:
  results/strategy_optimization.csv  -- full table ranked by OOS Sharpe
  Console                            -- top-20 by OOS Sharpe + recommended strategy

Usage:
    python optimize_strategy.py --yfinance
    python optimize_strategy.py --yfinance --min-trades 10 --commission 0.002
    python optimize_strategy.py --yfinance --allow-short   # include short strategies
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZONS = [1, 2, 3, 5, 10]

# Long-only modes (all entries are BUY)
SIGNAL_MODES_LONG  = ["follow_bullish", "fade_bearish"]
# Short-side modes (entries are SELL -- requires allow_short=True in agent)
SIGNAL_MODES_SHORT = ["follow_bearish", "fade_bullish"]

COMMISSION = 0.002   # 0.2% round-trip


# -- Data helpers (copied from bootstrap_model.py) ----------------------------

def _fetch_yfinance(tickers: list, days: int, cache_dir: str | None) -> dict:
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


# -- Statistics helper ---------------------------------------------------------

def _compute_stats(net_returns: pd.Series, horizon: int) -> dict:
    """Compute trade-level statistics for a series of net returns."""
    n = len(net_returns)
    if n == 0:
        return dict(n_trades=0, win_rate=np.nan, mean_ret=np.nan,
                    std_ret=np.nan, sharpe=np.nan, total_ret=np.nan,
                    max_drawdown=np.nan)

    win_rate  = float((net_returns > 0).mean())
    mean_ret  = float(net_returns.mean())
    std_ret   = float(net_returns.std(ddof=1)) if n > 1 else np.nan
    total_ret = float((1 + net_returns).prod() - 1)

    # Annualised Sharpe: each trade holds for `horizon` days
    # Expected trades per year = 252 / horizon
    if std_ret and std_ret > 0:
        sharpe = mean_ret / std_ret * np.sqrt(252 / horizon)
    else:
        sharpe = np.nan

    # Max drawdown on cumulative equity curve
    cum = (1 + net_returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    return dict(
        n_trades=n,
        win_rate=win_rate,
        mean_ret=mean_ret,
        std_ret=std_ret,
        sharpe=sharpe,
        total_ret=total_ret,
        max_drawdown=max_drawdown,
    )


def _compute_benchmark_stats(index_close: pd.Series,
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp) -> dict:
    """
    Compute buy & hold stats for STOXX50E over [start_date, end_date].

    Returns {total_ret, annualized_ret, sharpe}.
    """
    if index_close is None or index_close.empty:
        return {"total_ret": np.nan, "annualized_ret": np.nan, "sharpe": np.nan}

    idx = index_close.copy()
    idx.index = pd.to_datetime(idx.index)
    mask = (idx.index >= start_date) & (idx.index <= end_date)
    sub = idx[mask].dropna()

    if len(sub) < 2:
        return {"total_ret": np.nan, "annualized_ret": np.nan, "sharpe": np.nan}

    daily_ret = sub.pct_change().dropna()
    total_ret = float((1 + daily_ret).prod() - 1)
    n_days = len(daily_ret)
    trading_days_per_year = 252
    annualized_ret = float((1 + total_ret) ** (trading_days_per_year / n_days) - 1) if n_days > 0 else np.nan

    if daily_ret.std(ddof=1) > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(trading_days_per_year))
    else:
        sharpe = np.nan

    return {"total_ret": total_ret, "annualized_ret": annualized_ret, "sharpe": sharpe}


def _run_backtest_split(labeled: pd.DataFrame, alert_type: str,
                        signal_mode: str, horizon: int,
                        commission: float) -> tuple[dict, dict, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Run one (alert_type, signal_mode, horizon) combo.

    Walk-forward split: sort events by date, split at the 70th percentile date.

    Returns (is_stats, oos_stats, oos_start_date, oos_end_date).
    """
    fwd_col = f"fwd_ret_{horizon}d"
    if fwd_col not in labeled.columns:
        return {}, {}, None, None

    # Filter to the requested alert type
    sub = labeled[labeled["alert_name"] == alert_type].copy()
    if sub.empty:
        return (_compute_stats(pd.Series([], dtype=float), horizon),
                _compute_stats(pd.Series([], dtype=float), horizon),
                None, None)

    sub = sub.sort_values("date").reset_index(drop=True)
    sub["date"] = pd.to_datetime(sub["date"])

    # Compute trade_return based on signal mode.
    # Long-only modes: positive return = BUY profit.
    # Short modes: positive return = SHORT profit (short sells, profits when price falls).
    if signal_mode == "follow_bullish":
        sub = sub[sub["direction"] == "bullish"].copy()
        sub["trade_return"] = sub[fwd_col]            # BUY momentum: profit if price rises
    elif signal_mode == "fade_bearish":
        sub = sub[sub["direction"] == "bearish"].copy()
        sub["trade_return"] = -sub[fwd_col]           # BUY contrarian: profit if price reverses up
    elif signal_mode == "follow_bearish":
        sub = sub[sub["direction"] == "bearish"].copy()
        sub["trade_return"] = -sub[fwd_col]           # SHORT momentum: profit if price falls
    elif signal_mode == "fade_bullish":
        sub = sub[sub["direction"] == "bullish"].copy()
        sub["trade_return"] = -sub[fwd_col]           # SHORT contrarian: profit if bullish alert fails (price falls)
    else:
        return {}, {}, None, None

    sub = sub.dropna(subset=["trade_return"]).reset_index(drop=True)
    if sub.empty:
        return (_compute_stats(pd.Series([], dtype=float), horizon),
                _compute_stats(pd.Series([], dtype=float), horizon),
                None, None)

    # Walk-forward split at 70th percentile date
    split_date = sub["date"].quantile(0.70)
    is_mask  = sub["date"] <= split_date
    oos_mask = sub["date"] >  split_date

    is_ret  = (sub.loc[is_mask,  "trade_return"] - commission).reset_index(drop=True)
    oos_ret = (sub.loc[oos_mask, "trade_return"] - commission).reset_index(drop=True)

    oos_dates = sub.loc[oos_mask, "date"]
    oos_start = oos_dates.min() if not oos_dates.empty else None
    oos_end   = oos_dates.max() if not oos_dates.empty else None

    return _compute_stats(is_ret, horizon), _compute_stats(oos_ret, horizon), oos_start, oos_end


# -- Main ----------------------------------------------------------------------

def main(use_yfinance: bool = True, min_trades: int = 10, commission: float = COMMISSION,
         allow_short: bool = False, top_n: int = 5):
    # -- Load config (same pattern as bootstrap_model.py) ---------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/ibkr_contracts.yaml") as f:
        contracts_cfg = yaml.safe_load(f)["contracts"]

    parent_tickers_file = cfg["universe"]["parent_tickers_file"]
    with open(parent_tickers_file) as f:
        tickers = yaml.safe_load(f)["tickers"]

    cache_dir    = cfg["data"]["cache_dir"]
    history_days = cfg["data"]["history_days"]

    log.info("Strategy optimization starting. Universe: %d tickers. yfinance=%s",
             len(tickers), use_yfinance)

    # -- Step 1: Fetch data ----------------------------------------------------
    if use_yfinance:
        universe_data = _fetch_yfinance(tickers, history_days, cache_dir)
    else:
        from agent.data_feed import IBKRFeed
        feed = IBKRFeed(cfg, paper=True)
        ibkr_ok = False
        try:
            feed.connect()
            ibkr_ok = True
        except ConnectionError as e:
            log.warning("IBKR connect failed (%s). Falling back to yfinance.", e)
        if ibkr_ok:
            universe_data = feed.fetch_universe(tickers, contracts_cfg,
                                                days=history_days, cache_dir=cache_dir)
            feed.disconnect()
        else:
            universe_data = _fetch_yfinance(tickers, history_days, cache_dir)

    log.info("Fetched data for %d tickers", len(universe_data))
    if not universe_data:
        log.error("No data fetched. Check connectivity and ticker list.")
        sys.exit(1)

    # -- Step 2: Build panel ---------------------------------------------------
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

    # -- Step 3: Features + alerts + labels ------------------------------------
    try:
        import yfinance as yf
        idx = yf.download("^STOXX50E", start="2010-01-01", auto_adjust=True, progress=False)
        idx.columns = [c[0] if isinstance(c, tuple) else c for c in idx.columns]
        index_close = idx["Close"]
        index_close.index = pd.to_datetime(index_close.index)
    except Exception:
        log.warning("Could not fetch index -- regime features disabled")
        index_close = None

    from src.features.engineering import build_features, add_alert_features
    from src.features.labels import compute_forward_returns, assign_labels
    from src.alerts.engine import run_alert_engine

    log.info("Building features...")
    feat_panel = build_features(panel, index_close=index_close)
    feat_panel = compute_forward_returns(feat_panel, horizons=HORIZONS)

    events = run_alert_engine(panel)
    events = add_alert_features(events, panel)
    labeled = assign_labels(events, feat_panel, horizons=HORIZONS, theta=0.005)

    log.info("Labeled events: %d", len(labeled))

    # -- Step 4: Enumerate all combinations -----------------------------------
    signal_modes = SIGNAL_MODES_LONG + (SIGNAL_MODES_SHORT if allow_short else [])
    alert_types  = sorted(labeled["alert_name"].dropna().unique())
    log.info("Alert types found: %d  Horizons: %s  Signal modes: %s",
             len(alert_types), HORIZONS, signal_modes)

    rows = []
    total = len(alert_types) * len(signal_modes) * len(HORIZONS)
    done  = 0
    # Track the OOS period across all combos for benchmark comparison
    global_oos_start: pd.Timestamp | None = None
    global_oos_end:   pd.Timestamp | None = None

    for alert_type in alert_types:
        for signal_mode in signal_modes:
            for horizon in HORIZONS:
                is_stats, oos_stats, oos_start, oos_end = _run_backtest_split(
                    labeled, alert_type, signal_mode, horizon, commission
                )
                if not is_stats or not oos_stats:
                    done += 1
                    continue

                # Track global OOS date range for benchmark
                if oos_start is not None:
                    if global_oos_start is None or oos_start < global_oos_start:
                        global_oos_start = oos_start
                if oos_end is not None:
                    if global_oos_end is None or oos_end > global_oos_end:
                        global_oos_end = oos_end

                # Compute benchmark return for this OOS window
                bm = _compute_benchmark_stats(index_close, oos_start, oos_end) if oos_start else {}

                row = {
                    "alert_type":    alert_type,
                    "signal_mode":   signal_mode,
                    "requires_short": signal_mode in SIGNAL_MODES_SHORT,
                    "horizon":       horizon,
                    # In-sample
                    "n_is_trades":       is_stats["n_trades"],
                    "is_win_rate":       is_stats["win_rate"],
                    "is_mean_ret":       is_stats["mean_ret"],
                    "is_std_ret":        is_stats["std_ret"],
                    "is_sharpe":         is_stats["sharpe"],
                    "is_total_ret":      is_stats["total_ret"],
                    "is_max_drawdown":   is_stats["max_drawdown"],
                    # Out-of-sample
                    "n_oos_trades":      oos_stats["n_trades"],
                    "oos_win_rate":      oos_stats["win_rate"],
                    "oos_mean_ret":      oos_stats["mean_ret"],
                    "oos_std_ret":       oos_stats["std_ret"],
                    "oos_sharpe":        oos_stats["sharpe"],
                    "oos_total_ret":     oos_stats["total_ret"],
                    "oos_max_drawdown":  oos_stats["max_drawdown"],
                    # Benchmark comparison
                    "benchmark_total_ret": bm.get("total_ret", np.nan),
                    "alpha_vs_benchmark":  (oos_stats["total_ret"] - bm["total_ret"])
                                           if not np.isnan(bm.get("total_ret", np.nan)) else np.nan,
                }
                rows.append(row)
                done += 1
                if done % 50 == 0:
                    log.info("  Progress: %d/%d combos", done, total)

    results = pd.DataFrame(rows)
    if results.empty:
        log.error("No results -- check data and alert engine output.")
        sys.exit(1)

    results = results.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)

    # -- Step 5: Save results --------------------------------------------------
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strategy_optimization.csv"
    results.to_csv(out_path, index=False, float_format="%.5f")
    log.info("Saved full results to %s", out_path)

    # -- Step 6: Benchmark stats -----------------------------------------------
    bm_global = _compute_benchmark_stats(index_close, global_oos_start, global_oos_end)
    oos_start_str = str(global_oos_start.date()) if global_oos_start is not None else "n/a"
    oos_end_str   = str(global_oos_end.date())   if global_oos_end   is not None else "n/a"
    print("\n" + "=" * 100)
    print(f"  STOXX50E BUY & HOLD -- OOS period: {oos_start_str} to {oos_end_str}")
    if not np.isnan(bm_global.get("total_ret", np.nan)):
        print(f"    total_ret={bm_global['total_ret']:+.1%}  "
              f"annualized={bm_global['annualized_ret']:+.1%}  "
              f"sharpe={bm_global['sharpe']:+.3f}")
    else:
        print("    (STOXX50E data unavailable -- benchmark comparison disabled)")
    print("=" * 100)

    # -- Print top-20 ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("  TOP-20 COMBINATIONS BY OOS SHARPE")
    print("=" * 100)
    top20 = results.head(20)
    col_fmt = (
        f"{'#':>3}  {'alert_type':<30}  {'mode':<16}  {'h':>2}  "
        f"{'n_oos':>6}  {'oos_wr':>7}  {'oos_mu':>8}  "
        f"{'oos_sr':>8}  {'oos_tot':>9}  {'alpha':>8}  {'oos_dd':>9}"
    )
    print(col_fmt)
    print("-" * 100)
    for rank, (_, r) in enumerate(top20.iterrows(), 1):
        wr    = f"{r['oos_win_rate']:.2%}" if pd.notna(r['oos_win_rate']) else "  n/a  "
        mu    = f"{r['oos_mean_ret']:+.4f}"  if pd.notna(r['oos_mean_ret']) else "    n/a "
        sr    = f"{r['oos_sharpe']:+.3f}"   if pd.notna(r['oos_sharpe'])   else "   n/a "
        tot   = f"{r['oos_total_ret']:+.3f}" if pd.notna(r['oos_total_ret']) else "    n/a"
        alpha = f"{r.get('alpha_vs_benchmark', np.nan):+.3f}" if pd.notna(r.get('alpha_vs_benchmark', np.nan)) else "   n/a"
        dd    = f"{r['oos_max_drawdown']:+.3f}" if pd.notna(r['oos_max_drawdown']) else "    n/a"
        print(
            f"{rank:>3}  {r['alert_type']:<30}  {r['signal_mode']:<16}  "
            f"{int(r['horizon']):>2}  {int(r['n_oos_trades']):>6}  "
            f"{wr:>7}  {mu:>8}  {sr:>8}  {tot:>9}  {alpha:>8}  {dd:>9}"
        )
    print("=" * 100)

    # -- Step 7: Recommended strategy -----------------------------------------
    print("\n" + "=" * 80)
    print("  RECOMMENDED STRATEGY (per variant -- longonly and both)")
    print(f"  Filter: OOS Sharpe > 0.3  AND  n_oos_trades >= {min_trades}  "
          f"Top-N alerts per horizon: {top_n}")
    print("=" * 80)

    qualified = results[
        (results["oos_sharpe"] > 0.3) &
        (results["n_oos_trades"] >= min_trades)
    ].copy()

    rec_output: dict = {}  # for strategy_recommendation.yaml

    if qualified.empty:
        print("  No combinations meet the filter criteria.")
        print("  Consider lowering --min-trades or reviewing data quality.")
    else:
        long_qual  = qualified[~qualified["requires_short"]]
        short_qual = qualified[qualified["requires_short"]]

        def _build_variant_recommendation(subset: pd.DataFrame, mode_suffix: str,
                                           label: str, top_n: int) -> dict:
            """
            Build per-variant recommendation entries for all horizons in this mode.

            Returns a dict keyed by full variant name (e.g. "h10d_longonly").
            Each value is the recommendation dict for that variant.

            mode_suffix: "longonly" or "both" -- matches bootstrap_model.py variant naming.
            """
            variant_recs: dict = {}

            if subset.empty:
                print(f"\n  [{label}] No qualifying combinations.")
                return variant_recs

            print(f"\n  [{label}] Per-horizon top-{top_n} alert types:")
            for h in sorted(subset["horizon"].unique()):
                grp = subset[subset["horizon"] == h]
                best = (
                    grp.sort_values("oos_sharpe", ascending=False)
                       .groupby("alert_type")
                       .first()
                       .reset_index()
                       .sort_values("oos_sharpe", ascending=False)
                       .head(top_n)
                )
                if best.empty:
                    continue

                horizon = int(h)
                variant_name = f"h{horizon}d_{mode_suffix}"
                alert_whitelist = sorted(best["alert_type"].tolist())
                best_row = best.iloc[0]

                print(f"\n    Horizon h{horizon}d ({variant_name}):")
                for _, r in best.iterrows():
                    bm_ref = f"  alpha={r.get('alpha_vs_benchmark', np.nan):+.3f}" if pd.notna(r.get('alpha_vs_benchmark', np.nan)) else ""
                    print(
                        f"      {r['alert_type']:<30}  mode={r['signal_mode']:<16}  "
                        f"oos_sr={r['oos_sharpe']:+.3f}  "
                        f"oos_wr={r['oos_win_rate']:.2%}  "
                        f"n={int(r['n_oos_trades'])}{bm_ref}"
                    )

                print(f"\n    [{label}] {variant_name} => alert_whitelist: {alert_whitelist}")
                print(f"      CLI: python bootstrap_model.py --yfinance --variant {variant_name} "
                      f"--alert-whitelist {','.join(alert_whitelist)}")

                bm_total = float(best_row.get("benchmark_total_ret", np.nan))
                variant_recs[variant_name] = {
                    "alert_whitelist":    alert_whitelist,
                    "fade_threshold":     0.60,   # P(failure) >= threshold -> FADE
                    "follow_threshold":   0.40,   # P(failure) <= threshold -> FOLLOW
                    "oos_sharpe":         round(float(best_row["oos_sharpe"]), 3) if pd.notna(best_row["oos_sharpe"]) else None,
                    "oos_win_rate":       round(float(best_row["oos_win_rate"]), 3) if pd.notna(best_row["oos_win_rate"]) else None,
                    # Cap total_ret at ±10x to avoid compounding artifacts in display
                    "oos_mean_ret":       round(float(best_row["oos_mean_ret"]), 5) if pd.notna(best_row["oos_mean_ret"]) else None,
                    "benchmark_sharpe":   round(float(bm_total), 3) if not np.isnan(bm_total) else None,
                }

            return variant_recs

        long_recs  = _build_variant_recommendation(long_qual,  "longonly", "LONG-ONLY",     top_n)
        short_recs = _build_variant_recommendation(short_qual, "both",     "SHORT-ENABLED", top_n) if allow_short else {}

        # Merge all per-variant entries into rec_output (keyed by full variant name)
        rec_output.update(long_recs)
        rec_output.update(short_recs)

        # -- Save strategy_recommendation.yaml --------------------------------
        # Keyed by full variant name (e.g. "h10d_longonly") for direct lookup in run_agent.py
        rec_file = out_dir / "strategy_recommendation.yaml"
        rec_data = {
            "generated":        datetime.now().strftime("%Y-%m-%d"),
            "oos_period_start": oos_start_str,
            "oos_period_end":   oos_end_str,
        }
        rec_data.update(rec_output)
        with open(rec_file, "w") as f:
            f.write("# Auto-generated by optimize_strategy.py -- do not edit manually\n")
            f.write("# Loaded by run_agent.py at startup; keyed by full variant name\n")
            f.write("# e.g. h10d_longonly: {alert_whitelist: [...], fade_threshold: 0.60, follow_threshold: 0.40}\n")
            yaml.dump(rec_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        log.info("Saved strategy_recommendation.yaml to %s", rec_file)

    print("\n" + "=" * 80)
    print(f"Full results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest all (alert_type, signal_mode, horizon) combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--yfinance", action="store_true",
                     help="Use yfinance (no IBKR required) -- recommended")
    src.add_argument("--paper", action="store_true",
                     help="Use IBKR paper trading port (4002)")
    src.add_argument("--live", action="store_true",
                     help="Use IBKR live trading port (4001)")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Min OOS trades for a combo to be recommended (default: 10)")
    parser.add_argument("--commission", type=float, default=COMMISSION,
                        help="Round-trip commission per trade (default: 0.002 = 0.2%%)")
    parser.add_argument("--allow-short", action="store_true",
                        help="Also test SHORT strategies (follow_bearish, fade_bullish). "
                             "Use when IBKR short locates are available.")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top alert types to include in the alert_whitelist "
                             "recommendation per horizon (default: 5)")
    args = parser.parse_args()

    main(
        use_yfinance=args.yfinance or (not args.paper and not args.live),
        min_trades=args.min_trades,
        commission=args.commission,
        allow_short=args.allow_short,
        top_n=args.top_n,
    )
