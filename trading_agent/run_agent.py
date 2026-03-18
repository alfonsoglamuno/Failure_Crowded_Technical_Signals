"""
Trading agent main entry point.

Usage:
    python run_agent.py --paper                      # interactive variant selection + paper trading
    python run_agent.py --paper --variant h3d_longonly  # skip menu, use specific variant
    python run_agent.py --live                       # live trading (confirm required)
    python run_agent.py --status                     # print journal summary, no trading
    python run_agent.py --paper --once               # run one cycle now

Model variants (horizon x direction mode):
    h1d_longonly   1-day hold, BUY-only (fastest intraday fades)
    h3d_longonly   3-day hold, BUY-only  ← default
    h5d_longonly   5-day hold, BUY-only  — multi-day positions
    h1d_both       1-day hold, BUY+SELL  (allow_short=true required)
    h3d_both       3-day hold, BUY+SELL
    h5d_both       5-day hold, BUY+SELL

Continuous intraday loop:
    - Fast cycle (5 min):  trail stops, time exits, exit sync
    - Slow cycle (30 min): full signal scan, new trades
    - EOD close at 16:30 CET: cancel all brackets, flatten positions
    - Morning check: re-evaluate any overnight positions FIRST
    - Reconnect recovery: on IBKR reconnect, reconcile positions before resuming
"""

import argparse
import logging
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("agent")

CET = ZoneInfo("Europe/Berlin")

# Euro STOXX 50 exchanges open 09:00-17:35 CET; we trade 09:10-17:00 window.
_MARKET_OPEN  = dtime(9, 10)
_MARKET_CLOSE = dtime(17, 0)

# GBP-denominated tickers — skip for EUR sizing (exchange rate not applied)
_GBP_TICKERS = {"CRH.L", "FLTR.L", "NG.L"}

# Maximum fraction of universe tickers allowed to be stale/missing before
# the agent halts and demands manual review. The EURO STOXX 50 is reconstituted
# quarterly; a sudden spike in missing tickers is the earliest symptom.
_MAX_STALE_TICKER_FRACTION = 0.20   # >20% stale → halt

# Horizon-dependent correlation thresholds.
# Intraday (h1d): positions close EOD — overlap risk is low, gate is soft.
# Swing/position (h3d, h5d): positions can overlap for days — gate is tighter.
_PEER_CORR_THRESHOLD = {1: 0.85, 3: 0.70, 5: 0.65}
_PEER_CORR_WINDOW    = 20     # trading days for correlation lookback


def _check_universe_integrity(tickers: list[str], universe_data: dict) -> tuple[list[str], bool]:
    """
    Check whether the cached universe data is fresh and complete.

    The EURO STOXX 50 is reconstituted quarterly. When a constituent is replaced,
    its Yahoo Finance ticker either stops updating or returns no data. This function
    catches that early so the agent does not trade stale or delisted symbols.

    Returns:
        stale  : list of tickers with no data or no price update in the last 5 days
        halt   : True if stale fraction exceeds _MAX_STALE_TICKER_FRACTION
                 (suggests a major index reconstitution — manual review required)
    """
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=7)   # allow for weekends + 1 holiday buffer

    stale = []
    for ticker in tickers:
        df = universe_data.get(ticker)
        if df is None or df.empty:
            stale.append(ticker)
            continue
        try:
            last_date = df["date"].max() if "date" in df.columns else df.index.max()
            if hasattr(last_date, "date"):
                last_date = last_date.date()
            if last_date < cutoff:
                stale.append(ticker)
        except Exception:
            stale.append(ticker)

    halt = len(stale) / max(len(tickers), 1) > _MAX_STALE_TICKER_FRACTION

    if stale:
        log.warning(
            "[UNIVERSE] %d/%d tickers have stale or missing data: %s",
            len(stale), len(tickers), stale,
        )
        if halt:
            log.error(
                "[UNIVERSE] %.0f%% of tickers are stale (threshold %.0f%%). "
                "This may indicate a EURO STOXX 50 reconstitution. "
                "TRADING HALTED. Update configs/eurostoxx50_tickers.yaml and "
                "trading_agent/configs/ibkr_contracts.yaml, then retrain the models.",
                len(stale) / len(tickers) * 100,
                _MAX_STALE_TICKER_FRACTION * 100,
            )
        else:
            log.warning(
                "[UNIVERSE] Stale tickers will be skipped this session. "
                "Verify against https://www.stoxx.com/index-details?isin=EU0009658145",
            )
    else:
        log.info("[UNIVERSE] All %d tickers have recent data.", len(tickers))

    return stale, halt


def _corr_threshold(horizon_days: int) -> float:
    """Return correlation gate threshold for a given hold horizon."""
    return _PEER_CORR_THRESHOLD.get(horizon_days, 0.70)


def _parse_horizon(variant: str) -> int:
    """Extract hold horizon in days from variant name (e.g. 'h3d_longonly' -> 3)."""
    try:
        return int(variant.split("d_")[0].lstrip("h"))
    except Exception:
        return 3


def _max_corr_with_open(candidate: str,
                         candidate_direction: str,
                         open_positions: dict[str, str],
                         universe_data: dict,
                         allow_short: bool = False) -> float:
    """
    Return the maximum *effective* 20-day return correlation between `candidate`
    and any currently-open position.

    Long-only mode (allow_short=False):
      All positions are longs, candidate is long. Raw |correlation| is used.
      High raw corr = concentrated same-direction bet = block.

    Short-enabled mode (allow_short=True):
      Direction-adjusted effective correlation:
        effective_corr = sign(candidate) * sign(open) * raw_corr
      - Same direction + high raw corr  -> effective_corr > 0 (concentrated) -> block
      - Opposite directions + high corr -> effective_corr < 0 (hedged pair)  -> allow

      Rationale: LONG A and SHORT B on correlated stocks partially delta-hedge each
      other. Blocking this pair would prevent pairs-trade-style entries. The gate
      should only block same-direction concentration.

    Returns 0.0 if open_positions is empty or data is insufficient.
    """
    if not open_positions or candidate not in universe_data:
        return 0.0

    import pandas as pd
    import math
    cand_df = universe_data[candidate]
    if "close" not in cand_df.columns or len(cand_df) < _PEER_CORR_WINDOW + 2:
        return 0.0

    cand_ret  = cand_df["close"].pct_change().dropna().tail(_PEER_CORR_WINDOW * 2)
    cand_sign = 1 if candidate_direction == "BUY" else -1
    max_eff   = 0.0

    for open_t, open_dir in open_positions.items():
        if open_t not in universe_data:
            continue
        peer_df = universe_data[open_t]
        if "close" not in peer_df.columns or len(peer_df) < _PEER_CORR_WINDOW + 2:
            continue
        peer_ret = peer_df["close"].pct_change().dropna().tail(_PEER_CORR_WINDOW * 2)
        aligned  = pd.concat([cand_ret, peer_ret], axis=1, join="inner")
        if len(aligned) < _PEER_CORR_WINDOW:
            continue
        raw_corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if math.isnan(raw_corr):
            continue

        if allow_short:
            open_sign    = 1 if open_dir == "BUY" else -1
            effective    = cand_sign * open_sign * raw_corr
        else:
            effective    = abs(raw_corr)   # longonly: all same direction

        max_eff = max(max_eff, effective)

    return max_eff


# ── Model variant helpers ─────────────────────────────────────────────────────

_VARIANTS = [
    ("h1d_longonly", "1-day hold, BUY-only  — intraday fades, EOD close [default]"),
    ("h3d_longonly", "3-day hold, BUY-only  — swing fades, no shorts"),
    ("h5d_longonly", "5-day hold, BUY-only  — multi-day positions"),
    ("h1d_both",     "1-day hold, BUY+SELL  — requires allow_short=true"),
    ("h3d_both",     "3-day hold, BUY+SELL  — requires allow_short=true"),
    ("h5d_both",     "5-day hold, BUY+SELL  — requires allow_short=true"),
]


def _variant_paths(model_dir: Path, variant: str) -> tuple[Path, Path]:
    return (
        model_dir / f"xgboost_{variant}.joblib",
        model_dir / f"feature_cols_{variant}.json",
    )


def _apply_variant(cfg: dict, variant: str) -> None:
    """Rewrite model paths in cfg based on variant name. Exits if not found.

    Also auto-configures allow_short and follow_disabled from the variant suffix:
      _longonly  → allow_short=False, follow_disabled=True  (fade long-only)
      _both      → allow_short=True,  follow_disabled=False (fade+follow, both dirs)
    """
    model_dir = Path(cfg["model"]["path"]).parent
    model_path, cols_path = _variant_paths(model_dir, variant)
    if not model_path.exists():
        log.error(
            "Variant '%s' not found at %s\n"
            "  Run: python bootstrap_model.py --yfinance --variant %s",
            variant, model_path, variant,
        )
        sys.exit(1)
    cfg["model"]["variant"]           = variant
    cfg["model"]["path"]              = str(model_path)
    cfg["model"]["feature_cols_path"] = str(cols_path)

    # Auto-configure direction mode from variant suffix
    if variant.endswith("_both"):
        cfg["strategy"]["allow_short"]     = True
        cfg["strategy"]["follow_disabled"] = False
        log.info("Variant %s → allow_short=True  follow_disabled=False", variant)
    elif variant.endswith("_longonly"):
        cfg["strategy"]["allow_short"]     = False
        cfg["strategy"]["follow_disabled"] = True
        log.info("Variant %s → allow_short=False  follow_disabled=True", variant)
    else:
        log.warning("Unrecognised variant suffix in '%s' — "
                    "allow_short/follow_disabled not changed", variant)

    log.info("Active model variant: %s", variant)


def _choose_variant_interactive(cfg: dict) -> str:
    """
    Interactive startup menu — shown when --variant is not passed and stdin
    is a terminal.  Returns the chosen variant name.
    """
    model_dir = Path(cfg["model"]["path"]).parent
    current   = cfg["model"].get("variant", "h3d_longonly")

    print("\n" + "=" * 64)
    print("  TRADING AGENT — SELECT MODEL VARIANT")
    print("=" * 64)
    for i, (name, desc) in enumerate(_VARIANTS, 1):
        mp, _ = _variant_paths(model_dir, name)
        avail  = "ready " if mp.exists() else "MISSING"
        marker = " ← current" if name == current else ""
        print(f"  {i}. [{avail}]  {name:<20s}  {desc}{marker}")
    print(f"  0. Keep current config ({current})")
    print()

    while True:
        try:
            raw = input(f"Choose variant [0-{len(_VARIANTS)}] (Enter = keep current): ").strip()
        except (EOFError, KeyboardInterrupt):
            return current
        if raw == "" or raw == "0":
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(_VARIANTS):
            choice = _VARIANTS[int(raw) - 1][0]
            mp, _ = _variant_paths(model_dir, choice)
            if not mp.exists():
                print(f"  Not trained yet. Run: python bootstrap_model.py --yfinance --variant {choice}")
                continue
            return choice
        # Accept variant name typed directly
        if any(raw == v for v, _ in _VARIANTS):
            mp, _ = _variant_paths(model_dir, raw)
            if mp.exists():
                return raw
            print(f"  Not trained yet.")
        else:
            print("  Invalid choice.")


# ── Reconnection recovery ─────────────────────────────────────────────────────

def _reconnect_evaluate(paper: bool, cfg: dict) -> None:
    """
    Called once after the agent reconnects to IBKR following a connection loss.

    Steps:
      1. Sync exits — reconcile any TP/SL fills that happened during the outage.
      2. Position health check — ensure every open position has a valid SL order.
      3. Log the outcome for each position so the next scan has accurate context.

    This does NOT place new trades.  It only brings the agent's state back into
    sync with reality after an uncontrolled disconnect.
    """
    from agent.data_feed import IBKRFeed, load_contracts
    from agent.journal import Journal
    from agent.model import FailurePredictor
    from agent.learner import AdaptiveLearner
    from agent.monitor import PositionMonitor

    log.warning("=== RECONNECT: syncing positions after connection loss ===")

    journal       = Journal(cfg["journal"]["db_path"])
    contracts_cfg = load_contracts(cfg["universe"]["ibkr_contracts_file"])
    predictor     = FailurePredictor(
        model_path=cfg["model"]["path"],
        feature_cols_path=cfg["model"]["feature_cols_path"],
    )
    predictor.load()
    learner = AdaptiveLearner(cfg, journal, predictor)

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        monitor = PositionMonitor(
            feed.ib, journal, learner,
            commission=cfg["risk"]["commission_per_trade_eur"],
            commission_rate=cfg["risk"].get("commission_rate_pct", 0.05) / 100,
            commission_min=cfg["risk"].get("commission_min_eur", 2.0),
            paper=paper,
            account_id=feed.account_id,
        )
        # Step 1: catch any fills that arrived while we were offline
        monitor.check_exits()
        log.info("Reconnect: exit sync complete")

        # Step 2: SL health check + trail any eligible positions
        monitor.monitor_positions(
            contracts_cfg=contracts_cfg,
            feed=feed,
            cfg=cfg,
            account=feed.account_id,
        )
        log.info("Reconnect: position health check complete")

    except Exception as e:
        log.error("Reconnect evaluation failed: %s", e)
    finally:
        feed.disconnect()
    log.warning("=== RECONNECT: sync complete — resuming normal operation ===")


def is_market_open(now: datetime | None = None) -> bool:
    """Returns True if we are inside European market hours on a weekday."""
    t = (now or datetime.now(CET)).astimezone(CET)
    if t.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= t.time() <= _MARKET_CLOSE


def setup_file_logging(log_path: str):
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def print_status(journal):
    print("\n" + "="*50)
    print("TRADING AGENT STATUS")
    print("="*50)
    perf = journal.get_performance_summary()
    if perf["n_trades"] == 0:
        print("No completed trades yet.")
    else:
        print(f"Completed trades : {perf['n_trades']}")
        print(f"Total P&L (net)  : {perf['total_pnl']:.2f} EUR")
        print(f"Avg P&L per trade: {perf['avg_pnl']:.4f} EUR")
        print(f"Hit rate         : {perf['hit_rate']:.1%}")
        print(f"Best trade       : {perf['best_trade']:.2f} EUR")
        print(f"Worst trade      : {perf['worst_trade']:.2f} EUR")
    print()
    print("Recent signals:")
    for s in journal.get_recent_signals(5):
        print(f"  {s['date']} {s['ticker']:10s} {s['alert_name']:25s} "
              f"P={s['failure_proba']:.3f}  {s['action']}")
    print("="*50 + "\n")


def run_once(paper: bool, cfg: dict):
    """Execute one full trading cycle."""
    from agent.data_feed import IBKRFeed, load_contracts
    from agent.alerts import detect_universe_alerts
    from agent.features import build_feature_row
    from agent.model import FailurePredictor
    from agent.strategy import make_signals, filter_signals
    from agent.risk import RiskManager
    from agent.executor import IBKRExecutor
    from agent.journal import Journal
    from agent.learner import AdaptiveLearner
    from agent.monitor import PositionMonitor
    from agent.explain import build_explanation

    now = datetime.now(CET)
    log.info("=== Trading cycle start [%s] === %s", "PAPER" if paper else "LIVE", now.strftime("%Y-%m-%d %H:%M CET"))
    today = now.date()

    if not is_market_open(now):
        log.warning("Market is closed right now (%s). Signals will be based on yesterday's close "
                    "and orders queued until next open. Continuing anyway.", now.strftime("%H:%M CET"))

    # ── Initialise ───────────────────────────────────────────────────────────
    journal = Journal(cfg["journal"]["db_path"])
    setup_file_logging(cfg["journal"]["log_path"])

    predictor = FailurePredictor(
        model_path=cfg["model"]["path"],
        feature_cols_path=cfg["model"]["feature_cols_path"],
    )
    if not predictor.load():
        log.error("Model not found. Run bootstrap_model.py first.")
        return

    contracts_cfg = load_contracts(cfg["universe"]["ibkr_contracts_file"])
    with open(cfg["universe"]["parent_tickers_file"]) as f:
        tickers = yaml.safe_load(f)["tickers"]

    risk = RiskManager(cfg, horizon_days=_parse_horizon(cfg["model"].get("variant", "h1d_both")))
    learner = AdaptiveLearner(cfg, journal, predictor)

    # ── Connect to IBKR ──────────────────────────────────────────────────────
    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
    except ConnectionError as e:
        log.error("IBKR connection failed: %s", e)
        return

    # ── Account / mode banner ─────────────────────────────────────────────────
    mode_str = "PAPER" if paper else "*** LIVE — REAL MONEY ***"
    log.info("=" * 60)
    log.info("  MODE    : %s", mode_str)
    log.info("  ACCOUNT : %s", feed.account_id)
    log.info("  VARIANT : %s  (allow_short=%s)", cfg["model"].get("variant", "h3d_longonly"),
             cfg["strategy"].get("allow_short", False))
    log.info("=" * 60)

    try:
        # ── Cancel stale PreSubmitted DAY orders from previous sessions ──────
        # IBKR paper sometimes fails to auto-expire DAY market orders for exotic
        # exchanges (e.g. HEX/EUSTARS for NOKIA). Cancel them on reconnect so
        # they don't block position slots or generate journal noise.
        try:
            stale_cancelled = 0
            for trade in feed.ib.trades():
                o = trade.order
                if (getattr(o, "orderType", "") in ("MKT", "LMT")
                        and getattr(o, "tif", "") == "DAY"
                        and getattr(o, "parentId", 0) == 0   # parent only
                        and trade.orderStatus.status == "PreSubmitted"):
                    feed.ib.cancelOrder(o)
                    stale_cancelled += 1
            if stale_cancelled:
                feed.ib.sleep(1)
                log.info("Cancelled %d stale PreSubmitted DAY parent order(s)", stale_cancelled)
        except Exception as e:
            log.warning("Stale order cleanup failed: %s", e)

        # ── Check exits from previous bracket orders ──────────────────────────
        monitor = PositionMonitor(feed.ib, journal, learner,
                                  commission=cfg["risk"]["commission_per_trade_eur"],
                                  commission_rate=cfg["risk"].get("commission_rate_pct", 0.05) / 100,
                                  commission_min=cfg["risk"].get("commission_min_eur", 2.0),
                                  paper=paper, account_id=feed.account_id)
        monitor.check_exits()

        nav = feed.get_nav()
        daily_pnl = feed.get_daily_pnl()
        log.info("Account NAV: %.2f EUR  Daily P&L: %.2f EUR", nav, daily_pnl)

        # ── Daily loss check ─────────────────────────────────────────────────
        if not risk.check_daily_loss(daily_pnl):
            journal.log_daily_summary(nav, daily_pnl, 0, 0, paper, today)
            return

        # ── Fetch universe data ───────────────────────────────────────────────
        universe_data = feed.fetch_universe(
            tickers, contracts_cfg,
            days=cfg["model"]["min_history_days"] + 50,
            cache_dir=cfg["data"]["cache_dir"],
        )
        log.info("Fetched data for %d tickers", len(universe_data))

        # ── Universe integrity check — detect EURO STOXX 50 reconstitution ────
        stale_tickers, should_halt = _check_universe_integrity(tickers, universe_data)
        if should_halt:
            print()
            print("!" * 68)
            print("!!  UNIVERSE INTEGRITY CHECK FAILED — TRADING HALTED           !!")
            print("!!")
            print(f"!!  {len(stale_tickers)} tickers have stale or missing data (>{_MAX_STALE_TICKER_FRACTION:.0%} threshold).")
            print("!!  The EURO STOXX 50 may have been reconstituted.")
            print("!!")
            print("!!  Action required:")
            print("!!    1. Verify constituents at stoxx.com")
            print("!!    2. Update configs/eurostoxx50_tickers.yaml")
            print("!!    3. Update trading_agent/configs/ibkr_contracts.yaml")
            print("!!    4. Retrain: python bootstrap_model.py --yfinance")
            print(f"!!  Stale: {stale_tickers}")
            print("!" * 68)
            return
        # Remove stale tickers from active universe for this session
        for t in stale_tickers:
            universe_data.pop(t, None)

        # ── Index for regime features ─────────────────────────────────────────
        try:
            import yfinance as yf
            idx = yf.download("^STOXX50E", period="2y", auto_adjust=True, progress=False)
            idx.columns = [c[0] if isinstance(c, tuple) else c for c in idx.columns]
            index_close = idx["Close"]
            index_close.index = __import__("pandas").to_datetime(index_close.index)
        except Exception:
            index_close = None

        # ── Alert detection ───────────────────────────────────────────────────
        events = detect_universe_alerts(universe_data)
        log.info("Today's alerts: %d events across %d tickers",
                 len(events), events["ticker"].nunique() if not events.empty else 0)

        if events.empty:
            log.info("No alerts today — no trades")
            journal.log_daily_summary(nav, daily_pnl, 0, 0, paper, today)
            return

        # ── Feature engineering ───────────────────────────────────────────────
        feature_df = build_feature_row(
            events, universe_data,
            feature_cols=predictor.feature_cols,
            index_close=index_close,
        )
        if feature_df.empty:
            log.warning("No features built — skipping")
            return

        # ── Predict ───────────────────────────────────────────────────────────
        probas = predictor.predict(feature_df)
        log.info("Predictions: min=%.3f  max=%.3f  mean=%.3f",
                 probas.min(), probas.max(), probas.mean())

        # ── Strategy + risk filter ────────────────────────────────────────────
        # Resolve active variant and horizon here so make_signals and the
        # execution loop both use the same value.
        _active_variant = cfg["model"].get("variant", "h3d_longonly")
        _horizon_days   = _parse_horizon(_active_variant)

        # When follow_disabled=True, set follow_threshold to an impossible value
        # so make_signals never emits FOLLOW signals.  This is the canonical way
        # to honour the flag — P(failure) cannot be < 0, so no FOLLOW is emitted.
        _follow_thr = (
            0.0 if cfg["strategy"].get("follow_disabled", True)
            else learner.follow_threshold
        )
        signals = make_signals(
            feature_df, probas,
            fade_threshold=learner.fade_threshold,
            follow_threshold=_follow_thr,
            horizon_days=_horizon_days,
            crowding_min_score=cfg["strategy"].get("crowding_min_score", 0.30),
            regime_threshold_boost=cfg["strategy"].get("regime_threshold_boost", 0.05),
        )
        signals = filter_signals(
            signals,
            max_trades=cfg["risk"]["max_trades_per_day"],
            allow_short=cfg["strategy"].get("allow_short", False),
        )

        # ── Learner per-alert suppression ─────────────────────────────────────
        # Remove signals whose (action, alert_name) pair has a sustained poor
        # win rate (< suppress_win_rate_threshold after min_trades_for_recalibration
        # samples). This lets the learner silence alert types that have proven
        # unprofitable rather than waiting for a full threshold recalibration.
        suppressed_before = len(signals)
        signals = [
            s for s in signals
            if not learner.is_alert_suppressed(s.action, s.alert_name)
        ]
        if len(signals) < suppressed_before:
            log.info("Learner suppressed %d signal(s) due to poor alert-type performance",
                     suppressed_before - len(signals))

        # ── Log all signals (iterate feature_df rows — aligned with probas) ────
        # feature_df may have fewer rows than events (skipped tickers)
        signal_id_map: dict[str, int] = {}  # ticker+alert_name → signal_id
        for feat_idx in range(len(feature_df)):
            feat_row  = feature_df.iloc[feat_idx]
            proba_val = float(probas.iloc[feat_idx])
            ticker     = feat_row.get("ticker", "")
            alert_name = feat_row.get("_alert_name_raw", "unknown")
            direction  = feat_row.get("_dir_raw", "")
            found_sig  = next(
                (s for s in signals if s.ticker == ticker and s.alert_name == alert_name),
                None,
            )
            action     = found_sig.action           if found_sig else "SKIP"
            trade_dir  = found_sig.trade_direction  if found_sig else None
            conviction = found_sig.conviction       if found_sig else 0.0
            crowding   = found_sig.crowding_score   if found_sig else 0.0

            # Build explanation only for actionable (non-SKIP) signals
            explanation = ""
            if found_sig and action != "SKIP":
                try:
                    explanation = build_explanation(
                        model=predictor._model,
                        feature_row=feat_row,
                        feature_cols=predictor.feature_cols,
                        failure_proba=proba_val,
                        alert_name=alert_name,
                        alert_direction=direction,
                        action=action,
                        trade_direction=trade_dir or "",
                        crowding_score=crowding,
                    )
                    log.info("[WHY] %s", explanation)
                except Exception as exc:
                    log.debug("Explanation build failed for %s: %s", ticker, exc)

            sid = journal.log_signal(
                ticker=ticker,
                alert_name=alert_name,
                alert_direction=direction,
                failure_proba=proba_val,
                action=action,
                trade_direction=trade_dir,
                conviction=conviction,
                trade_date=today,
                crowding_score=crowding,
                explanation=explanation,
            )
            signal_id_map[f"{ticker}|{alert_name}"] = sid

        # Log events that failed feature building as SKIP
        feature_keys = set(
            f"{r.get('ticker','')}|{r.get('_alert_name_raw','')}"
            for _, r in feature_df.iterrows()
        )
        for _, ev in events.iterrows():
            key = f"{ev.get('ticker','')}|{ev.get('alert_name','')}"
            if key not in feature_keys:
                journal.log_signal(
                    ticker=ev.get("ticker", ""),
                    alert_name=ev.get("alert_name", ""),
                    alert_direction=ev.get("direction", ""),
                    failure_proba=0.5,
                    action="SKIP",
                    trade_direction=None,
                    conviction=0.0,
                    trade_date=today,
                )

        log.info("Actionable signals: %d", len(signals))

        # ── Execute ───────────────────────────────────────────────────────────
        executor = IBKRExecutor(
            feed.ib, contracts_cfg, paper=paper, account=feed.account_id,
            allow_short=cfg["strategy"].get("allow_short", False),
        )
        n_trades = 0

        # Dedup: don't re-trade a ticker that was already traded today, regardless of
        # whether the position is still open or already closed.  A second entry on the
        # same ticker on the same day doubles concentration risk and inflates commissions.
        # Exclude pure errors (order never submitted to IBKR) — those may legitimately retry.
        already_traded_today = {
            t["ticker"] for t in journal.get_recent_trades(100)
            if str(t.get("date", "")) == str(today)
               and t.get("status") not in ("error",)
        }
        if already_traded_today:
            log.info("Already traded today: %s — will not re-enter", sorted(already_traded_today))

        # Build direction map for open positions — used by the correlation gate.
        # Knowing each position's direction (BUY/SELL) lets us distinguish hedged
        # pairs (long+short on correlated names) from concentrated same-direction bets.
        open_trades_all = journal.get_recent_trades(100)
        open_positions_dir: dict[str, str] = {
            t["ticker"]: t.get("trade_direction", "BUY")
            for t in open_trades_all
            if t.get("status") in ("submitted", "filled")
            and t.get("exit_price") is None
        }

        # Correlation gate threshold — horizon-scaled (set above near make_signals).
        _corr_thr    = _corr_threshold(_horizon_days)
        _allow_short = cfg["strategy"].get("allow_short", False)

        # Build set of tickers already active in IBKR:
        #   - any non-zero positions (long OR short — shorts block new long entries)
        #   - pending/submitted orders (PendingSubmit, Submitted, PreSubmitted)
        # Both must be excluded to prevent double-entry across scans.
        open_ibkr_positions: set[str] = set()
        try:
            symbol_to_yahoo = {v["symbol"]: k for k, v in contracts_cfg.items()}

            # Filled positions (long or short — both block new entries)
            for pos in feed.ib.positions():
                if abs(pos.position) > 0:
                    yahoo = symbol_to_yahoo.get(pos.contract.symbol, "")
                    if yahoo:
                        open_ibkr_positions.add(yahoo)

            # Pending / live parent orders (not yet filled)
            for trade in feed.ib.trades():
                status = trade.orderStatus.status
                if status in ("PendingSubmit", "Submitted", "PreSubmitted", "Filled"):
                    if not getattr(trade.order, "parentId", 0):  # parent orders only
                        yahoo = symbol_to_yahoo.get(trade.contract.symbol, "")
                        if yahoo:
                            open_ibkr_positions.add(yahoo)

            if open_ibkr_positions:
                log.info("Active in IBKR (positions + pending orders): %s",
                         sorted(open_ibkr_positions))
        except Exception:
            pass

        max_open = cfg["strategy"].get("max_open_positions", 10)
        slots_available = max_open - len(open_ibkr_positions)
        if slots_available <= 0:
            log.info("Position cap reached (%d/%d) — no new trades this scan",
                     len(open_ibkr_positions), max_open)
            journal.log_daily_summary(nav, daily_pnl, len(events), 0, paper, today)
            return

        log.info("Open positions: %d/%d — %d slot(s) available",
                 len(open_ibkr_positions), max_open, slots_available)

        for sig in signals:
            # Skip GBP-quoted stocks — position sizing assumes EUR
            if sig.ticker in _GBP_TICKERS:
                log.info("Skipping %s (GBP-denominated, EUR sizing not supported)", sig.ticker)
                continue

            # Hard guard: SELL entries are forbidden in long-only mode.
            # strategy.py should filter these before they reach here; this is a
            # belt-and-suspenders check so a config error or strategy bug cannot
            # accidentally create a short position.
            if sig.trade_direction == "SELL" and not _allow_short:
                log.error(
                    "BLOCKED SELL entry for %s at run_agent level — "
                    "allow_short=False in config. Strategy should not emit SELL entries. "
                    "Check strategy.py or config allow_short setting.",
                    sig.ticker,
                )
                continue

            # Dedup — skip tickers already traded in an earlier scan today
            if sig.ticker in already_traded_today:
                log.info("Skipping %s — already traded today (one trade per ticker per day)", sig.ticker)
                continue

            # Don't add to an existing long position
            if sig.ticker in open_ibkr_positions:
                log.info("Skipping %s — already holding long position", sig.ticker)
                continue

            # Correlation gate — direction-aware, horizon-scaled.
            # Longonly: block if raw |corr| > threshold (all positions same direction).
            # Short-enabled: block only if effective_corr = sign(cand)*sign(open)*corr
            # exceeds threshold — allows hedged long+short pairs on correlated stocks.
            # Threshold is softer for h1d (0.85, EOD close) vs h3d (0.70) vs h5d (0.65).
            max_corr = _max_corr_with_open(
                sig.ticker, sig.trade_direction,
                open_positions_dir, universe_data,
                allow_short=_allow_short,
            )
            if max_corr >= _corr_thr:
                log.info(
                    "Skipping %s — effective corr %.2f >= %.2f threshold (h%dd, short=%s)",
                    sig.ticker, max_corr, _corr_thr, _horizon_days, _allow_short,
                )
                continue

            # Stop adding once we've filled all available slots this scan
            if n_trades >= slots_available:
                log.info("Slot cap reached for this scan (%d new trades) — stopping", n_trades)
                break

            # Qualify the IBKR contract (validates symbol/exchange, gets conId)
            contract = executor._make_contract(sig.ticker)
            if contract is None:
                log.warning("No IBKR contract mapping for %s", sig.ticker)
                continue
            if not feed.qualify_contract(contract):
                log.warning("Contract qualification failed for %s — skipping", sig.ticker)
                continue

            # EUR-only guard: reject any contract that is not denominated in EUR.
            # This prevents FX commission leakage and unintended cross-currency exposure.
            if getattr(contract, "currency", "EUR") != "EUR":
                log.error(
                    "BLOCKED %s — contract currency is %s, not EUR. "
                    "Remove this ticker from ibkr_contracts.yaml or fix its currency field.",
                    sig.ticker, contract.currency,
                )
                continue

            # ── L1 bid/ask depth: price + available volume at quoted price ────────
            # For EURO STOXX 50 stocks with ~2,000 EUR positions (typically 20-50
            # shares) L1 liquidity is almost always sufficient, but we log it for
            # every live trade so slippage is visible in the trading log.
            ohlcv = universe_data.get(sig.ticker, __import__("pandas").DataFrame())
            fallback = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 0.0
            depth = feed.get_bid_ask_depth(contract, fallback_price=fallback)

            current_price = depth["mid"] if depth["mid"] > 0 else fallback
            if current_price <= 0:
                log.warning("No price available for %s — skipping", sig.ticker)
                continue

            sizing = risk.size_position(sig, nav, current_price)
            if sizing is None:
                continue

            # Liquidity check: warn if order size may exceed available L1 volume
            liq = risk.check_liquidity(
                direction=sig.trade_direction,
                quantity=sizing["quantity"],
                bid=depth["bid"],
                ask=depth["ask"],
                bid_size=depth["bid_size"],
                ask_size=depth["ask_size"],
                ticker=sig.ticker,
            )

            log.info(
                "[SIGNAL] %s %s P(failure)=%.3f crowd=%.2f -> %s "
                "qty=%d  bid=%.4f(%.0fsh) ask=%.4f(%.0fsh)  "
                "entry=%.4f SL=%.4f TP=%.4f  spread=%.3f%%%s",
                sig.action, sig.ticker, sig.failure_proba, sig.crowding_score,
                sig.trade_direction,
                sizing["quantity"],
                depth["bid"], depth["bid_size"],
                depth["ask"], depth["ask_size"],
                sizing["entry_price"],
                sizing["stop_loss"], sizing["take_profit"],
                liq["spread_pct"] * 100,
                " [SWEEP RISK]" if liq["sweep_risk"] else "",
            )

            result = executor.place_bracket(
                yahoo_ticker=sig.ticker,
                trade_direction=sig.trade_direction,
                quantity=sizing["quantity"],
                entry_price=sizing["entry_price"],
                stop_loss=sizing["stop_loss"],
                take_profit=sizing["take_profit"],
                action=sig.action,
                contract=contract,
                horizon_days=_horizon_days,   # drives DAY vs GTC TIF for TP/SL
            )

            signal_id = signal_id_map.get(f"{sig.ticker}|{sig.alert_name}", -1)
            trade_id = journal.log_trade(
                signal_id=signal_id,
                ticker=sig.ticker,
                ibkr_symbol=result.ibkr_symbol,
                trade_direction=sig.trade_direction,
                quantity=sizing["quantity"],
                entry_price=sizing["entry_price"],
                stop_loss=sizing["stop_loss"],
                take_profit=sizing["take_profit"],
                ibkr_order_id=result.parent_order_id,
                status=result.status,
                paper=paper,
                trade_date=today,
                hold_horizon_days=_horizon_days,
            )

            # Record actual fill price (captured in executor after market order fills)
            if result.actual_fill_price > 0 and trade_id:
                journal.update_entry_fill(
                    trade_id=trade_id,
                    fill_price=result.actual_fill_price,
                    slippage_pct=result.slippage_pct,
                )

            if result.status in ("submitted", "filled"):
                n_trades += 1
                if result.actual_fill_price > 0:
                    fill_info = (f"fill={result.actual_fill_price:.4f} "
                                 f"slippage={result.slippage_pct * 100:+.3f}%")
                else:
                    fill_info = "fill pending"
                log.info("Order placed for %s (orderId=%d  %s)",
                         sig.ticker, result.parent_order_id, fill_info)
            else:
                log.error("Order FAILED for %s: %s", sig.ticker, result.error_msg)

        # ── Daily summary ─────────────────────────────────────────────────────
        journal.log_daily_summary(nav, daily_pnl, len(events), n_trades, paper, today)
        log.info("=== Cycle complete: %d signals, %d trades placed ===",
                 len(events), n_trades)

        # ── Learning loops ────────────────────────────────────────────────────
        learner.maybe_recalibrate()
        learner.maybe_retrain(universe_data, index_close)

    finally:
        feed.disconnect()


def _monitor_open_positions(paper: bool, cfg: dict):
    """
    Fast cycle (every 5 min): trail stop-losses and enforce time-based exits.
    Does NOT scan for new signals — that stays in run_once (every 30 min).
    Skips IBKR connection entirely if journal has no open trades.
    """
    from agent.data_feed import IBKRFeed, load_contracts
    from agent.journal import Journal
    from agent.model import FailurePredictor
    from agent.learner import AdaptiveLearner
    from agent.monitor import PositionMonitor

    journal = Journal(cfg["journal"]["db_path"])

    # Fast check — avoid connecting to IBKR if nothing is open
    open_trades = [
        t for t in journal.get_recent_trades(50)
        if t.get("status") in ("submitted", "filled", "pending_close")
        and t.get("exit_price") is None or t.get("status") == "pending_close"
    ]
    if not open_trades:
        log.debug("No open trades — skipping position monitor")
        return

    contracts_cfg = load_contracts(cfg["universe"]["ibkr_contracts_file"])
    predictor = FailurePredictor(
        model_path=cfg["model"]["path"],
        feature_cols_path=cfg["model"]["feature_cols_path"],
    )
    predictor.load()
    learner = AdaptiveLearner(cfg, journal, predictor)

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        monitor = PositionMonitor(
            feed.ib, journal, learner,
            commission=cfg["risk"]["commission_per_trade_eur"],
            commission_rate=cfg["risk"].get("commission_rate_pct", 0.05) / 100,
            commission_min=cfg["risk"].get("commission_min_eur", 2.0),
            paper=paper,
            account_id=feed.account_id,
        )
        monitor.check_exits()
        monitor.monitor_positions(
            contracts_cfg=contracts_cfg,
            feed=feed,
            cfg=cfg,
            account=feed.account_id,
        )
    except Exception as e:
        log.warning("Position monitor cycle error: %s", e)
    finally:
        feed.disconnect()


def _evaluate_overnight_positions(paper: bool, cfg: dict):
    """
    Run once at market open for any positions carried over from a previous session.

    Decision logic for each position:
      1. IBKR shows 0 qty  → already closed; check_exits will reconcile journal.
      2. Accidental SHORT   → cover immediately (MKT BUY).
      3. Price < original SL (gap-through-SL)
                            → close immediately; SL was bypassed overnight.
      4. Held > max_hold_days and still at a loss
                            → close; thesis is stale.
      5. SL missing + position is in profit (≥ 50% of TP distance)
                            → re-place SL at break-even (locks in gain).
      6. SL missing + small profit or loss (within original SL)
                            → re-place SL at original SL level (respect initial risk).
      7. SL active          → log current state; no action needed.

    This runs BEFORE any new signal scan so position inventory is accurate.
    """
    from agent.data_feed import IBKRFeed, load_contracts
    from agent.journal import Journal
    from ib_insync import Stock, StopOrder, MarketOrder
    from datetime import date
    import pandas as pd

    journal = Journal(cfg["journal"]["db_path"])
    today = date.today()

    open_trades = [
        t for t in journal.get_recent_trades(50)
        if t.get("status") in ("submitted", "filled")
        and t.get("exit_price") is None
        and str(t.get("date", "")) != str(today)   # from a previous day
    ]
    if not open_trades:
        return

    log.warning("Found %d overnight position(s) from previous session — evaluating", len(open_trades))
    contracts_cfg = load_contracts(cfg["universe"]["ibkr_contracts_file"])

    sl_pct      = cfg["risk"].get("stop_loss_pct", 0.015)
    tp_pct      = cfg["risk"].get("take_profit_pct", 0.025)
    max_hold_d  = cfg["model"].get("max_hold_days", 3)

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        ib = feed.ib
        ib.sleep(2)
        account = feed.account_id

        ibkr_positions = {pos.contract.symbol: pos.position for pos in ib.positions()}

        ib.reqAllOpenOrders()
        ib.sleep(1)
        # Bracket child stops keyed by parentId (same-session bracket orders)
        existing_stops = {
            t.order.parentId: t.order.auxPrice
            for t in ib.trades()
            if getattr(t.order, "orderType", "") == "STP"
            and getattr(t.order, "parentId", 0)
            and t.orderStatus.status not in ("Filled", "Cancelled", "Inactive")
        }
        # Standalone stops (restored after bracket expiry) keyed by symbol
        standalone_stop_symbols = {
            t.contract.symbol
            for t in ib.trades()
            if getattr(t.order, "orderType", "") == "STP"
            and not getattr(t.order, "parentId", 0)
            and t.orderStatus.status not in ("Filled", "Cancelled", "Inactive")
        }

        for rec in open_trades:
            ticker    = rec.get("ticker", "")
            symbol    = rec.get("ibkr_symbol", "")
            parent_id = rec.get("ibkr_order_id")
            direction = rec.get("trade_direction", "BUY")
            qty       = int(rec.get("quantity", 0))
            entry_px  = float(rec.get("entry_price") or 0)
            orig_sl   = float(rec.get("stop_loss") or 0)
            orig_tp   = float(rec.get("take_profit") or 0)

            # How many days has this been held?
            try:
                entry_date = pd.to_datetime(rec.get("date", today)).date()
                days_held  = (today - entry_date).days
            except Exception:
                days_held = 1

            ibkr_qty = ibkr_positions.get(symbol, 0)

            # ── Case 1: already closed in IBKR ────────────────────────────
            if ibkr_qty == 0:
                log.info("Overnight %s: IBKR position = 0 — check_exits will reconcile", ticker)
                continue

            # ── Case 2: accidental short ───────────────────────────────────
            if ibkr_qty < 0:
                log.warning("Overnight %s: accidental SHORT %d shares — covering at open", ticker, ibkr_qty)
                spec = contracts_cfg.get(ticker)
                if spec:
                    contract = Stock(spec["symbol"], "SMART", spec["currency"])
                    order = MarketOrder("BUY", abs(ibkr_qty))
                    order.account = account
                    order.tif = "DAY"
                    ib.placeOrder(contract, order)
                continue

            spec = contracts_cfg.get(ticker)
            if not spec:
                log.warning("Overnight %s: no contract spec — skipping", ticker)
                continue
            contract = Stock(spec["symbol"], "SMART", spec["currency"])
            current_px = feed.get_latest_price(contract, fallback_price=entry_px)

            # P&L assessment
            if direction == "BUY":
                profit_pct = (current_px - entry_px) / entry_px if entry_px else 0
                tp_dist    = (orig_tp - entry_px) / entry_px if orig_tp and entry_px else tp_pct
            else:
                profit_pct = (entry_px - current_px) / entry_px if entry_px else 0
                tp_dist    = (entry_px - orig_tp) / entry_px if orig_tp and entry_px else tp_pct

            # ── Case 3: gap-through-SL ─────────────────────────────────────
            # SL was bypassed by the overnight gap — original risk exceeded.
            gap_through_sl = (
                direction == "BUY"  and orig_sl > 0 and current_px < orig_sl
            ) or (
                direction == "SELL" and orig_sl > 0 and current_px > orig_sl
            )
            if gap_through_sl:
                log.warning(
                    "Overnight %s: price %.4f bypassed original SL %.4f (P&L=%.1f%%) "
                    "— closing at open [gap-through-SL]",
                    ticker, current_px, orig_sl, profit_pct * 100,
                )
                exit_action = "SELL" if direction == "BUY" else "BUY"
                order = MarketOrder(exit_action, qty)
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(contract, order)
                continue

            # ── Case 4: stale thesis — held too long and still losing ──────
            if days_held > max_hold_d and profit_pct <= 0:
                log.warning(
                    "Overnight %s: held %d days, P&L=%.1f%% — "
                    "closing (thesis stale, max_hold_days=%d)",
                    ticker, days_held, profit_pct * 100, max_hold_d,
                )
                exit_action = "SELL" if direction == "BUY" else "BUY"
                order = MarketOrder(exit_action, qty)
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(contract, order)
                continue

            # ── Case 5 & 6: SL active — just log ──────────────────────────
            # Check both bracket-child stops (keyed by parentId) and standalone
            # stops (restored after bracket expiry, keyed by symbol).
            if parent_id in existing_stops:
                log.info(
                    "Overnight %s: SL active @ %.4f  P&L=%.1f%%  held=%dd",
                    ticker, existing_stops[parent_id], profit_pct * 100, days_held,
                )
                continue
            if symbol in standalone_stop_symbols:
                log.info(
                    "Overnight %s: standalone SL active  P&L=%.1f%%  held=%dd",
                    ticker, profit_pct * 100, days_held,
                )
                continue

            # ── No active SL — re-place with smart level ───────────────────
            if direction == "BUY":
                if profit_pct >= tp_dist * 0.5:
                    # At least halfway to TP: lock in profit at break-even or better
                    sl_price = round(max(entry_px, current_px * (1 - sl_pct)), 4)
                    sl_reason = f"break-even lock (P&L=+{profit_pct:.1%})"
                elif profit_pct > 0:
                    # Small profit: normal SL from current price
                    sl_price  = round(current_px * (1 - sl_pct), 4)
                    sl_reason = f"normal from current (P&L=+{profit_pct:.1%})"
                else:
                    # At a loss: restore to original SL level (respect initial risk)
                    sl_price  = round(max(orig_sl, current_px * (1 - sl_pct)), 4) if orig_sl else round(current_px * (1 - sl_pct), 4)
                    sl_reason = f"original level (P&L={profit_pct:.1%})"
                sl_order = StopOrder("SELL", qty, sl_price)
            else:
                if profit_pct >= tp_dist * 0.5:
                    sl_price  = round(min(entry_px, current_px * (1 + sl_pct)), 4)
                    sl_reason = f"break-even lock (P&L=+{profit_pct:.1%})"
                else:
                    sl_price  = round(current_px * (1 + sl_pct), 4)
                    sl_reason = f"normal from current (P&L={profit_pct:.1%})"
                sl_order = StopOrder("BUY", qty, sl_price)

            sl_order.account  = account
            sl_order.tif      = "DAY"
            # No parentId — the original bracket parent (DAY order) has expired.
            # This is a standalone stop, not a bracket child.
            ib.placeOrder(contract, sl_order)
            log.warning(
                "Overnight %s: SL re-placed @ %.4f  [%s]  held=%dd",
                ticker, sl_price, sl_reason, days_held,
            )

        ib.sleep(2)
    except Exception as e:
        log.error("Overnight position evaluation failed: %s", e)
    finally:
        feed.disconnect()


def _eod_close(paper: bool, cfg: dict):
    """Cancel intraday brackets and close intraday positions at EOD.

    Multi-day positions (hold_horizon_days > 1) are intentionally left open
    overnight — their GTC SL/TP orders remain active and the morning check
    will re-evaluate them the next session.

    Accidental shorts are always covered regardless of horizon.
    """
    from agent.data_feed import IBKRFeed
    from agent.journal import Journal
    from ib_insync import MarketOrder, Stock
    from dotenv import load_dotenv
    load_dotenv()

    journal = Journal(cfg["journal"]["db_path"])

    # Identify which IBKR symbols belong to intended multi-day positions
    # (hold_horizon_days > 1) — we must NOT close or cancel orders for these.
    open_trades = journal.get_recent_trades(n=100)
    multiday_symbols: set[str] = set()
    multiday_order_ids: set[int] = set()
    for t in open_trades:
        is_open = (
            t.get("status") in ("submitted", "filled", "pending_close")
            and t.get("exit_price") is None
        )
        if is_open and (t.get("hold_horizon_days") or 1) > 1:
            sym = t.get("ibkr_symbol", "")
            oid = t.get("ibkr_order_id")
            if sym:
                multiday_symbols.add(sym.upper())
            if oid:
                multiday_order_ids.add(oid)

    if multiday_symbols:
        log.info("EOD: keeping multi-day positions open: %s", sorted(multiday_symbols))

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        ib = feed.ib
        account = feed.account_id
        ib.sleep(2)

        # Cancel open orders that belong to INTRADAY positions only.
        # Multi-day positions need their GTC SL/TP orders to survive overnight.
        open_orders = ib.reqAllOpenOrders()
        ib.sleep(1)
        cancelled = 0
        kept = 0
        for t in open_orders:
            order = t.order
            parent_id = getattr(order, "parentId", 0) or order.orderId
            # Skip cancellation if this order belongs to a multi-day position
            if parent_id in multiday_order_ids:
                kept += 1
                continue
            # Also skip if the contract symbol is a multi-day position
            sym = getattr(t.contract, "symbol", "").upper()
            if sym in multiday_symbols:
                kept += 1
                continue
            try:
                ib.cancelOrder(order)
                cancelled += 1
            except Exception:
                pass
        ib.sleep(2)
        log.info("EOD: cancelled %d intraday orders, kept %d multi-day orders",
                 cancelled, kept)

        # Close all intraday positions (horizon=1) and any accidental shorts.
        # Leave multi-day longs intact.
        positions = ib.positions()
        closed_long = 0
        closed_short = 0
        skipped_multiday = 0
        for pos in positions:
            qty = int(pos.position)
            if qty == 0:
                continue
            sym = pos.contract.symbol.upper()
            if qty > 0 and sym in multiday_symbols:
                skipped_multiday += 1
                log.info("EOD: leaving multi-day long open: %s qty=%d", sym, qty)
                continue
            # Route via SMART — pos.contract.exchange is exchange-specific (IBIS,
            # ENEXT.BE, etc.) which causes Error 321 "Missing order exchange".
            ccy      = getattr(pos.contract, "currency", "EUR")
            smart_ct = Stock(pos.contract.symbol, "SMART", ccy)
            if qty > 0:
                order = MarketOrder("SELL", qty)
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(smart_ct, order)
                log.info("EOD close LONG: SELL %d %s", qty, sym)
                closed_long += 1
            else:
                # Accidental short — always cover
                order = MarketOrder("BUY", abs(qty))
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(smart_ct, order)
                log.warning("EOD close SHORT: BUY %d %s (accidental short)", abs(qty), sym)
                closed_short += 1

        ib.sleep(3)
        log.info(
            "EOD complete — longs closed=%d  shorts covered=%d  multi-day kept=%d",
            closed_long, closed_short, skipped_multiday,
        )

    except Exception as e:
        log.error("EOD close failed: %s", e)
    finally:
        feed.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Crowded Signal Failure Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model variants:
  h1d_longonly  1-day hold, BUY-only  — fastest intraday fades
  h3d_longonly  3-day hold, BUY-only  — default
  h5d_longonly  5-day hold, BUY-only  — multi-day positions
  h1d_both      1-day hold, BUY+SELL  — allow_short=true required
  h3d_both      3-day hold, BUY+SELL
  h5d_both      5-day hold, BUY+SELL
""",
    )
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading mode (real money — confirm required)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle immediately and exit")
    parser.add_argument("--status", action="store_true",
                        help="Print journal summary and exit")
    parser.add_argument("--eod-close", action="store_true",
                        help="Cancel all brackets and close all positions now")
    parser.add_argument("--variant", metavar="VARIANT",
                        help="Model variant to use (e.g. h3d_longonly, h1d_both). "
                             "If omitted an interactive menu is shown on startup.")
    parser.add_argument("--no-menu", action="store_true",
                        help="Skip interactive variant menu, use config default")
    args = parser.parse_args()

    if args.live and args.paper:
        args.paper = False   # --live takes precedence

    cfg = load_config()

    # ── Status / utility commands (no variant needed) ─────────────────────────
    if args.status:
        from agent.journal import Journal
        journal = Journal(cfg["journal"]["db_path"])
        print_status(journal)
        return

    if args.eod_close:
        _eod_close(paper=not args.live, cfg=cfg)
        return

    # ── Variant selection ─────────────────────────────────────────────────────
    if args.variant:
        _apply_variant(cfg, args.variant)
    elif not args.no_menu and sys.stdin.isatty():
        chosen = _choose_variant_interactive(cfg)
        if chosen != cfg["model"].get("variant"):
            _apply_variant(cfg, chosen)
        else:
            log.info("Active model variant: %s (from config)", cfg["model"].get("variant"))
    else:
        log.info("Active model variant: %s (from config)", cfg["model"].get("variant"))

    # ── Live trading confirmation ─────────────────────────────────────────────
    if args.live:
        variant      = cfg["model"].get("variant", "unknown")
        allow_short  = cfg["strategy"].get("allow_short", False)
        pos_pct      = cfg["risk"].get("max_position_pct", 0.02) * 100
        capital      = cfg["capital"].get("initial", 0)
        max_loss     = cfg["risk"].get("max_daily_loss_eur", 0)
        sl_pct       = cfg["risk"].get("stop_loss_pct", 0.015) * 100
        tp_pct       = cfg["risk"].get("take_profit_pct", 0.025) * 100
        print()
        print("!" * 68)
        print("!!                                                                  !!")
        print("!!              *** LIVE TRADING  --  REAL MONEY ***               !!")
        print("!!                                                                  !!")
        print("!" * 68)
        print()
        print("  PERSONAL PROJECT DISCLAIMER")
        print("  ------------------------------------------------------------------")
        print("  This software is a personal research project. It is NOT a")
        print("  registered investment product and is NOT intended for real")
        print("  trading. Algorithmic trading involves significant financial risk.")
        print("  You can lose part or all of the capital you deploy.")
        print()
        print("  THE AUTHORS ACCEPT NO RESPONSIBILITY WHATSOEVER FOR ANY")
        print("  FINANCIAL LOSS, MISSED OPPORTUNITY, OR DAMAGE OF ANY KIND")
        print("  ARISING FROM THE USE OF THIS SOFTWARE.")
        print()
        print("  By continuing you acknowledge that you are acting on your own")
        print("  responsibility and at your own risk.")
        print("  ------------------------------------------------------------------")
        print()
        print("  Trade parameters")
        print(f"    Variant       : {variant}")
        print(f"    Allow short   : {allow_short}")
        print(f"    Position size : {pos_pct:.0f}% NAV  (~{capital * pos_pct / 100:,.0f} EUR per trade)")
        print(f"    Max daily loss: {max_loss:,.0f} EUR  (trading halts if hit)")
        print(f"    SL / TP       : {sl_pct:.1f}% / {tp_pct:.1f}%")
        print()
        print("  Orders will be sent to IBKR and will CONSUME REAL FUNDS.")
        print("  Verify IB Gateway is running on the LIVE port (4001).")
        print()
        confirm = input("  Type 'yes' to accept all risks and start live trading: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return
        print()
        confirm2 = input("  Are you sure? Type 'confirmed' to proceed: ")
        if confirm2.strip().lower() != "confirmed":
            print("Aborted.")
            return
        print()
        print("  Live trading confirmed. Starting agent...")
        print("!" * 68)
        print()

    if args.once:
        run_once(paper=not args.live, cfg=cfg)
        return

    # ── Continuous intraday loop ──────────────────────────────────────────────
    monitor_interval = cfg["strategy"].get("monitor_interval_min", 5) * 60
    scan_interval    = cfg["strategy"].get("scan_interval_min", 30) * 60
    eod_str          = cfg["strategy"].get("eod_close_time", "16:30")
    eod_h, eod_m     = int(eod_str.split(":")[0]), int(eod_str.split(":")[1])

    log.info(
        "Agent started — variant=%s  monitor=%dmin  scan=%dmin  EOD=%s CET",
        cfg["model"].get("variant", "?"),
        cfg["strategy"].get("monitor_interval_min", 5),
        cfg["strategy"].get("scan_interval_min", 30),
        eod_str,
    )
    log.info("Press Ctrl+C to stop.")

    last_scan_time      = 0.0    # force immediate scan on startup
    overnight_checked   = False  # run morning check once per market open
    last_cycle_ok       = True   # track whether previous cycle connected OK
    reconnect_pending   = False  # set when connection loss is detected

    while True:
        now = datetime.now(CET)

        # ── EOD close — runs once per day at eod_close_time ──────────────────
        if now.weekday() < 5 and now.hour == eod_h and now.minute >= eod_m and now.minute < eod_m + 5:
            log.info("EOD close time (%s CET) — flattening all positions", eod_str)
            _eod_close(paper=not args.live, cfg=cfg)
            overnight_checked = False   # reset for next day
            reconnect_pending  = False
            last_cycle_ok      = True
            time.sleep(300)             # sleep past the 5-min window
            continue

        if is_market_open(now):
            # ── RECONNECT RECOVERY (highest priority) ─────────────────────
            # If the previous cycle failed due to connection loss, sync positions
            # before doing anything else.  This catches TP/SL fills that happened
            # while we were offline and ensures SL orders are active.
            if reconnect_pending:
                try:
                    _reconnect_evaluate(paper=not args.live, cfg=cfg)
                    reconnect_pending = False
                    last_cycle_ok     = True
                except Exception as e:
                    log.warning("Reconnect evaluation failed: %s — will retry", e)
                    time.sleep(monitor_interval)
                    continue

            # ── MORNING PRIORITY: overnight position check FIRST ──────────
            if not overnight_checked:
                log.info("=== Morning check: evaluating overnight positions ===")
                _evaluate_overnight_positions(paper=not args.live, cfg=cfg)
                overnight_checked = True
                log.info("=== Morning check complete ===")

            # ── Fast loop — position monitor (trail stops, time exits) ─────
            try:
                _monitor_open_positions(paper=not args.live, cfg=cfg)
                last_cycle_ok = True
            except Exception as e:
                if not last_cycle_ok:
                    reconnect_pending = True
                log.warning("Monitor cycle error: %s", e)
                last_cycle_ok = False

            # ── Slow loop — full signal scan + new trades ─────────────────
            if time.time() - last_scan_time >= scan_interval:
                try:
                    run_once(paper=not args.live, cfg=cfg)
                    last_scan_time = time.time()
                    last_cycle_ok  = True
                except Exception as e:
                    log.warning("Scan cycle error: %s", e)
                    last_cycle_ok = False
                    if "connection" in str(e).lower() or "disconnected" in str(e).lower():
                        reconnect_pending = True
                        log.warning("Connection issue detected — will run reconnect evaluation on next cycle")
        else:
            overnight_checked  = False   # market closed — reset for next open
            reconnect_pending  = False
            log.info("Market closed (%s) — waiting...", now.strftime("%H:%M CET"))

        time.sleep(monitor_interval)


if __name__ == "__main__":
    main()
