"""
Trading agent main entry point.

Usage:
    python run_agent.py --paper          # paper trading (recommended first)
    python run_agent.py --live           # live trading (use with caution)
    python run_agent.py --status         # print journal summary, no trading
    python run_agent.py --paper --once   # run once now, don't loop

Continuous intraday mode (default):
    Scans every scan_interval_min minutes during market hours, closes all
    positions at eod_close_time CET.  Press Ctrl+C to stop.
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

_PEER_CORR_THRESHOLD = 0.70   # skip candidate if corr with any open position exceeds this
_PEER_CORR_WINDOW    = 20     # trading days for correlation lookback


def _max_corr_with_open(candidate: str, open_tickers: set[str],
                         universe_data: dict) -> float:
    """
    Return the maximum 20-day return correlation between `candidate` and any
    currently-open position. Used to avoid concentrating correlated exposure.

    Returns 0.0 if open_tickers is empty or data is insufficient.
    """
    if not open_tickers or candidate not in universe_data:
        return 0.0

    import pandas as pd
    cand_df = universe_data[candidate]
    if "close" not in cand_df.columns or len(cand_df) < _PEER_CORR_WINDOW + 2:
        return 0.0

    cand_ret = cand_df["close"].pct_change().dropna().tail(_PEER_CORR_WINDOW * 2)
    max_corr = 0.0

    for open_t in open_tickers:
        if open_t not in universe_data:
            continue
        peer_df = universe_data[open_t]
        if "close" not in peer_df.columns or len(peer_df) < _PEER_CORR_WINDOW + 2:
            continue
        peer_ret = peer_df["close"].pct_change().dropna().tail(_PEER_CORR_WINDOW * 2)
        aligned = pd.concat([cand_ret, peer_ret], axis=1, join="inner")
        if len(aligned) < _PEER_CORR_WINDOW:
            continue
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if not __import__("math").isnan(corr):
            max_corr = max(max_corr, abs(corr))

    return max_corr


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
              f"P={s['failure_proba']:.3f} → {s['action']}")
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

    risk = RiskManager(cfg)
    learner = AdaptiveLearner(cfg, journal, predictor)

    # ── Connect to IBKR ──────────────────────────────────────────────────────
    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
    except ConnectionError as e:
        log.error("IBKR connection failed: %s", e)
        return

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
                                  commission=cfg["risk"]["commission_per_trade_eur"])
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
        signals = make_signals(
            feature_df, probas,
            fade_threshold=learner.fade_threshold,
            follow_threshold=learner.follow_threshold,
            horizon_days=cfg["strategy"]["default_horizon"],
            crowding_min_score=cfg["strategy"].get("crowding_min_score", 0.30),
            regime_threshold_boost=cfg["strategy"].get("regime_threshold_boost", 0.05),
        )
        signals = filter_signals(
            signals,
            max_trades=cfg["risk"]["max_trades_per_day"],
            allow_short=cfg["strategy"].get("allow_short", False),
        )

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

        # Dedup: don't re-trade a ticker that still has an active (unclosed) trade today.
        # Includes filled trades with no exit yet (position open but SL/TP pending).
        already_traded_today = {
            t["ticker"] for t in journal.get_recent_trades(100)
            if str(t.get("date", "")) == str(today)
               and t.get("status") in ("submitted", "open", "pending", "filled")
               and t.get("exit_price") is None
        }
        if already_traded_today:
            log.info("Already traded today: %s — will not re-enter", sorted(already_traded_today))

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

            # Dedup — skip tickers already traded in an earlier scan today
            if sig.ticker in already_traded_today:
                log.info("Skipping %s — already have an open trade from today's earlier scan", sig.ticker)
                continue

            # Don't add to an existing long position
            if sig.ticker in open_ibkr_positions:
                log.info("Skipping %s — already holding long position", sig.ticker)
                continue

            # Correlation gate — avoid concentrating correlated positions.
            # If the candidate is highly correlated (>0.70) with any open position,
            # adding it is just leveraging the same factor — skip it.
            max_corr = _max_corr_with_open(sig.ticker, open_ibkr_positions, universe_data)
            if max_corr >= _PEER_CORR_THRESHOLD:
                log.info(
                    "Skipping %s — max corr with open positions %.2f >= %.2f threshold",
                    sig.ticker, max_corr, _PEER_CORR_THRESHOLD,
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

            # Get current price; fall back to last OHLCV close if live quote unavailable
            ohlcv = universe_data.get(sig.ticker, __import__("pandas").DataFrame())
            fallback = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 0.0
            current_price = feed.get_latest_price(contract, fallback_price=fallback)

            if current_price <= 0:
                log.warning("No price available for %s — skipping", sig.ticker)
                continue

            sizing = risk.size_position(sig, nav, current_price)
            if sizing is None:
                continue

            log.info(
                "[SIGNAL] %s %s P(failure)=%.3f crowd=%.2f -> %s "
                "qty=%d entry=%.4f SL=%.4f TP=%.4f%s",
                sig.action, sig.ticker, sig.failure_proba, sig.crowding_score,
                sig.trade_direction, sizing["quantity"], sizing["entry_price"],
                sizing["stop_loss"], sizing["take_profit"],
                " [regime-boost]" if sig.regime_boost_applied else "",
            )

            result = executor.place_bracket(
                yahoo_ticker=sig.ticker,
                trade_direction=sig.trade_direction,
                quantity=sizing["quantity"],
                entry_price=sizing["entry_price"],
                stop_loss=sizing["stop_loss"],
                take_profit=sizing["take_profit"],
                action=sig.action,
                contract=contract,   # pass pre-qualified contract
            )

            signal_id = signal_id_map.get(f"{sig.ticker}|{sig.alert_name}", -1)
            journal.log_trade(
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
            )

            if result.status == "submitted":
                n_trades += 1
                log.info("Order submitted for %s (orderId=%d)", sig.ticker, result.parent_order_id)
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
    Run once at market open if journal has positions from previous session.

    For each overnight position:
      1. Check if it's still valid in IBKR (position > 0).
      2. Ensure a stop-loss order exists at a sensible level.
         If SL is missing (e.g. bracket cancelled by EOD), re-place it.
      3. Log a warning for accidental short positions (should be covered).

    This does NOT make trading decisions — it just protects open positions
    until the first full signal scan runs and can evaluate them properly.
    """
    from agent.data_feed import IBKRFeed, load_contracts
    from agent.journal import Journal
    from ib_insync import Stock, StopOrder
    from datetime import date

    journal = Journal(cfg["journal"]["db_path"])
    today = date.today()

    open_trades = [
        t for t in journal.get_recent_trades(50)
        if t.get("status") in ("submitted", "filled")
        and t.get("exit_price") is None
        and str(t.get("date", "")) != str(today)   # from previous day
    ]
    if not open_trades:
        return

    log.warning("Found %d overnight positions from previous session — evaluating", len(open_trades))
    contracts_cfg = load_contracts(cfg["universe"]["ibkr_contracts_file"])

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        ib = feed.ib
        ib.sleep(2)
        account = feed.account_id

        # Map IBKR symbol → current position qty
        ibkr_positions = {
            pos.contract.symbol: pos.position
            for pos in ib.positions()
        }

        # Existing SL orders keyed by parentId
        ib.reqAllOpenOrders()
        ib.sleep(1)
        existing_stops = {
            t.order.parentId: t.order.auxPrice
            for t in ib.trades()
            if getattr(t.order, "orderType", "") == "STP"
            and getattr(t.order, "parentId", 0)
            and t.orderStatus.status not in ("Filled", "Cancelled", "Inactive")
        }

        sl_pct = cfg["risk"].get("stop_loss_pct", 0.015)

        for rec in open_trades:
            ticker    = rec.get("ticker", "")
            symbol    = rec.get("ibkr_symbol", "")
            parent_id = rec.get("ibkr_order_id")
            direction = rec.get("trade_direction", "BUY")
            qty       = int(rec.get("quantity", 0))
            entry_px  = rec.get("entry_price", 0)

            ibkr_qty = ibkr_positions.get(symbol, 0)

            if ibkr_qty == 0:
                # Position already closed in IBKR but journal not updated — will be
                # reconciled by check_exits; nothing to do here
                log.info("Overnight %s: IBKR shows 0 position — check_exits will reconcile", ticker)
                continue

            if ibkr_qty < 0:
                log.warning("Overnight %s: accidental SHORT (%d shares) — covering at open", ticker, ibkr_qty)
                spec = contracts_cfg.get(ticker)
                if spec:
                    contract = Stock(spec["symbol"], "SMART", spec["currency"])
                    from ib_insync import MarketOrder
                    order = MarketOrder("BUY", abs(ibkr_qty))
                    order.account = account
                    order.tif = "DAY"
                    ib.placeOrder(contract, order)
                continue

            # Check SL
            if parent_id in existing_stops:
                log.info("Overnight %s: SL already active @ %.4f", ticker, existing_stops[parent_id])
                continue

            # SL missing — re-place at current price - SL%
            spec = contracts_cfg.get(ticker)
            if not spec:
                log.warning("Overnight %s: no contract spec — cannot re-place SL", ticker)
                continue

            contract = Stock(spec["symbol"], "SMART", spec["currency"])
            current_px = feed.get_latest_price(contract, fallback_price=entry_px)

            if direction == "BUY":
                sl_price = round(current_px * (1 - sl_pct), 2)
                sl_order = StopOrder("SELL", qty, sl_price)
            else:
                sl_price = round(current_px * (1 + sl_pct), 2)
                sl_order = StopOrder("BUY", qty, sl_price)

            sl_order.account  = account
            sl_order.tif      = "DAY"
            sl_order.parentId = parent_id
            ib.placeOrder(contract, sl_order)
            log.warning("Overnight %s: re-placed SL @ %.4f (%.1f%% from %.4f)",
                        ticker, sl_price, sl_pct * 100, current_px)

        ib.sleep(2)
    except Exception as e:
        log.error("Overnight position evaluation failed: %s", e)
    finally:
        feed.disconnect()


def _eod_close(paper: bool, cfg: dict):
    """Cancel all bracket orders and close all open positions with market orders.

    Handles both longs (SELL) and accidental shorts (BUY to cover).
    Uses ib.positions() after a settle sleep to ensure complete position data.
    """
    from agent.data_feed import IBKRFeed
    from ib_insync import MarketOrder
    from dotenv import load_dotenv
    load_dotenv()

    feed = IBKRFeed(cfg, paper=paper)
    try:
        feed.connect()
        ib = feed.ib
        account = feed.account_id

        # Give IBKR time to push all position/order data after reconnect
        ib.sleep(2)

        # Cancel all open orders (TP/SL brackets and any stale orders)
        open_orders = ib.reqAllOpenOrders()
        ib.sleep(1)
        cancelled = 0
        for t in open_orders:
            try:
                ib.cancelOrder(t.order)
                cancelled += 1
            except Exception:
                pass
        ib.sleep(2)
        log.info("Cancelled %d open orders", cancelled)

        # Close all non-zero positions
        positions = ib.positions()
        closed_long = 0
        closed_short = 0
        for pos in positions:
            qty = int(pos.position)
            if qty == 0:
                continue
            if qty > 0:
                # Long: sell to close
                order = MarketOrder("SELL", qty)
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(pos.contract, order)
                log.info("EOD close LONG: SELL %d %s", qty, pos.contract.symbol)
                closed_long += 1
            else:
                # Short (accidental): buy to cover
                order = MarketOrder("BUY", abs(qty))
                order.account = account
                order.tif = "DAY"
                ib.placeOrder(pos.contract, order)
                log.warning("EOD close SHORT: BUY %d %s (accidental short — covering)",
                            abs(qty), pos.contract.symbol)
                closed_short += 1

        ib.sleep(3)
        log.info("EOD close complete — %d longs closed, %d shorts covered",
                 closed_long, closed_short)

    except Exception as e:
        log.error("EOD close failed: %s", e)
    finally:
        feed.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Crowded Signal Failure Trading Agent")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading mode (real money)")
    parser.add_argument("--once", action="store_true",
                        help="Run once immediately, do not schedule")
    parser.add_argument("--status", action="store_true",
                        help="Print journal status and exit")
    parser.add_argument("--eod-close", action="store_true",
                        help="Cancel all brackets and close all longs now (EOD safety net)")
    args = parser.parse_args()

    if args.live and args.paper:
        args.paper = False   # --live takes precedence

    cfg = load_config()

    if args.status:
        from agent.journal import Journal
        journal = Journal(cfg["journal"]["db_path"])
        print_status(journal)
        return

    if args.eod_close:
        _eod_close(paper=not args.live, cfg=cfg)
        return

    if args.live:
        log.warning("⚠  LIVE TRADING MODE — real money will be used")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    if args.once:
        run_once(paper=not args.live, cfg=cfg)
        return

    # Dual-speed intraday loop:
    #   Fast  (monitor_interval_min =  5 min): trail stops + time exits
    #   Slow  (scan_interval_min    = 30 min): full signal scan + new trades
    monitor_interval = cfg["strategy"].get("monitor_interval_min", 5) * 60
    scan_interval    = cfg["strategy"].get("scan_interval_min", 30) * 60
    eod_str          = cfg["strategy"].get("eod_close_time", "16:30")
    eod_h, eod_m     = int(eod_str.split(":")[0]), int(eod_str.split(":")[1])

    log.info(
        "Intraday monitor started — position check every %d min, "
        "signal scan every %d min, EOD close at %s CET (Ctrl+C to stop)",
        cfg["strategy"].get("monitor_interval_min", 5),
        cfg["strategy"].get("scan_interval_min", 30),
        eod_str,
    )

    last_scan_time    = 0.0    # force immediate scan on startup
    overnight_checked = False  # run overnight check once per market open

    while True:
        now = datetime.now(CET)

        # EOD close — runs once when we cross the EOD time
        if now.weekday() < 5 and now.hour == eod_h and now.minute >= eod_m and now.minute < eod_m + 5:
            log.info("EOD close time reached (%s CET) — closing all open positions", eod_str)
            _eod_close(paper=not args.live, cfg=cfg)
            overnight_checked = False   # reset so next day's open triggers check
            time.sleep(300)   # sleep past the 5-min window
            continue

        if is_market_open(now):
            # ── MORNING PRIORITY: evaluate any overnight positions FIRST ──────
            # Always runs before any new signal scan or trade placement.
            # Covers shorts (cover immediately), missing SLs (re-place), and
            # positions that were closed by IBKR SL/TP while agent was offline.
            if not overnight_checked:
                log.info("=== Morning check: evaluating overnight positions ===")
                _evaluate_overnight_positions(paper=not args.live, cfg=cfg)
                overnight_checked = True
                log.info("=== Morning check complete ===")


            # Fast loop — always runs (trail stops, time exits, exit sync)
            _monitor_open_positions(paper=not args.live, cfg=cfg)

            # Slow loop — full signal scan every scan_interval
            if time.time() - last_scan_time >= scan_interval:
                run_once(paper=not args.live, cfg=cfg)
                last_scan_time = time.time()
        else:
            overnight_checked = False   # market closed — reset flag for next open
            log.info("Market closed at %s — waiting...", now.strftime("%H:%M CET"))

        time.sleep(monitor_interval)


if __name__ == "__main__":
    main()
