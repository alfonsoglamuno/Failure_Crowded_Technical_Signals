"""
Position monitor — checks open bracket orders against IBKR fills,
trails stop-losses, and enforces time-based exits.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from ib_insync import IB, MarketOrder, Stock

log = logging.getLogger(__name__)


def _tick_round(price: float) -> float:
    """Round to nearest 0.01 tick (safe default for Euro STOXX 50 names)."""
    tick = 0.01
    return round(round(price / tick) * tick, 4)


class PositionMonitor:
    def __init__(self, ib: IB, journal, learner, commission: float = 1.75):
        self.ib = ib
        self.journal = journal
        self.learner = learner
        self.commission = commission

    # ── Exit sync ──────────────────────────────────────────────────────────

    def check_exits(self):
        """
        Compare open trade records in the journal against IBKR fills.
        For any trades that have closed (filled exit order), record the outcome.
        """
        open_trades = self._get_open_journal_trades()
        if not open_trades:
            return

        filled_orders = self._get_filled_exit_orders()
        log.info("Monitoring %d open trades, %d filled exits found",
                 len(open_trades), len(filled_orders))

        for trade in open_trades:
            order_id = trade.get("ibkr_order_id")
            if order_id is None:
                continue

            exit_fill = filled_orders.get(order_id)
            if exit_fill is None:
                continue

            exit_price  = exit_fill["price"]
            entry_price = trade.get("entry_price", 0)
            quantity    = trade.get("quantity", 0)
            direction   = trade.get("trade_direction", "BUY")

            if direction == "BUY":
                pnl_gross = (exit_price - entry_price) * quantity
            else:
                pnl_gross = (entry_price - exit_price) * quantity

            pnl_net = pnl_gross - self.commission

            self.journal.update_trade_exit(
                trade_id=trade["id"],
                exit_price=exit_price,
                exit_date=date.today(),
                pnl_gross=pnl_gross,
                commission=self.commission,
            )

            self.learner.record_outcome(
                alert_name=trade.get("alert_name", "unknown"),
                action=trade.get("action", "FADE"),
                pnl_net=pnl_net,
            )

            log.info(
                "Exit recorded: %s  entry=%.4f  exit=%.4f  pnl_net=%.2f EUR",
                trade.get("ticker"), entry_price, exit_price, pnl_net,
            )

    # ── Active position management ─────────────────────────────────────────

    def monitor_positions(self, contracts_cfg: dict, feed, cfg: dict, account: str = ""):
        """
        Trail stops and enforce time-based exits. Called every 5 minutes.

        Trailing stop logic:
          Once unrealized gain >= trail_trigger_pct, raise SL to
          (current_price - trail_step_pct). Never lowers SL.
          Effect: locks in break-even at +0.5%, then keeps trailing upward.

        Time exit:
          If position has been open > max_hold_hours AND gain is still below
          half the trail trigger, close flat. Avoids dead-money drag.
        """
        open_trades = self._get_open_journal_trades()
        if not open_trades:
            log.debug("No open trades to monitor")
            return

        trail_trigger = cfg["risk"].get("trail_trigger_pct", 0.005)
        trail_step    = cfg["risk"].get("trail_step_pct",    0.003)
        max_hold_h    = cfg["risk"].get("max_hold_hours",    4.0)

        # Fetch active stop orders from IBKR, keyed by parentId.
        # Use orderType == "STP" instead of isinstance(StopOrder) because
        # orders received from IBKR on reconnect are plain Order objects,
        # not StopOrder instances — isinstance check always returns False.
        self.ib.reqAllOpenOrders()
        self.ib.sleep(1)
        stop_trades = {
            t.order.parentId: t
            for t in self.ib.trades()
            if getattr(t.order, "orderType", "") == "STP"
            and getattr(t.order, "parentId", 0)
            and t.orderStatus.status not in ("Filled", "Cancelled", "Inactive")
        }
        log.info("Active stop orders tracked: %d  |  Open journal trades: %d",
                 len(stop_trades), len(open_trades))

        for rec in open_trades:
            ticker    = rec.get("ticker", "")
            parent_id = rec.get("ibkr_order_id")
            entry_px  = rec.get("entry_price") or 0
            direction = rec.get("trade_direction", "BUY")

            if not parent_id or not entry_px:
                continue

            spec = contracts_cfg.get(ticker)
            if not spec:
                log.debug("No contract spec for %s — skipping monitor", ticker)
                continue
            contract = Stock(spec["symbol"], "SMART", spec["currency"])

            current_px = feed.get_latest_price(contract, fallback_price=entry_px)
            if current_px <= 0:
                log.warning("No price for %s — skipping monitor", ticker)
                continue

            # Hours since journal entry timestamp (UTC)
            try:
                entry_time = datetime.fromisoformat(rec["ts"]).replace(tzinfo=timezone.utc)
                hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            except Exception:
                hours_open = 0.0

            unrealized_pct = (
                (current_px - entry_px) / entry_px if direction == "BUY"
                else (entry_px - current_px) / entry_px
            )

            log.info("[MONITOR] %s  px=%.4f  entry=%.4f  P&L=%.2f%%  open=%.1fh",
                     ticker, current_px, entry_px, unrealized_pct * 100, hours_open)

            # ── Trailing stop ──────────────────────────────────────────────
            if unrealized_pct >= trail_trigger:
                if parent_id in stop_trades:
                    sl_trade   = stop_trades[parent_id]
                    current_sl = sl_trade.order.auxPrice

                    if direction == "BUY":
                        new_sl = _tick_round(max(current_sl, current_px * (1 - trail_step)))
                    else:
                        new_sl = _tick_round(min(current_sl, current_px * (1 + trail_step)))

                    if new_sl != current_sl:
                        sl_trade.order.auxPrice = new_sl
                        try:
                            # ib_insync: modify by re-submitting with same orderId
                            self.ib.placeOrder(contract, sl_trade.order)
                            log.info("[TRAIL] %s  SL %.4f -> %.4f  (gain +%.2f%%)",
                                     ticker, current_sl, new_sl, unrealized_pct * 100)
                        except Exception as e:
                            log.warning("Trail SL modify failed for %s: %s", ticker, e)
                    else:
                        log.debug("[TRAIL] %s  SL already at max (%.4f)", ticker, current_sl)
                else:
                    log.debug("[TRAIL] %s  no active stop order found (may have filled)", ticker)

            # ── Time exit ─────────────────────────────────────────────────
            if hours_open >= max_hold_h and unrealized_pct < trail_trigger / 2:
                log.info("[TIME EXIT] %s open %.1fh unrealized %.2f%% — closing flat",
                         ticker, hours_open, unrealized_pct * 100)
                self._force_exit(rec, contract, current_px, account)

    def _force_exit(self, rec: dict, contract: Stock, current_px: float, account: str = ""):
        """Cancel bracket children and exit with a market order."""
        parent_id = rec.get("ibkr_order_id")
        direction = rec.get("trade_direction", "BUY")
        qty       = int(rec.get("quantity", 0))

        # Cancel TP and SL child orders first
        for t in self.ib.trades():
            if getattr(t.order, "parentId", 0) == parent_id:
                try:
                    self.ib.cancelOrder(t.order)
                except Exception:
                    pass
        self.ib.sleep(1)

        if qty > 0:
            exit_action = "SELL" if direction == "BUY" else "BUY"
            order = MarketOrder(exit_action, qty)
            order.tif = "DAY"
            if account:
                order.account = account
            self.ib.placeOrder(contract, order)
            log.info("[TIME EXIT] Market %s %d %s @ ~%.4f",
                     exit_action, qty, contract.symbol, current_px)

        # Record estimated exit in journal
        entry_px  = rec.get("entry_price", current_px)
        pnl_gross = ((current_px - entry_px) * qty if direction == "BUY"
                     else (entry_px - current_px) * qty)
        self.journal.update_trade_exit(
            trade_id=rec["id"],
            exit_price=current_px,
            exit_date=date.today(),
            pnl_gross=pnl_gross,
            commission=self.commission,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_open_journal_trades(self) -> list[dict]:
        """Trades marked submitted/filled with no exit price yet."""
        all_trades = self.journal.get_recent_trades(n=50)
        return [
            t for t in all_trades
            if t.get("status") in ("submitted", "filled")
            and t.get("exit_price") is None
        ]

    def _get_filled_exit_orders(self) -> dict[int, dict]:
        """
        Returns {parent_order_id: {price, fill_date}} for completed exits.

        Checks session trades first (fast, covers same-session brackets),
        then falls back to reqExecutions + reqAllOpenOrders for cross-session
        fills (e.g. SL/TP triggered overnight while agent was offline).
        """
        result = {}
        try:
            # ── Same-session: ib.trades() has full Order objects with parentId ──
            for trade in self.ib.trades():
                order = trade.order
                if not getattr(order, "parentId", 0):
                    continue
                if trade.orderStatus.status != "Filled":
                    continue
                if order.parentId not in result and trade.fills:
                    fill = trade.fills[-1].execution
                    result[order.parentId] = {
                        "price": fill.avgPrice,
                        "fill_date": fill.time,
                    }

            # ── Cross-session fallback: position-based detection ─────────────
            # reqAllOpenOrders() only returns OPEN orders — once SL/TP fires,
            # those orders vanish and we lose the parentId link.
            # Instead: if journal says a symbol is open but IBKR has no position,
            # the trade was closed. Find the exit price from today's executions.
            current_pos_symbols = {
                pos.contract.symbol
                for pos in self.ib.positions()
                if pos.position > 0
            }
            all_fills = self.ib.reqExecutions()
            # Most recent SELL fill per symbol
            sell_fills: dict[str, dict] = {}
            for fi in all_fills:
                exec_ = fi.execution
                side  = getattr(exec_, "side", "")
                if side in ("SLD", "SELL"):
                    sym = fi.contract.symbol
                    if sym not in sell_fills:
                        sell_fills[sym] = {
                            "price": exec_.avgPrice,
                            "fill_date": exec_.time,
                        }

            open_trades = self._get_open_journal_trades()
            for t in open_trades:
                parent_id   = t.get("ibkr_order_id")
                ibkr_symbol = t.get("ibkr_symbol", "")
                if not parent_id or not ibkr_symbol or parent_id in result:
                    continue
                if ibkr_symbol not in current_pos_symbols and ibkr_symbol in sell_fills:
                    result[parent_id] = sell_fills[ibkr_symbol]
                    log.info("Cross-session exit detected: %s @ %.4f",
                             ibkr_symbol, sell_fills[ibkr_symbol]["price"])

        except Exception as e:
            log.warning("Could not fetch exit orders: %s", e)
        return result
