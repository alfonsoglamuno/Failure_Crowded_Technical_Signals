"""
Position monitor — checks open bracket orders against IBKR fills
and records outcomes to the journal so the learner can act on them.

Call monitor.check_exits() at the start of each daily cycle,
before placing new orders.
"""

from __future__ import annotations

import logging
from datetime import date

from ib_insync import IB

log = logging.getLogger(__name__)


class PositionMonitor:
    def __init__(self, ib: IB, journal, learner, commission: float = 1.75):
        self.ib = ib
        self.journal = journal
        self.learner = learner
        self.commission = commission

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

            # Look for a child order fill (TP or SL) associated with this parent
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

            # Feed outcome to the learner immediately
            self.learner.record_outcome(
                alert_name=trade.get("alert_name", "unknown"),
                action=trade.get("action", "FADE"),
                pnl_net=pnl_net,
            )

            log.info(
                "Exit recorded: %s  entry=%.4f  exit=%.4f  pnl_net=%.2f EUR",
                trade.get("ticker"), entry_price, exit_price, pnl_net,
            )

    def _get_open_journal_trades(self) -> list[dict]:
        """Trades marked 'submitted' or 'filled' with no exit price yet."""
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
                    continue   # skip parent orders and standalone orders
                if trade.orderStatus.status != "Filled":
                    continue
                if order.parentId not in result and trade.fills:
                    fill = trade.fills[-1].execution
                    result[order.parentId] = {
                        "price": fill.avgPrice,
                        "fill_date": fill.time,
                    }

            # ── Cross-session fallback: executions + open-order map ──────────
            if not result:
                fills = self.ib.reqExecutions()
                open_orders = self.ib.reqAllOpenOrders()
                # Build orderId → parentId from currently open orders
                parent_map = {o.orderId: o.parentId
                              for o in open_orders if getattr(o, "parentId", 0)}
                for fill_item in fills:
                    exec_ = fill_item.execution
                    parent_id = parent_map.get(exec_.orderId)
                    if parent_id and parent_id not in result:
                        result[parent_id] = {
                            "price": exec_.avgPrice,
                            "fill_date": exec_.time,
                        }
        except Exception as e:
            log.warning("Could not fetch exit orders: %s", e)
        return result
