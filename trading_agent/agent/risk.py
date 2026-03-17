"""
Risk manager — position sizing and pre-trade guardrails.
"""

from __future__ import annotations

import logging

from agent.strategy import Signal

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: dict):
        r = cfg["risk"]
        self.max_daily_loss = r["max_daily_loss_eur"]
        self.max_position_pct = r["max_position_pct"]
        self.stop_loss_pct = r["stop_loss_pct"]
        self.take_profit_pct = r["take_profit_pct"]
        self.min_trade_eur = r["min_trade_size_eur"]
        self.commission = r["commission_per_trade_eur"]

    def check_daily_loss(self, daily_pnl: float) -> bool:
        """Returns True if we can still trade today."""
        if daily_pnl <= -self.max_daily_loss:
            log.warning("Daily loss limit hit (%.2f EUR). No more trades today.", daily_pnl)
            return False
        return True

    def size_position(
        self,
        signal: Signal,
        nav: float,
        current_price: float,
    ) -> dict | None:
        """
        Compute position size and bracket order prices.

        Returns dict with:
          quantity: number of shares (fractional)
          entry_price: market order (current_price)
          stop_loss: price
          take_profit: price

        Returns None if the trade is too small to be worth it.
        """
        capital_to_risk = nav * self.max_position_pct

        # Entry at current price (market order at open)
        entry = current_price
        if entry <= 0:
            log.warning("Invalid price %.4f for %s", entry, signal.ticker)
            return None

        if signal.trade_direction == "BUY":
            stop_loss = entry * (1 - self.stop_loss_pct)
            take_profit = entry * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss = entry * (1 + self.stop_loss_pct)
            take_profit = entry * (1 - self.take_profit_pct)

        # Position size: risk (stop distance) limited to max_position_pct of NAV
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return None

        # Shares = capital_to_risk / entry — round DOWN to whole shares
        # (European exchanges do not support fractional share orders)
        quantity = max(1, int(capital_to_risk / entry))

        # Sanity: total position value
        position_value = quantity * entry
        if position_value < self.min_trade_eur:
            log.info(
                "Position too small (%.2f EUR) for %s — skipping",
                position_value, signal.ticker
            )
            return None

        # Net expected return after commission
        expected_gross = quantity * abs(take_profit - entry)
        if expected_gross < self.commission * 2:
            log.info(
                "Expected gross (%.2f EUR) < 2x commission for %s — skipping",
                expected_gross, signal.ticker
            )
            return None

        return {
            "quantity": quantity,   # integer whole shares
            "entry_price": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "position_value_eur": round(position_value, 2),
        }
