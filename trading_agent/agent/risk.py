"""
Risk manager — position sizing and pre-trade guardrails.
"""

from __future__ import annotations

import logging

from agent.strategy import Signal

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: dict, horizon_days: int = 1):
        r = cfg["risk"]
        self.max_daily_loss   = r["max_daily_loss_eur"]
        self.max_position_pct = r["max_position_pct"]
        self.min_trade_eur    = r["min_trade_size_eur"]
        self._commission_rate = r.get("commission_rate_pct", 0.05) / 100
        self._commission_min  = r.get("commission_min_eur", 2.0)

        # Per-horizon overrides: h1d / h3d / h5d sections in config take priority
        # over the top-level defaults. This lets each model type use SL/TP levels
        # calibrated to its expected hold duration and typical price range.
        h_key = f"h{horizon_days}d"
        overrides = r.get(h_key, {})
        self.stop_loss_pct   = overrides.get("stop_loss_pct",   r["stop_loss_pct"])
        self.take_profit_pct = overrides.get("take_profit_pct", r["take_profit_pct"])
        self.horizon_days    = horizon_days

        if overrides:
            log.info(
                "RiskManager: horizon=%dd  SL=%.1f%%  TP=%.1f%%  (from h%dd overrides)",
                horizon_days, self.stop_loss_pct * 100,
                self.take_profit_pct * 100, horizon_days,
            )
        else:
            log.info(
                "RiskManager: horizon=%dd  SL=%.1f%%  TP=%.1f%%  (defaults)",
                horizon_days, self.stop_loss_pct * 100, self.take_profit_pct * 100,
            )

    def estimate_commission(self, trade_value: float) -> float:
        """IBKR tiered: 0.05% of trade value, minimum 2 EUR."""
        return max(self._commission_min, trade_value * self._commission_rate)

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

        # Net expected return after commission (entry + exit both charged)
        expected_gross = quantity * abs(take_profit - entry)
        round_trip_commission = self.estimate_commission(position_value) * 2
        if expected_gross < round_trip_commission:
            log.info(
                "Expected gross (%.2f EUR) < round-trip commission (%.2f EUR) for %s — skipping",
                expected_gross, round_trip_commission, signal.ticker
            )
            return None

        return {
            "quantity": quantity,   # integer whole shares
            "entry_price": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "position_value_eur": round(position_value, 2),
        }

    def check_liquidity(
        self,
        direction: str,
        quantity: int,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        ticker: str = "",
    ) -> dict:
        """
        Pre-trade L1 liquidity check.

        For a BUY market order the available liquidity is at the ask side.
        For a SELL market order the available liquidity is at the bid side.

        Returns a dict with:
          estimated_fill_price : best estimate of where the market order fills
          available_size       : shares available at the quoted price
          sweep_risk           : True if qty > available L1 size (order walks the book)
          spread_pct           : bid/ask spread as a fraction of mid price
        """
        if direction == "BUY":
            quoted_price   = ask
            available_size = ask_size
        else:
            quoted_price   = bid
            available_size = bid_size

        mid   = (bid + ask) / 2 if bid > 0 and ask > 0 else quoted_price
        sweep = available_size > 0 and quantity > available_size
        spread_pct = ((ask - bid) / mid) if mid > 0 else 0.0

        if sweep:
            log.warning(
                "[LIQUIDITY] %s %s %d shares — only %.0f available at L1 quote %.4f. "
                "Market order may fill at a worse average price.",
                ticker, direction, quantity, available_size, quoted_price,
            )
        if spread_pct > 0.002:   # spread > 0.20% — wider than typical for STOXX50
            log.warning(
                "[LIQUIDITY] %s spread is %.2f%% (bid=%.4f ask=%.4f) — "
                "wider than usual, consider limit order.",
                ticker, spread_pct * 100, bid, ask,
            )

        return {
            "estimated_fill_price": quoted_price,
            "available_size": available_size,
            "sweep_risk": sweep,
            "spread_pct": spread_pct,
        }
