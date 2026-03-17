"""
IBKR order executor — places bracket orders (entry + stop-loss + take-profit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, BracketOrder

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    yahoo_ticker: str
    ibkr_symbol: str
    action: str               # FADE or FOLLOW
    trade_direction: str      # BUY or SELL
    quantity: float
    entry_price: float        # pre-order quote used for SL/TP sizing
    stop_loss: float
    take_profit: float
    parent_order_id: int
    status: str               # submitted / filled / error
    error_msg: str = ""
    actual_fill_price: float = 0.0   # actual average fill price from IBKR (0 = not yet filled)
    slippage_pct: float = 0.0        # (actual_fill - entry_price) / entry_price


class IBKRExecutor:
    def __init__(self, ib: IB, contracts_cfg: dict, paper: bool = True,
                 account: str = "", allow_short: bool = False):
        self.ib = ib
        self.contracts_cfg = contracts_cfg
        self.paper = paper
        self.account = account       # IBKR account ID — set on every order for correctness
        self.allow_short = allow_short

    def _make_contract(self, yahoo_ticker: str) -> Optional[Stock]:
        spec = self.contracts_cfg.get(yahoo_ticker)
        if not spec:
            return None
        return Stock(spec["symbol"], spec["exchange"], spec["currency"])

    def place_bracket(
        self,
        yahoo_ticker: str,
        trade_direction: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        action: str = "",
        contract: Optional[Stock] = None,
    ) -> OrderResult:
        """
        Place a bracket order: market entry + stop-loss + take-profit limit.

        Pass a pre-qualified `contract` from run_agent to skip re-lookup and
        benefit from the conId already resolved by qualifyContracts().
        """
        # ── Hard safety: block short entries unconditionally unless explicitly enabled ──
        # The agent's design is LONG-only (fading = BUY the dip, not shorting the spike).
        # Any SELL entry here means a short position — which IBKR may fill but cannot
        # be borrowed reliably on European stocks. Block it hard so config bugs or
        # signal inversions cannot accidentally create short positions.
        if trade_direction == "SELL" and not self.allow_short:
            log.error(
                "BLOCKED short entry for %s — allow_short=False. "
                "Only SELL exits (SL/TP/EOD) are permitted, not new short entries.",
                yahoo_ticker,
            )
            return OrderResult(
                yahoo_ticker=yahoo_ticker,
                ibkr_symbol="",
                action=action,
                trade_direction=trade_direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                parent_order_id=-1,
                status="error",
                error_msg="Short entries disabled (allow_short=False in config)",
            )

        if contract is None:
            contract = self._make_contract(yahoo_ticker)
        if contract is None:
            return OrderResult(
                yahoo_ticker=yahoo_ticker,
                ibkr_symbol="",
                action=action,
                trade_direction=trade_direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                parent_order_id=-1,
                status="error",
                error_msg=f"No IBKR contract mapping for {yahoo_ticker}",
            )

        # Use SMART routing for order placement — avoids exchange-direct restrictions
        # (IB Gateway Precautionary Settings error 10311) and resolves conId-based
        # contracts regardless of which exchange they qualified on.
        # IBKR's SMART router uses conId to identify the exact contract and routes
        # to the best-priced exchange automatically.
        if getattr(contract, "conId", 0) and contract.exchange != "SMART":
            order_contract = Stock(contract.symbol, "SMART", contract.currency)
            order_contract.conId = contract.conId
        else:
            order_contract = contract

        ib_action = trade_direction        # "BUY" or "SELL"
        ib_exit_action = "SELL" if ib_action == "BUY" else "BUY"
        qty = int(round(quantity))         # whole shares only — required by European exchanges

        def _tick_round(price: float) -> float:
            """Round to nearest valid tick size for European equities.

            IBKR uses MTA tick tables; for stocks >= 1 EUR the common sizes are:
              price < 1   → 0.0001   price 1-10  → 0.001
              price 10-50 → 0.005    price >= 50 → 0.01
            Using 0.01 as the safe default covers most Euro STOXX 50 names.
            """
            tick = 0.01
            return round(round(price / tick) * tick, 4)

        try:
            # ── Step 1: market entry (transmit=False — hold until children are set) ──
            parent = MarketOrder(ib_action, qty)
            parent.transmit = False
            parent.outsideRth = False      # RTH only — no pre/post market fills
            parent.tif = "DAY"             # explicit TIF prevents order-preset override (error 10349)
            if self.account:
                parent.account = self.account

            parent_trade = self.ib.placeOrder(order_contract, parent)
            self.ib.sleep(0.5)             # wait for TWS to assign a valid orderId

            parent_id = parent.orderId
            if not parent_id:
                raise RuntimeError(f"TWS did not assign an orderId for {yahoo_ticker} parent order")

            # ── Step 2: take-profit limit order ──────────────────────────────────────
            tp = LimitOrder(ib_exit_action, qty, _tick_round(take_profit))
            tp.parentId = parent_id
            tp.transmit = False
            tp.outsideRth = False
            tp.tif = "GTC"                 # stay open until hit or manually cancelled
            if self.account:
                tp.account = self.account

            # ── Step 3: stop-loss order — transmit=True releases the whole bracket ──
            sl = StopOrder(ib_exit_action, qty, _tick_round(stop_loss))
            sl.parentId = parent_id
            sl.transmit = True
            sl.outsideRth = False
            sl.tif = "GTC"                 # stay open until hit or manually cancelled
            if self.account:
                sl.account = self.account

            self.ib.placeOrder(order_contract, tp)
            self.ib.placeOrder(order_contract, sl)
            self.ib.sleep(1)               # allow TWS to confirm all three

            # Verify the parent order was accepted
            all_trades = {t.order.orderId: t for t in self.ib.trades()}
            parent_status = all_trades.get(parent_id)
            status_str = parent_status.orderStatus.status if parent_status else "Unknown"

            if status_str in ("Inactive", "ApiCancelled", "Cancelled"):
                raise RuntimeError(f"Order rejected by IBKR — status: {status_str}")

            # ── Poll for actual fill price (market orders typically fill within 1-2s) ──
            # We wait up to 5 seconds to capture the actual average fill price.
            # This matters because the SL/TP were anchored to entry_price (the quote),
            # but the real cost basis is the fill price. We log slippage for every trade.
            actual_fill = 0.0
            slippage = 0.0
            for _ in range(10):
                self.ib.sleep(0.5)
                pt = {t.order.orderId: t for t in self.ib.trades()}.get(parent_id)
                if pt and pt.orderStatus.status == "Filled" and pt.fills:
                    actual_fill = float(pt.fills[-1].execution.avgPrice)
                    break

            if actual_fill > 0 and entry_price > 0:
                slippage = (actual_fill - entry_price) / entry_price
                slippage_dir = "+" if slippage >= 0 else ""
                log.info(
                    "[%s] Fill confirmed: %s %s qty=%d  quote=%.4f  fill=%.4f  "
                    "slippage=%s%.4f%%  SL=%.4f  TP=%.4f  orderId=%d",
                    "PAPER" if self.paper else "LIVE",
                    ib_action, yahoo_ticker, qty,
                    entry_price, actual_fill,
                    slippage_dir, slippage * 100,
                    stop_loss, take_profit, parent_id,
                )
                if abs(slippage) > 0.001:   # > 0.10% — warn and note for re-anchor
                    log.warning(
                        "[SLIPPAGE] %s fill slippage %.4f%% exceeds 0.10%% — "
                        "SL/TP anchored to quote (%.4f), actual cost basis is %.4f",
                        yahoo_ticker, slippage * 100, entry_price, actual_fill,
                    )
            else:
                # Order submitted but not yet filled (e.g., pre-open session)
                status_str = "submitted"
                log.info(
                    "[%s] Bracket submitted (fill pending): %s %s qty=%d "
                    "quote=%.4f SL=%.4f TP=%.4f orderId=%d",
                    "PAPER" if self.paper else "LIVE",
                    ib_action, yahoo_ticker, qty,
                    entry_price, stop_loss, take_profit, parent_id,
                )

            return OrderResult(
                yahoo_ticker=yahoo_ticker,
                ibkr_symbol=contract.symbol,
                action=action,
                trade_direction=trade_direction,
                quantity=qty,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                parent_order_id=parent_id,
                status="filled" if actual_fill > 0 else "submitted",
                actual_fill_price=actual_fill,
                slippage_pct=slippage,
            )

        except Exception as e:
            log.error("Order placement failed for %s: %s", yahoo_ticker, e)
            return OrderResult(
                yahoo_ticker=yahoo_ticker,
                ibkr_symbol=contract.symbol if contract else "",
                action=action,
                trade_direction=trade_direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                parent_order_id=-1,
                status="error",
                error_msg=str(e),
            )

    def cancel_all_open_orders(self):
        """Cancel all open orders — called on daily shutdown."""
        open_orders = self.ib.reqAllOpenOrders()
        for order in open_orders:
            self.ib.cancelOrder(order.order)
        log.info("Cancelled %d open orders", len(open_orders))

    def get_open_positions(self) -> list:
        return self.ib.positions()
