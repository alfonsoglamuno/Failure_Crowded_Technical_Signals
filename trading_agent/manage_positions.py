"""
Intraday position management — inspect, selectively close, or emergency-flatten.

Use this script any time you want to manually intervene in open positions:
  - You see bad news overnight and want to cut a long before market open
  - You want to close a specific ticker before EOD
  - Emergency: flatten everything immediately
  - EOD close: same as agent --eod-close but callable independently

This extends close_shorts.py with support for all position types and
explicit overnight-news handling.

Usage:
    python manage_positions.py                          # show positions, interactive menu
    python manage_positions.py --ticker ASML.AS IFX.DE # close specific tickers
    python manage_positions.py --eod-close              # cancel all orders + flatten all
    python manage_positions.py --emergency              # same as --eod-close (alias)
    python manage_positions.py --shorts-only            # cover accidental shorts only
    python manage_positions.py --live ...               # real money (extra confirmation)

Always paper unless --live is explicitly passed.
Orders use DAY TIF — if markets are closed they expire; re-run at next open.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from dotenv import load_dotenv
load_dotenv()

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [yes/no]: ").strip().lower() == "yes"


def _pnl_str(val: float) -> str:
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def _show_positions(positions: list) -> None:
    """Print a formatted position table."""
    has_shorts = any(p.position < 0 for p in positions)
    print(f"\n{'Symbol':<12} {'Qty':>8} {'Type':<12} {'AvgCost':>10} {'UnrealPNL':>12} {'Currency':>9}")
    print("-" * 68)
    for p in positions:
        kind = "SHORT(*)" if p.position < 0 else "LONG"
        ccy  = getattr(p.contract, "currency", "?")
        print(
            f"  {p.contract.symbol:<10} {p.position:>8.0f}  {kind:<12} "
            f"{p.averageCost:>10.4f} {_pnl_str(p.unrealizedPNL):>12}  {ccy:>9}"
        )
    if has_shorts:
        print("\n  (*) Accidental short — agent never intentionally creates these.")
        print("      Cover with: python manage_positions.py --shorts-only")
    print()


def run(
    tickers: list[str] | None,
    shorts_only: bool,
    eod_close: bool,
    paper: bool,
) -> None:
    from agent.data_feed import IBKRFeed
    from ib_insync import MarketOrder

    feed = IBKRFeed(cfg, paper=paper)
    feed.connect()
    ib   = feed.ib
    account = feed.account_id

    ib.sleep(2)  # allow IBKR to push full position data

    all_positions = [p for p in ib.positions() if abs(p.position) > 0]
    if not all_positions:
        print("No open positions.")
        feed.disconnect()
        return

    _show_positions(all_positions)

    # ── Determine what to close ────────────────────────────────────────────────
    if eod_close:
        # Cancel ALL open orders first, then flatten everything (same as agent EOD)
        print("EOD / Emergency close — cancelling all open orders first...")
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
        print(f"  Cancelled {cancelled} open order(s).")
        to_close = all_positions
        label    = "ALL positions (EOD/emergency)"

    elif shorts_only:
        to_close = [p for p in all_positions if p.position < 0]
        label    = "all SHORT positions"

    elif tickers:
        # Accept both Yahoo format (ASML.AS) and IBKR symbol (ASML)
        requested_syms = {t.split(".")[0].upper() for t in tickers}
        to_close  = [p for p in all_positions
                     if p.contract.symbol.upper() in requested_syms]
        not_found = requested_syms - {p.contract.symbol.upper() for p in to_close}
        if not_found:
            print(f"Warning: {sorted(not_found)} not found in current positions.")
        label = f"specified: {[p.contract.symbol for p in to_close]}"

    else:
        # Interactive menu
        print("Options:")
        print("  1. Close ALL positions (EOD / emergency)")
        print("  2. Cover accidental SHORTS only")
        print("  3. Close specific symbol(s)")
        print("  4. Cancel — do nothing")
        choice = input("Choice [1/2/3/4]: ").strip()
        if choice == "1":
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
            print(f"  Cancelled {cancelled} open order(s).")
            to_close = all_positions
            label    = "ALL positions"
        elif choice == "2":
            to_close = [p for p in all_positions if p.position < 0]
            label    = "SHORT positions"
        elif choice == "3":
            raw  = input("Enter IBKR symbols (space-separated, e.g. ASML IFX): ").strip().upper().split()
            to_close = [p for p in all_positions if p.contract.symbol.upper() in set(raw)]
            label    = f"specified: {[p.contract.symbol for p in to_close]}"
        else:
            print("Cancelled — no action taken.")
            feed.disconnect()
            return

    if not to_close:
        print("Nothing to close matching the filter.")
        feed.disconnect()
        return

    # ── Confirm and execute ────────────────────────────────────────────────────
    mode_str = "PAPER" if paper else "*** LIVE (real money) ***"
    print(f"Will close {label} on {mode_str} account:")
    total_unreal = 0.0
    for p in to_close:
        action_str = "BUY (cover short)" if p.position < 0 else "SELL (close long)"
        print(f"  {action_str:<22} {abs(p.position):.0f} {p.contract.symbol}  "
              f"unreal P&L: {_pnl_str(p.unrealizedPNL)} EUR")
        total_unreal += p.unrealizedPNL
    print(f"  Total unrealized P&L: {_pnl_str(total_unreal)} EUR")

    if not _confirm(f"\nProceed?"):
        print("Aborted — no orders placed.")
        feed.disconnect()
        return

    for p in to_close:
        qty    = int(abs(p.position))
        action = "BUY" if p.position < 0 else "SELL"
        order  = MarketOrder(action, qty)
        order.account    = account
        order.tif        = "DAY"
        order.outsideRth = False
        ib.placeOrder(p.contract, order)
        print(f"  Submitted: {action} {qty} {p.contract.symbol}")

    ib.sleep(3)
    print("\nOrders submitted. Check IB Gateway for fills.")
    print("If markets are closed, DAY orders expire — re-run at next open.")

    feed.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intraday position management — inspect, close, or emergency-flatten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", nargs="+", metavar="TICKER",
        help="Close specific ticker(s), e.g. ASML.AS IFX.DE",
    )
    parser.add_argument(
        "--eod-close", "--emergency", action="store_true",
        help="Cancel all open orders and flatten ALL positions (EOD / emergency)",
    )
    parser.add_argument(
        "--shorts-only", action="store_true",
        help="Cover accidental short positions only",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live account (default: paper)",
    )
    args = parser.parse_args()

    paper = not args.live

    if not paper:
        print("!" * 60)
        print("  LIVE ACCOUNT — REAL MONEY")
        print("  This will place MARKET orders on your live IBKR account.")
        print("!" * 60)
        if not _confirm("Confirm live trading"):
            print("Aborted.")
            return

    run(
        tickers=args.ticker,
        shorts_only=args.shorts_only,
        eod_close=args.eod_close,
        paper=paper,
    )


if __name__ == "__main__":
    main()
