"""
Position management utility — inspect and close positions selectively.

The agent NEVER intentionally creates short positions (allow_short=false in config).
Any short positions you see are ACCIDENTAL — caused by EOD close selling more than was
held (e.g. when multiple brackets for the same ticker existed and some had already been
closed by TP/SL before EOD ran). These should be covered as soon as the market opens.

Usage:
    python close_shorts.py                          # show all positions, interactive
    python close_shorts.py --ticker BMW.DE DB1.DE   # close specific tickers
    python close_shorts.py --shorts-only            # cover all accidental short positions
    python close_shorts.py --emergency              # close EVERYTHING (requires confirm)
    python close_shorts.py --live ...               # live account (requires extra confirm)

Always paper trading unless --live is explicitly passed.
Uses DAY orders. If markets are closed, orders expire at session end — re-run at open.
"""

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


def _pnl_str(unreal: float) -> str:
    return f"+{unreal:.2f}" if unreal >= 0 else f"{unreal:.2f}"


def run(tickers: list[str] | None, shorts_only: bool, emergency: bool, paper: bool):
    from agent.data_feed import IBKRFeed
    from ib_insync import MarketOrder

    feed = IBKRFeed(cfg, paper=paper)
    feed.connect()
    ib = feed.ib
    account = feed.account_id

    ib.sleep(2)  # wait for full position data

    all_positions = [p for p in ib.positions() if abs(p.position) > 0]
    if not all_positions:
        print("No open positions.")
        feed.disconnect()
        return

    has_shorts = any(p.position < 0 for p in all_positions)
    print(f"\n{'Ticker':10}  {'Qty':>8}  {'Type':10}  {'AvgCost':>9}  {'UnrealPNL':>10}")
    print("-" * 58)
    for p in all_positions:
        if p.position < 0:
            kind = "SHORT(*)"   # (*) = accidental, agent never intentionally shorts
        else:
            kind = "LONG"
        print(f"  {p.contract.symbol:10}  {p.position:>8.0f}  {kind:10}  "
              f"{p.averageCost:>9.4f}  {_pnl_str(p.unrealizedPNL):>10}")
    if has_shorts:
        print("\n  (*) Accidental short — EOD sold more than was held.")
        print("      Cover with: python close_shorts.py --shorts-only")
    print()

    # Determine which positions to close
    if emergency:
        to_close = all_positions
        label = "ALL positions (emergency)"
    elif shorts_only:
        to_close = [p for p in all_positions if p.position < 0]
        label = "all SHORT positions"
    elif tickers:
        # Match by IBKR symbol — tickers arg may be Yahoo format (BMW.DE) or IBKR (BMW)
        ticker_symbols = {t.split(".")[0].upper() for t in tickers}
        to_close = [p for p in all_positions
                    if p.contract.symbol.upper() in ticker_symbols]
        not_found = ticker_symbols - {p.contract.symbol.upper() for p in to_close}
        if not_found:
            print(f"Warning: {not_found} not found in current positions.")
        label = f"specified tickers: {[p.contract.symbol for p in to_close]}"
    else:
        # Interactive mode — show and ask
        print("No filter specified. Options:")
        print("  1. Close all SHORT positions (accidental shorts only)")
        print("  2. Specify by number from the list above")
        print("  3. Cancel — do nothing")
        choice = input("Choice [1/2/3]: ").strip()
        if choice == "1":
            to_close = [p for p in all_positions if p.position < 0]
            label = "all SHORT positions"
        elif choice == "2":
            syms = input("Enter IBKR symbols space-separated: ").strip().upper().split()
            to_close = [p for p in all_positions if p.contract.symbol.upper() in syms]
            label = f"specified: {[p.contract.symbol for p in to_close]}"
        else:
            print("Cancelled.")
            feed.disconnect()
            return

    if not to_close:
        print("Nothing to close.")
        feed.disconnect()
        return

    print(f"\nWill close {label}:")
    for p in to_close:
        action = "BUY (cover)" if p.position < 0 else "SELL"
        print(f"  {action:12}  {abs(p.position):.0f}  {p.contract.symbol}  "
              f"unrealized: {_pnl_str(p.unrealizedPNL)} EUR")

    mode_str = "PAPER" if paper else "*** LIVE ***"
    if not _confirm(f"\nProceed on {mode_str} account?"):
        print("Aborted.")
        feed.disconnect()
        return

    for p in to_close:
        qty = int(abs(p.position))
        action = "BUY" if p.position < 0 else "SELL"
        order = MarketOrder(action, qty)
        order.account = account
        order.tif = "DAY"
        ib.placeOrder(p.contract, order)
        print(f"  Submitted: {action} {qty} {p.contract.symbol}")

    ib.sleep(3)
    print("\nOrders submitted. Check IB Gateway for fills.")
    print("Note: if markets are closed, DAY orders expire — re-run at next open.")

    feed.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Close specific positions")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="Close these specific tickers (e.g. BMW.DE DB1.DE)")
    parser.add_argument("--shorts-only", action="store_true",
                        help="Cover all accidental short positions only")
    parser.add_argument("--emergency", action="store_true",
                        help="Close ALL open positions immediately")
    parser.add_argument("--live", action="store_true",
                        help="Use live account (default: paper)")
    args = parser.parse_args()

    paper = not args.live

    if not paper:
        print("⚠  LIVE ACCOUNT MODE — real money")
        if not _confirm("Confirm live trading"):
            print("Aborted.")
            return

    run(
        tickers=args.ticker,
        shorts_only=args.shorts_only,
        emergency=args.emergency,
        paper=paper,
    )


if __name__ == "__main__":
    main()
