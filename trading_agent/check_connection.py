"""
Pre-flight connection check for the trading agent.
Run this before starting the agent to verify everything is wired correctly.

Usage:
    python check_connection.py          # paper (default)
    python check_connection.py --live   # live account
"""

from __future__ import annotations

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CET = ZoneInfo("Europe/Berlin")
OK  = "[OK]  "
ERR = "[FAIL]"
WARN= "[WARN]"


def check(label: str, passed: bool, detail: str = "", warn: bool = False) -> bool:
    icon = OK if passed else (WARN if warn else ERR)
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return passed


def run_checks(paper: bool) -> bool:
    print()
    print("=" * 60)
    print(f"  IBKR Trading Agent  —  Pre-flight Check")
    print(f"  {datetime.now(CET).strftime('%Y-%m-%d %H:%M CET')}  |  Mode: {'PAPER' if paper else 'LIVE'}")
    print("=" * 60)

    all_ok = True

    # ── 1. Config ─────────────────────────────────────────────
    print("\n  1. Configuration")
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        check("config.yaml loads", True)
    except Exception as e:
        check("config.yaml loads", False, str(e))
        return False

    with open("configs/ibkr_contracts.yaml") as f:
        contracts = yaml.safe_load(f)["contracts"]
    check(f"IBKR contracts ({len(contracts)} tickers)", True)

    # ── 2. .env / account ─────────────────────────────────────
    print("\n  2. Credentials")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    account = os.getenv("IBKR_ACCOUNT", "")
    check(".env loaded", True)
    check(f"IBKR_ACCOUNT set", bool(account), "(set) — will auto-detect from IB Gateway if blank")
    port_var = "IBKR_PAPER_PORT" if paper else "IBKR_LIVE_PORT"
    port_key = "paper_port" if paper else "live_port"
    port = int(os.getenv(port_var, cfg["ibkr"][port_key]))
    check(f"Port configured ({port_var}={port})", True)

    # ── 3. Model ──────────────────────────────────────────────
    print("\n  3. ML Model")
    model_path = Path(cfg["model"]["path"])
    feat_path  = Path(cfg["model"]["feature_cols_path"])
    model_ok = check("Model file exists", model_path.exists(), str(model_path))
    feat_ok  = check("Feature cols file exists", feat_path.exists(), str(feat_path))
    all_ok = all_ok and model_ok and feat_ok

    if model_ok and feat_ok:
        try:
            import joblib
            model = joblib.load(model_path)
            with open(feat_path) as f:
                feat_cols = json.load(f)
            check(f"Model loads ({len(feat_cols)} features)", True,
                  f"n_estimators={model.n_estimators}")
        except Exception as e:
            check("Model loads", False, str(e))
            all_ok = False

    learner_path = model_path.parent / "learner_state.json"
    if learner_path.exists():
        with open(learner_path) as f:
            ls = json.load(f)
        check(f"Learner state (fade={ls.get('fade_threshold'):.3f})", True,
              f"updated {ls.get('updated','?')[:10]}")
    else:
        check("Learner state", True, "not yet created — will initialise on first run", warn=True)

    # ── 4. IBKR connection ────────────────────────────────────
    print(f"\n  4. IBKR Gateway Connection (port {port})")
    try:
        from agent.data_feed import IBKRFeed
        feed = IBKRFeed(cfg, paper=paper)
        feed.connect()
        check("IB Gateway connected", True, f"account={feed.account_id}")

        nav = feed.get_nav()
        check(f"Account NAV readable", nav > 0, f"{nav:,.2f} {cfg['capital']['currency']}")

        # Test one contract: SAP.DE (liquid, well-mapped)
        from ib_insync import Stock
        test_ticker = "SAP.DE"
        spec = contracts.get(test_ticker, {})
        if spec:
            contract = Stock(spec["symbol"], spec["exchange"], spec["currency"])
            qualified = feed.qualify_contract(contract)
            check(f"Contract qualification ({test_ticker})", qualified,
                  f"{spec['symbol']} on {spec['exchange']}")

            if qualified:
                price = feed.get_latest_price(contract)
                price_ok = price > 0
                # Check if OHLCV cache exists as fallback
                cache_dir = Path(cfg["data"]["cache_dir"])
                cache_file = cache_dir / f"{test_ticker.replace('/', '_')}.parquet"
                has_cache = cache_file.exists()
                if price_ok:
                    check(f"Market data / price quote", True,
                          f"{price:.2f} {spec['currency']}")
                elif has_cache:
                    check(f"Market data / price quote", True,
                          f"no live quote — will use OHLCV cache fallback (ok for paper)",
                          warn=True)
                else:
                    check(f"Market data / price quote", False,
                          "no quote and no cache — run bootstrap_model.py --yfinance first")
                    all_ok = False

        feed.disconnect()

    except ConnectionError as e:
        check("IB Gateway connected", False, str(e))
        print()
        print("  IB Gateway troubleshooting:")
        print("    1. Open IB Gateway and log in (paper trading mode)")
        print("    2. Go to Configure -> Settings -> API")
        print("    3. Enable 'Enable ActiveX and Socket Clients'")
        print(f"    4. Set Socket port to {port}")
        print("    5. Tick 'Allow connections from localhost only'")
        print("    6. Click OK, restart IB Gateway")
        all_ok = False

    except Exception as e:
        check("IBKR checks", False, str(e))
        all_ok = False

    # ── 5. Data / cache ───────────────────────────────────────
    print("\n  5. Data & Cache")
    cache_dir = Path(cfg["data"]["cache_dir"])
    cached = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    check(f"Cache directory", cache_dir.exists(), str(cache_dir))
    check(f"Cached tickers", len(cached) > 0, f"{len(cached)} parquet files")

    from pathlib import Path as P
    db_path = P(cfg["journal"]["db_path"])
    check("Journal DB", db_path.exists(),
          str(db_path) if db_path.exists() else "will be created on first run")

    # ── 6. Market hours ───────────────────────────────────────
    print("\n  6. Market Hours")
    now = datetime.now(CET)
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    is_weekday = now.weekday() < 5
    from run_agent import is_market_open
    market_open = is_market_open(now)
    check(f"Today is a trading day ({weekday_names[now.weekday()]})", is_weekday, warn=not is_weekday)
    check(f"Market open right now ({now.strftime('%H:%M CET')})", market_open,
          "orders will queue for next open if market is closed", warn=not market_open)

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_ok:
        print("  RESULT: ALL CHECKS PASSED — ready to trade")
        print()
        print("  Start paper trading:")
        print("    start.bat           (Windows)")
        print("    ./start.sh          (Linux/Mac)")
        print("    python run_agent.py --paper --once   (one cycle now)")
    else:
        print("  RESULT: SOME CHECKS FAILED — fix issues above before trading")
    print("=" * 60)
    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="IBKR pre-flight connection check")
    parser.add_argument("--live", action="store_true", help="Check live account (default: paper)")
    args = parser.parse_args()
    success = run_checks(paper=not args.live)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
