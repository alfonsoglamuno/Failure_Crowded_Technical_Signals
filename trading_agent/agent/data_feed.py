"""
IBKR data feed — connects to IB Gateway / TWS and fetches OHLCV data.

Paper trading: port 4002
Live trading:  port 4001
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import os

import pandas as pd
import yaml
from dotenv import load_dotenv
from ib_insync import IB, Stock, util

load_dotenv()   # loads .env if present — never required, just convenient
log = logging.getLogger(__name__)


def load_contracts(contracts_file: str) -> dict:
    with open(contracts_file) as f:
        return yaml.safe_load(f)["contracts"]


class IBKRFeed:
    def __init__(self, cfg: dict, paper: bool = True):
        self.cfg = cfg
        self.paper = paper
        self.ib = IB()
        self._connected = False

    def connect(self):
        # Ports and account can be overridden via environment variables
        paper_port = int(os.getenv("IBKR_PAPER_PORT", self.cfg["ibkr"]["paper_port"]))
        live_port  = int(os.getenv("IBKR_LIVE_PORT",  self.cfg["ibkr"]["live_port"]))
        port = paper_port if self.paper else live_port
        host = self.cfg["ibkr"]["host"]
        client_id = self.cfg["ibkr"]["client_id"]
        timeout = self.cfg["ibkr"].get("timeout", 30)

        log.info("Connecting to IB Gateway %s:%d (paper=%s)", host, port, self.paper)
        try:
            self.ib.connect(host, port, clientId=client_id, timeout=timeout)
            self._connected = True
            log.info("Connected. Account: %s", self.account_id)
            # Use frozen/delayed data — works on paper accounts without live subscriptions.
            # Type 1=live, 2=frozen(last price), 3=delayed(15min), 4=delayed+frozen
            # Type 4 is safest: gives last known price even outside market hours.
            market_data_type = 2 if self.paper else 1
            self.ib.reqMarketDataType(market_data_type)
            log.info("Market data type set to %d (%s)",
                     market_data_type, "frozen/paper" if self.paper else "live")
        except Exception as e:
            raise ConnectionError(f"Could not connect to IB Gateway at {host}:{port} — {e}") from e

    def disconnect(self):
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            log.info("Disconnected from IB Gateway")

    @property
    def account_id(self) -> str:
        # Priority: env var → config → auto-detect from IBKR
        env_account = os.getenv("IBKR_ACCOUNT", "").strip()
        cfg_account = self.cfg["ibkr"].get("account", "").strip()
        if env_account:
            return env_account
        if cfg_account:
            return cfg_account
        accounts = self.ib.managedAccounts()
        return accounts[0] if accounts else ""

    def get_account_summary(self) -> dict:
        summary = self.ib.accountSummary(self.account_id)
        result = {}
        for item in summary:
            if item.currency == self.cfg["capital"]["currency"] or item.currency == "BASE":
                result[item.tag] = item.value
        return result

    def get_nav(self) -> float:
        """Net asset value in account currency."""
        summary = self.get_account_summary()
        nav = summary.get("NetLiquidation") or summary.get("TotalCashValue", "0")
        return float(nav)

    def get_daily_pnl(self) -> float:
        """Today's realised + unrealised P&L."""
        summary = self.get_account_summary()
        return float(summary.get("RealizedPnL", 0)) + float(summary.get("UnrealizedPnL", 0))

    def make_contract(self, yahoo_ticker: str, contracts_cfg: dict) -> Optional[Stock]:
        """Build an IBKR Stock contract from a Yahoo Finance ticker."""
        spec = contracts_cfg.get(yahoo_ticker)
        if not spec:
            log.warning("No IBKR contract mapping for %s", yahoo_ticker)
            return None
        return Stock(spec["symbol"], spec["exchange"], spec["currency"])

    def fetch_ohlcv(
        self,
        contract: Stock,
        days: int = 252,
        bar_size: str = "1 day",
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV bars.
        Returns DataFrame with columns: date, open, high, low, close, volume.
        """
        if not self._connected:
            raise RuntimeError("Not connected to IB Gateway")

        duration = f"{days} D"
        end = ""   # empty = now
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="ADJUSTED_LAST",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            log.warning("No data returned for %s", contract.symbol)
            return pd.DataFrame()

        df = util.df(bars)
        df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                 "low": "low", "close": "close", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df[df["close"] > 0].reset_index(drop=True)
        return df

    def fetch_universe(
        self,
        tickers: list[str],
        contracts_cfg: dict,
        days: int = 252,
        cache_dir: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for all tickers in universe.
        Caches results to parquet to avoid re-fetching.
        """
        cache = Path(cache_dir) if cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

        results = {}
        for yahoo_ticker in tickers:
            if cache:
                cache_file = cache / f"{yahoo_ticker.replace('/', '_')}.parquet"
                if cache_file.exists():
                    age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                    if age_days < 1:
                        results[yahoo_ticker] = pd.read_parquet(cache_file)
                        log.debug("Cache hit: %s", yahoo_ticker)
                        continue

            contract = self.make_contract(yahoo_ticker, contracts_cfg)
            if contract is None:
                continue

            try:
                df = self.fetch_ohlcv(contract, days=days)
                if df.empty:
                    continue
                df["ticker"] = yahoo_ticker
                if cache:
                    df.to_parquet(cache_file, index=False)
                results[yahoo_ticker] = df
                log.info("Fetched %s: %d rows", yahoo_ticker, len(df))
            except Exception as e:
                log.error("Failed to fetch %s: %s", yahoo_ticker, e)

        return results

    def qualify_contract(self, contract: Stock) -> bool:
        """Ask IBKR to fill in missing contract details (conId, etc.). Returns True if valid."""
        try:
            qualified = self.ib.qualifyContracts(contract)
            return len(qualified) > 0
        except Exception as e:
            log.warning("Contract qualification failed for %s: %s", contract.symbol, e)
            return False

    def get_latest_price(self, contract: Stock, fallback_price: float = 0.0) -> float:
        """
        Get current market price for a contract.

        Strategy (in order):
          1. Live bid/ask midpoint via reqMktData (works with subscriptions)
          2. ticker.marketPrice() which uses last trade if no bid/ask
          3. fallback_price (last known close from OHLCV cache)

        Paper accounts without subscriptions: IB Gateway returns frozen/delayed
        data after reqMarketDataType(2) is set on connect.
        """
        try:
            # snapshot=True: one-time request, no need to cancel manually
            ticker = self.ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)

            # Poll for up to 4 seconds
            for _ in range(8):
                self.ib.sleep(0.5)
                bid, ask, last = ticker.bid, ticker.ask, ticker.last

                # Valid bid/ask midpoint
                if (bid and ask
                        and not math.isnan(bid) and not math.isnan(ask)
                        and bid > 0 and ask > 0):
                    mid = (bid + ask) / 2
                    self.ib.cancelMktData(contract)
                    log.debug("Price %s: bid/ask mid = %.4f", contract.symbol, mid)
                    return float(mid)

                # Last traded price
                if last and not math.isnan(last) and last > 0:
                    self.ib.cancelMktData(contract)
                    log.debug("Price %s: last = %.4f", contract.symbol, last)
                    return float(last)

            self.ib.cancelMktData(contract)

        except Exception as e:
            log.warning("reqMktData failed for %s: %s", contract.symbol, e)

        # Fallback: use last known close from cached OHLCV
        if fallback_price > 0:
            log.info("Price %s: using OHLCV fallback close = %.4f", contract.symbol, fallback_price)
            return fallback_price

        log.warning("No price available for %s — skipping trade", contract.symbol)
        return 0.0
