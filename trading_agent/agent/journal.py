"""
Trade journal — SQLite-backed log of every signal, trade, and outcome.
"""

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, date
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    date            TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    alert_name      TEXT,
    alert_direction TEXT,
    failure_proba   REAL,
    action          TEXT,    -- FADE / FOLLOW / SKIP
    trade_direction TEXT,    -- BUY / SELL / None
    conviction      REAL,
    crowding_score  REAL,
    explanation     TEXT     -- human-readable rationale (SHAP-based)
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES signals(id),
    ts              TEXT NOT NULL,
    date            TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    ibkr_symbol     TEXT,
    trade_direction TEXT,
    quantity        REAL,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    ibkr_order_id   INTEGER,
    status          TEXT,    -- submitted / filled / cancelled / error
    fill_price      REAL,
    exit_price      REAL,
    exit_date       TEXT,
    pnl_gross       REAL,
    pnl_net         REAL,
    paper           INTEGER  -- 1 = paper, 0 = live
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date            TEXT PRIMARY KEY,
    nav             REAL,
    daily_pnl       REAL,
    n_signals       INTEGER,
    n_trades        INTEGER,
    paper           INTEGER
);
"""


class Journal:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # Safe migration — add columns that may be missing in older databases
            for col, typedef in [
                ("crowding_score", "REAL"),
                ("explanation",    "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {typedef}")
                except Exception:
                    pass  # column already exists
            for col, typedef in [
                ("slippage_pct", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")
                except Exception:
                    pass

    def log_signal(
        self,
        ticker: str,
        alert_name: str,
        alert_direction: str,
        failure_proba: float,
        action: str,
        trade_direction: str | None,
        conviction: float,
        trade_date: date | None = None,
        crowding_score: float = 0.0,
        explanation: str = "",
    ) -> int:
        """Insert or update a signal for today. Upserts on (date, ticker, alert_name)
        so re-running the agent in the same day doesn't create duplicate log entries."""
        ts = datetime.utcnow().isoformat()
        d = str(trade_date or date.today())
        with self._conn() as conn:
            # Check if this signal already exists for today
            existing = conn.execute(
                "SELECT id FROM signals WHERE date=? AND ticker=? AND alert_name=?",
                (d, ticker, alert_name),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE signals SET ts=?, failure_proba=?, action=?,
                       trade_direction=?, conviction=?, crowding_score=?,
                       explanation=?
                       WHERE id=?""",
                    (ts, float(failure_proba), action, trade_direction,
                     float(conviction), float(crowding_score), explanation,
                     existing[0]),
                )
                return existing[0]
            cur = conn.execute(
                """INSERT INTO signals
                   (ts, date, ticker, alert_name, alert_direction,
                    failure_proba, action, trade_direction, conviction,
                    crowding_score, explanation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, d, ticker, alert_name, alert_direction,
                 float(failure_proba), action, trade_direction, float(conviction),
                 float(crowding_score), explanation),
            )
            return cur.lastrowid

    def log_trade(
        self,
        signal_id: int,
        ticker: str,
        ibkr_symbol: str,
        trade_direction: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        ibkr_order_id: int,
        status: str,
        paper: bool,
        trade_date: date | None = None,
    ) -> int:
        ts = datetime.utcnow().isoformat()
        d = str(trade_date or date.today())
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (signal_id, ts, date, ticker, ibkr_symbol, trade_direction,
                    quantity, entry_price, stop_loss, take_profit,
                    ibkr_order_id, status, paper)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (signal_id, ts, d, ticker, ibkr_symbol, trade_direction,
                 quantity, entry_price, stop_loss, take_profit,
                 ibkr_order_id, status, int(paper)),
            )
            return cur.lastrowid

    def update_entry_fill(
        self,
        trade_id: int,
        fill_price: float,
        slippage_pct: float,
    ):
        """
        Record the actual IBKR fill price for the entry order.

        fill_price   : avgPrice from IBKR execution fills
        slippage_pct : (fill_price - entry_price) / entry_price
                       positive = filled worse than quote (BUY slippage)
                       negative = filled better than quote
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE trades SET fill_price=?, slippage_pct=?, status='filled' WHERE id=?",
                (fill_price, slippage_pct, trade_id),
            )

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_date: date,
        pnl_gross: float,
        commission: float,
        estimated: bool = False,
    ):
        """
        Record trade exit.
        Set estimated=True when the exit price is from a live quote (not a real fill),
        so that check_exits can override it with the actual fill price later.
        """
        pnl_net = pnl_gross - commission
        status = "pending_close" if estimated else "filled"
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET exit_price=?, exit_date=?,
                   pnl_gross=?, pnl_net=?, status=?
                   WHERE id=?""",
                (exit_price, str(exit_date), pnl_gross, pnl_net, status, trade_id),
            )

    def log_daily_summary(self, nav: float, daily_pnl: float,
                           n_signals: int, n_trades: int, paper: bool,
                           summary_date: date | None = None):
        d = str(summary_date or date.today())
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_summary
                   (date, nav, daily_pnl, n_signals, n_trades, paper)
                   VALUES (?,?,?,?,?,?)""",
                (d, nav, daily_pnl, n_signals, n_trades, int(paper)),
            )

    def get_recent_trades(self, n: int = 50) -> list[dict]:
        """Returns trades enriched with action and alert_name from signals table."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT t.*, s.action, s.alert_name
                   FROM trades t
                   LEFT JOIN signals s ON t.signal_id = s.id
                   ORDER BY t.ts DESC LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_signals(self, n: int = 50) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_performance_summary(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as n_trades,
                    SUM(pnl_net) as total_pnl,
                    AVG(pnl_net) as avg_pnl,
                    SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as wins,
                    MIN(pnl_net) as worst_trade,
                    MAX(pnl_net) as best_trade
                FROM trades
                WHERE pnl_net IS NOT NULL
            """).fetchone()
        if not row or row[0] == 0:
            return {"n_trades": 0}
        n = row[0]
        return {
            "n_trades": n,
            "total_pnl": round(row[1], 2),
            "avg_pnl": round(row[2], 4),
            "hit_rate": round(row[3] / n, 3),
            "worst_trade": round(row[4], 2),
            "best_trade": round(row[5], 2),
        }
