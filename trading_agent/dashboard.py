"""
Trading Agent Dashboard
=======================
Live P&L monitor, trade history, learner state, and daily breakdown.
Works offline (reads SQLite journal only — no IBKR connection needed).

Usage:
    python dashboard.py            # single snapshot
    python dashboard.py --watch    # refresh every 30 seconds
    python dashboard.py --trades   # detailed trade list
    python dashboard.py --signals  # today's signals
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import os

import yaml
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()   # read IBKR_ACCOUNT from .env if present

_W = 72   # terminal width

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def _bar(value: float, max_val: float, width: int = 20, fill: str = "#", empty: str = ".") -> str:
    if max_val == 0:
        return empty * width
    filled = int(min(abs(value) / max_val, 1.0) * width)
    return fill * filled + empty * (width - filled)


def _pnl_str(v: float) -> str:
    if v is None:
        return "     --"
    sign = "+" if v >= 0 else ""
    return f"{sign}{abs(v):.2f} EUR"


def _sep(char: str = "-", width: int = _W) -> str:
    return char * width


def _header(title: str) -> str:
    pad = (_W - len(title) - 2) // 2
    return _sep("=") + "\n" + " " * pad + title + "\n" + _sep("=")


# ── sections ──────────────────────────────────────────────────────────────────

def section_summary(conn: sqlite3.Connection, cfg: dict) -> str:
    row = conn.execute("""
        SELECT
            COUNT(*)                                         AS n,
            SUM(pnl_net)                                     AS total,
            AVG(pnl_net)                                     AS avg,
            SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END)    AS wins,
            MIN(pnl_net)                                     AS worst,
            MAX(pnl_net)                                     AS best,
            SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END) AS gross_win,
            SUM(CASE WHEN pnl_net < 0 THEN ABS(pnl_net) ELSE 0 END) AS gross_loss
        FROM trades WHERE pnl_net IS NOT NULL
    """).fetchone()

    open_row = conn.execute(
        "SELECT COUNT(*) AS n FROM trades WHERE exit_price IS NULL AND status != 'error'"
    ).fetchone()

    n = row["n"] or 0
    total = row["total"] or 0.0
    avg   = row["avg"] or 0.0
    wins  = row["wins"] or 0
    worst = row["worst"] or 0.0
    best  = row["best"] or 0.0
    gw    = row["gross_win"] or 0.0
    gl    = row["gross_loss"] or 0.0
    open_n = open_row["n"] or 0

    win_rate = wins / n if n else 0
    profit_factor = gw / gl if gl > 0 else float("inf")

    fade_thr  = cfg["strategy"]["fade_threshold"]
    foll_thr  = cfg["strategy"]["follow_threshold"]
    foll_dis  = cfg["strategy"].get("follow_disabled", True)

    learner_path = Path(cfg["model"]["path"]).parent / "learner_state.json"
    if learner_path.exists():
        with open(learner_path) as f:
            ls = json.load(f)
        fade_thr = ls.get("fade_threshold", fade_thr)
        foll_thr = ls.get("follow_threshold", foll_thr)

    lines = [
        _sep("="),
        f"  TRADING AGENT DASHBOARD  —  account: {os.getenv('IBKR_ACCOUNT', '(set IBKR_ACCOUNT in .env)')}",
        f"  {datetime.now().strftime('%d/%m/%Y  %H:%M:%S')}",
        _sep("="),
        "",
        "  PERFORMANCE SUMMARY",
        _sep("-"),
        f"  Completed trades   : {n:>6}          Open positions : {open_n}",
        f"  Total P&L (net)    : {_pnl_str(total):>14}",
        f"  Average per trade  : {_pnl_str(avg):>14}",
        f"  Hit rate           : {win_rate:>13.1%}  ({wins}W / {n-wins}L)",
        f"  Best trade         : {_pnl_str(best):>14}",
        f"  Worst trade        : {_pnl_str(worst):>14}",
        f"  Profit factor      : {'inf' if profit_factor == float('inf') else f'{profit_factor:.2f}':>13}",
        "",
        "  LEARNER STATE",
        _sep("-"),
        f"  Fade threshold     : {fade_thr:.3f}   (signal when P(failure) >= this)",
        f"  Follow threshold   : {foll_thr:.3f}   (signal when P(failure) <= this)",
        f"  Follow mode        : {'DISABLED' if foll_dis else 'ENABLED'}",
        f"  Config: max {cfg['risk']['max_trades_per_day']} trades/day | "
        f"pos {cfg['risk']['max_position_pct']*100:.0f}% NAV | "
        f"SL {cfg['risk']['stop_loss_pct']*100:.0f}% | "
        f"TP {cfg['risk']['take_profit_pct']*100:.0f}%",
    ]
    return "\n".join(lines)


def section_alert_breakdown(conn: sqlite3.Connection) -> str:
    rows = conn.execute("""
        SELECT
            s.alert_name,
            s.action,
            COUNT(*)                                            AS n,
            SUM(CASE WHEN t.pnl_net > 0 THEN 1 ELSE 0 END)   AS wins,
            SUM(t.pnl_net)                                      AS total_pnl
        FROM trades t
        JOIN signals s ON t.signal_id = s.id
        WHERE t.pnl_net IS NOT NULL
        GROUP BY s.alert_name, s.action
        ORDER BY total_pnl DESC
    """).fetchall()

    lines = ["", "  PERFORMANCE BY ALERT TYPE", _sep("-")]
    if not rows:
        lines.append("  (no completed trades yet)")
    else:
        lines.append(f"  {'Alert':<32} {'Act':4} {'n':>4} {'Win%':>6} {'P&L':>10}")
        lines.append("  " + "-" * 58)
        for r in rows:
            n = r["n"] or 0
            wr = (r["wins"] or 0) / n if n else 0
            pnl = r["total_pnl"] or 0.0
            bar = _bar(pnl, max(abs(pnl), 1), width=12,
                       fill="+" if pnl >= 0 else "-")
            lines.append(
                f"  {(r['alert_name'] or 'unknown'):<32} "
                f"{(r['action'] or 'FADE'):4} {n:>4} {wr:>6.1%} "
                f"{pnl:>+9.2f}  {bar}"
            )
    return "\n".join(lines)


def section_recent_trades(conn: sqlite3.Connection, n: int = 20) -> str:
    rows = conn.execute("""
        SELECT t.date, t.ticker, s.action, t.trade_direction,
               t.entry_price, t.exit_price, t.stop_loss, t.take_profit,
               t.pnl_net, t.status
        FROM trades t
        LEFT JOIN signals s ON t.signal_id = s.id
        ORDER BY t.ts DESC LIMIT ?
    """, (n,)).fetchall()

    lines = ["", f"  RECENT TRADES  (last {n})", _sep("-")]
    if not rows:
        lines.append("  (no trades yet)")
    else:
        lines.append(
            f"  {'Date':10} {'Ticker':10} {'Act':4} {'Dir':4} "
            f"{'Entry':>8} {'Exit':>8} {'P&L':>10}  Status"
        )
        lines.append("  " + "-" * 62)
        for r in rows:
            exit_str = f"{r['exit_price']:8.3f}" if r["exit_price"] else "    open"
            pnl_str  = _pnl_str(r["pnl_net"]) if r["pnl_net"] is not None else "      open"
            status   = r["status"] or ""
            lines.append(
                f"  {r['date']:10} {(r['ticker'] or ''):10} "
                f"{(r['action'] or 'FADE'):4} {(r['trade_direction'] or ''):4} "
                f"{r['entry_price']:8.3f} {exit_str} {pnl_str:>10}  {status}"
            )
    return "\n".join(lines)


def section_daily_summary(conn: sqlite3.Connection, days: int = 14) -> str:
    since = str(date.today() - timedelta(days=days))
    rows = conn.execute("""
        SELECT date, nav, daily_pnl, n_signals, n_trades, paper
        FROM daily_summary
        WHERE date >= ?
        ORDER BY date DESC
    """, (since,)).fetchall()

    lines = ["", f"  DAILY SUMMARY  (last {days} days)", _sep("-")]
    if not rows:
        lines.append("  (no daily data yet)")
    else:
        lines.append(
            f"  {'Date':10} {'NAV':>12} {'Daily P&L':>12} "
            f"{'Signals':>8} {'Trades':>7}  Mode"
        )
        lines.append("  " + "-" * 56)
        for r in rows:
            mode = "PAPER" if r["paper"] else "LIVE"
            pnl_str = _pnl_str(r["daily_pnl"])
            lines.append(
                f"  {r['date']:10} {r['nav']:>12.2f} {pnl_str:>12} "
                f"{r['n_signals']:>8} {r['n_trades']:>7}  {mode}"
            )
    return "\n".join(lines)


def section_open_positions(conn: sqlite3.Connection) -> str:
    rows = conn.execute("""
        SELECT t.date, t.ticker, s.action, t.trade_direction,
               t.quantity, t.entry_price, t.stop_loss, t.take_profit,
               t.ibkr_order_id
        FROM trades t
        LEFT JOIN signals s ON t.signal_id = s.id
        WHERE t.exit_price IS NULL AND t.status != 'error'
        ORDER BY t.ts DESC
    """).fetchall()

    lines = ["", "  OPEN POSITIONS", _sep("-")]
    if not rows:
        lines.append("  (none)")
    else:
        lines.append(
            f"  {'Date':10} {'Ticker':10} {'Act':4} {'Dir':4} "
            f"{'Qty':>8} {'Entry':>8} {'SL':>8} {'TP':>8}  OrderID"
        )
        lines.append("  " + "-" * 68)
        for r in rows:
            lines.append(
                f"  {r['date']:10} {(r['ticker'] or ''):10} "
                f"{(r['action'] or 'FADE'):4} {(r['trade_direction'] or ''):4} "
                f"{r['quantity']:>8.4f} {r['entry_price']:>8.3f} "
                f"{r['stop_loss']:>8.3f} {r['take_profit']:>8.3f}  {r['ibkr_order_id']}"
            )
    return "\n".join(lines)


def section_todays_signals(conn: sqlite3.Connection) -> str:
    today = str(date.today())
    rows = conn.execute("""
        SELECT ticker, alert_name, alert_direction, failure_proba,
               action, trade_direction, conviction,
               COALESCE(crowding_score, 0.0) AS crowding_score,
               explanation
        FROM signals
        WHERE date = ?
        ORDER BY conviction DESC
    """, (today,)).fetchall()

    lines = ["", f"  TODAY'S SIGNALS  ({today})", _sep("-")]
    if not rows:
        lines.append("  (none yet — agent runs at 09:15 CET)")
    else:
        lines.append(
            f"  {'Ticker':10} {'Alert':28} {'Dir':8} "
            f"{'P(fail)':>8} {'Crowd':>6} {'Action':7} {'Conv':>6}"
        )
        lines.append("  " + "-" * 80)
        for r in rows:
            lines.append(
                f"  {(r['ticker'] or ''):10} {(r['alert_name'] or ''):28} "
                f"{(r['alert_direction'] or ''):8} {r['failure_proba']:>8.3f} "
                f"{r['crowding_score']:>6.2f} "
                f"{(r['action'] or 'SKIP'):7} {(r['conviction'] or 0.0):>6.3f}"
            )
            # Show explanation on a second line for traded signals
            expl = (r["explanation"] or "").strip()
            if expl and r["action"] not in ("SKIP", None):
                # Word-wrap to terminal width
                max_len = _W - 6
                wrapped = expl if len(expl) <= max_len else expl[:max_len - 3] + "..."
                lines.append(f"    ↳ {wrapped}")
    return "\n".join(lines)


def section_trade_rationale(conn: sqlite3.Connection, n: int = 10) -> str:
    """Detailed why-we-traded explanation for the most recent completed trades."""
    rows = conn.execute("""
        SELECT t.date, t.ticker, s.action, t.trade_direction,
               t.entry_price, t.exit_price, t.pnl_net,
               COALESCE(s.crowding_score, 0.0) AS crowding_score,
               s.failure_proba, s.explanation
        FROM trades t
        LEFT JOIN signals s ON t.signal_id = s.id
        WHERE t.exit_price IS NOT NULL
        ORDER BY t.ts DESC LIMIT ?
    """, (n,)).fetchall()

    lines = ["", f"  TRADE RATIONALE  (last {n} closed)", _sep("-")]
    if not rows:
        lines.append("  (no closed trades yet)")
    else:
        for r in rows:
            outcome = ""
            if r["pnl_net"] is not None:
                outcome = f"  P&L: {_pnl_str(r['pnl_net'])}  {'WIN' if r['pnl_net'] > 0 else 'LOSS'}"
            lines.append(
                f"  {r['date']}  {(r['ticker'] or ''):10}  "
                f"{(r['action'] or 'FADE'):5} {(r['trade_direction'] or ''):4}  "
                f"entry={r['entry_price']:.3f} → exit={r['exit_price']:.3f}{outcome}"
            )
            expl = (r["explanation"] or "").strip()
            if expl:
                max_len = _W - 6
                wrapped = expl if len(expl) <= max_len else expl[:max_len - 3] + "..."
                lines.append(f"    ↳ {wrapped}")
            else:
                lines.append(
                    f"    ↳ P(fail)={r['failure_proba']:.2f}  "
                    f"crowd={r['crowding_score']:.2f}  (no explanation stored)"
                )
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def render(cfg: dict, show_trades: bool = False, show_signals: bool = False) -> str:
    db_path = cfg["journal"]["db_path"]
    if not Path(db_path).exists():
        return "Journal not found. Run the agent at least once."

    conn = _conn(db_path)
    parts = [
        section_summary(conn, cfg),
        section_open_positions(conn),
        section_daily_summary(conn),
        section_alert_breakdown(conn),
    ]
    if show_trades:
        parts.append(section_recent_trades(conn, n=30))
        parts.append(section_trade_rationale(conn, n=10))
    if show_signals:
        parts.append(section_todays_signals(conn))
    parts.append(_sep("="))
    conn.close()
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Trading agent dashboard")
    parser.add_argument("--watch",   action="store_true", help="Refresh every 30 seconds")
    parser.add_argument("--trades",  action="store_true", help="Show detailed trade list")
    parser.add_argument("--signals", action="store_true", help="Show today's signals")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (--watch)")
    args = parser.parse_args()

    cfg = _load_cfg()

    if args.watch:
        try:
            while True:
                # Clear screen
                print("\033[H\033[J", end="")
                print(render(cfg, show_trades=args.trades, show_signals=True))
                print(f"\n  Refreshing every {args.interval}s — Ctrl+C to exit")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
    else:
        print(render(cfg, show_trades=args.trades, show_signals=args.signals or True))


if __name__ == "__main__":
    main()
