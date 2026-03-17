"""
Trading report — daily, weekly, monthly performance summaries.

Usage:
    python report.py                  # today
    python report.py --today
    python report.py --week
    python report.py --month
    python report.py --date 2026-03-17
    python report.py --all
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path


DB_PATH = "data/journal.db"
W = 70


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _bar(value: float, max_val: float, width: int = 20, char: str = "#") -> str:
    if max_val == 0:
        return ""
    filled = int(abs(value) / max_val * width)
    return char * min(filled, width)


def _sign(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_trades(db: str, start: date, end: date) -> list[dict]:
    """Completed trades (with exit price) in date range."""
    with _conn(db) as conn:
        rows = conn.execute("""
            SELECT t.*, s.action, s.alert_name, s.alert_direction, s.failure_proba,
                   s.crowding_score, s.conviction
            FROM trades t
            LEFT JOIN signals s ON t.signal_id = s.id
            WHERE t.exit_price IS NOT NULL
              AND t.date >= ? AND t.date <= ?
            ORDER BY t.date, t.ts
        """, (str(start), str(end))).fetchall()
    return [dict(r) for r in rows]


def get_open_trades(db: str) -> list[dict]:
    with _conn(db) as conn:
        rows = conn.execute("""
            SELECT t.*, s.action, s.alert_name
            FROM trades t
            LEFT JOIN signals s ON t.signal_id = s.id
            WHERE t.exit_price IS NULL AND t.status NOT IN ('cancelled','error')
            ORDER BY t.date DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_daily_summaries(db: str, start: date, end: date) -> list[dict]:
    with _conn(db) as conn:
        rows = conn.execute("""
            SELECT * FROM daily_summary
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, (str(start), str(end))).fetchall()
    return [dict(r) for r in rows]


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}
    pnls = [t["pnl_net"] for t in trades if t.get("pnl_net") is not None]
    if not pnls:
        return {}
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    best  = max(trades, key=lambda t: t.get("pnl_net") or -9e9)
    worst = min(trades, key=lambda t: t.get("pnl_net") or 9e9)
    gross_wins   = sum(wins)   if wins   else 0
    gross_losses = abs(sum(losses)) if losses else 0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")
    return {
        "n":             len(pnls),
        "n_wins":        len(wins),
        "n_losses":      len(losses),
        "hit_rate":      len(wins) / len(pnls) if pnls else 0,
        "total_pnl":     sum(pnls),
        "avg_pnl":       sum(pnls) / len(pnls),
        "best_pnl":      best.get("pnl_net", 0),
        "worst_pnl":     worst.get("pnl_net", 0),
        "best_trade":    best,
        "worst_trade":   worst,
        "profit_factor": profit_factor,
        "avg_win":       sum(wins) / len(wins) if wins else 0,
        "avg_loss":      sum(losses) / len(losses) if losses else 0,
    }


def by_ticker(trades: list[dict]) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for t in trades:
        groups[t["ticker"]].append(t)
    return {tk: compute_stats(ts) for tk, ts in groups.items()}


def by_alert(trades: list[dict]) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for t in trades:
        key = t.get("alert_name") or "unknown"
        groups[key].append(t)
    return {k: compute_stats(v) for k, v in groups.items()}


# ── Rendering ──────────────────────────────────────────────────────────────────

def sep(char: str = "-") -> str:
    return char * W


def header(title: str, char: str = "=") -> str:
    return f"\n{char * W}\n  {title}\n{char * W}"


def render_overview(stats: dict, open_trades: list[dict] | None = None) -> str:
    if not stats:
        return "  No completed trades in this period.\n"
    lines = [
        f"  Completed trades   : {stats['n']}",
        f"  Wins / Losses      : {stats['n_wins']}W / {stats['n_losses']}L",
        f"  Hit rate           : {stats['hit_rate']:.1%}",
        f"  Total P&L (net)    : {_sign(stats['total_pnl'])} EUR",
        f"  Average per trade  : {_sign(stats['avg_pnl'])} EUR",
        f"  Avg win            : {_sign(stats['avg_win'])} EUR",
        f"  Avg loss           : {_sign(stats['avg_loss'])} EUR",
        f"  Profit factor      : {stats['profit_factor']:.2f}"
            if stats['profit_factor'] != float('inf')
            else "  Profit factor      : inf (no losses)",
        f"  Best trade         : {stats['best_trade'].get('ticker','?'):10s}  "
            f"{_sign(stats['best_pnl'])} EUR",
        f"  Worst trade        : {stats['worst_trade'].get('ticker','?'):10s}  "
            f"{_sign(stats['worst_pnl'])} EUR",
    ]
    if open_trades is not None:
        lines.append(f"  Currently open     : {len(open_trades)} position(s)")
    return "\n".join(lines)


def render_trades_table(trades: list[dict]) -> str:
    if not trades:
        return "  (none)"
    hdr = (f"  {'Date':10}  {'Ticker':8}  {'Dir':4}  {'Alert':22}  "
           f"{'Entry':8}  {'Exit':8}  {'P&L':>9}  Result")
    rows = [sep(), hdr, sep()]
    for t in trades:
        pnl    = t.get("pnl_net") or 0
        result = "WIN " if pnl > 0 else "LOSS"
        exit_reason = ""
        if t.get("take_profit") and t.get("exit_price"):
            if abs(t["exit_price"] - t["take_profit"]) < 0.05:
                exit_reason = " (TP)"
            elif t.get("stop_loss") and abs(t["exit_price"] - t["stop_loss"]) < 0.05:
                exit_reason = " (SL)"
        rows.append(
            f"  {str(t.get('date',''))[:10]:10}  "
            f"{str(t.get('ticker','?'))[:8]:8}  "
            f"{str(t.get('trade_direction',''))[:4]:4}  "
            f"{str(t.get('alert_name',''))[:22]:22}  "
            f"{(t.get('entry_price') or 0):8.2f}  "
            f"{(t.get('exit_price') or 0):8.2f}  "
            f"{_sign(pnl):>9}  "
            f"{result}{exit_reason}"
        )
    rows.append(sep())
    return "\n".join(rows)


def render_by_ticker(ticker_stats: dict[str, dict]) -> str:
    if not ticker_stats:
        return "  (none)"
    sorted_items = sorted(ticker_stats.items(),
                          key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    max_abs = max(abs(s.get("total_pnl", 0)) for _, s in sorted_items) or 1
    lines = [sep(), f"  {'Ticker':10}  {'Trades':6}  {'W/L':6}  "
             f"{'Hit%':6}  {'Total P&L':>10}  {'Avg':>8}"]
    lines.append(sep())
    for ticker, s in sorted_items:
        bar = _bar(s["total_pnl"], max_abs, width=12,
                   char="+" if s["total_pnl"] >= 0 else "-")
        lines.append(
            f"  {ticker:10}  {s['n']:6}  "
            f"{s['n_wins']}W/{s['n_losses']}L  "
            f"{s['hit_rate']:5.0%}  "
            f"{_sign(s['total_pnl']):>10}  "
            f"{_sign(s['avg_pnl']):>8}  {bar}"
        )
    lines.append(sep())
    return "\n".join(lines)


def render_by_alert(alert_stats: dict[str, dict]) -> str:
    if not alert_stats:
        return "  (none)"
    sorted_items = sorted(alert_stats.items(),
                          key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    lines = [sep(), f"  {'Alert type':26}  {'N':4}  {'W/L':6}  "
             f"{'Hit%':5}  {'Total':>9}  {'Avg':>8}"]
    lines.append(sep())
    for alert, s in sorted_items:
        lines.append(
            f"  {alert[:26]:26}  {s['n']:4}  "
            f"{s['n_wins']}W/{s['n_losses']}L  "
            f"{s['hit_rate']:4.0%}  "
            f"{_sign(s['total_pnl']):>9}  "
            f"{_sign(s['avg_pnl']):>8}"
        )
    lines.append(sep())
    return "\n".join(lines)


def render_open_positions(open_trades: list[dict]) -> str:
    if not open_trades:
        return "  (none)"
    lines = [sep(),
             f"  {'Ticker':10}  {'Dir':4}  {'Alert':22}  {'Entry':8}  {'SL':8}  {'TP':8}",
             sep()]
    for t in open_trades:
        lines.append(
            f"  {str(t.get('ticker','?'))[:10]:10}  "
            f"{str(t.get('trade_direction',''))[:4]:4}  "
            f"{str(t.get('alert_name',''))[:22]:22}  "
            f"{(t.get('entry_price') or 0):8.2f}  "
            f"{(t.get('stop_loss') or 0):8.2f}  "
            f"{(t.get('take_profit') or 0):8.2f}"
        )
    lines.append(sep())
    return "\n".join(lines)


# ── Report sections ────────────────────────────────────────────────────────────

def report_period(db: str, label: str, start: date, end: date,
                  show_open: bool = False) -> str:
    trades      = get_trades(db, start, end)
    stats       = compute_stats(trades)
    open_trades = get_open_trades(db) if show_open else None
    ticker_s    = by_ticker(trades)
    alert_s     = by_alert(trades)

    sections = [header(label)]

    sections.append("\n  OVERVIEW\n" + sep("-"))
    sections.append(render_overview(stats, open_trades))

    if show_open and open_trades:
        sections.append("\n  OPEN POSITIONS\n" + sep("-"))
        sections.append(render_open_positions(open_trades))

    if trades:
        sections.append("\n  COMPLETED TRADES\n" + sep("-"))
        sections.append(render_trades_table(trades))

        sections.append("\n  BY STOCK\n" + sep("-"))
        sections.append(render_by_ticker(ticker_s))

        sections.append("\n  BY ALERT TYPE\n" + sep("-"))
        sections.append(render_by_alert(alert_s))

        # Stocks traded list
        tickers_traded = sorted({t["ticker"] for t in trades})
        sections.append(f"\n  STOCKS TRADED: {', '.join(tickers_traded)}")

    return "\n".join(sections)


def report_daily(db: str, for_date: date | None = None) -> str:
    d = for_date or date.today()
    label = f"DAILY REPORT  -  {d.strftime('%A, %d %B %Y')}"
    return report_period(db, label, d, d, show_open=True)


def report_weekly(db: str, ref: date | None = None) -> str:
    d    = ref or date.today()
    mon  = d - timedelta(days=d.weekday())
    sun  = mon + timedelta(days=6)
    label = f"WEEKLY REPORT  -  Week {mon.strftime('%d %b')} to {sun.strftime('%d %b %Y')}"
    return report_period(db, label, mon, sun)


def report_monthly(db: str, ref: date | None = None) -> str:
    d     = ref or date.today()
    start = d.replace(day=1)
    # last day of month
    if d.month == 12:
        end = d.replace(day=31)
    else:
        end = d.replace(month=d.month + 1, day=1) - timedelta(days=1)
    label = f"MONTHLY REPORT  -  {d.strftime('%B %Y')}"
    return report_period(db, label, start, end)


def report_all(db: str) -> str:
    with _conn(db) as conn:
        first = conn.execute(
            "SELECT MIN(date) FROM trades WHERE exit_price IS NOT NULL"
        ).fetchone()[0]
    if not first:
        return "No completed trades found."
    start = date.fromisoformat(first)
    end   = date.today()
    label = f"ALL-TIME REPORT  -  {start} to {end}"
    return report_period(db, label, start, end)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading performance report")
    parser.add_argument("--today",  action="store_true", help="Today's report (default)")
    parser.add_argument("--week",   action="store_true", help="This week's report")
    parser.add_argument("--month",  action="store_true", help="This month's report")
    parser.add_argument("--all",    action="store_true", help="All-time report")
    parser.add_argument("--date",   type=str,            help="Specific date YYYY-MM-DD")
    parser.add_argument("--db",     type=str, default=DB_PATH, help="Path to journal.db")
    args = parser.parse_args()

    db = args.db
    if not Path(db).exists():
        print(f"Database not found: {db}")
        return

    if args.date:
        d = date.fromisoformat(args.date)
        print(report_daily(db, for_date=d))
    elif args.week:
        print(report_weekly(db))
    elif args.month:
        print(report_monthly(db))
    elif args.all:
        print(report_all(db))
    else:
        # Default: today
        print(report_daily(db))

    print()


if __name__ == "__main__":
    main()
