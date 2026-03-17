"""
Trading report — daily, weekly, monthly performance summaries.

All P&L figures are NET of commissions.
Return % is always relative to capital actually deployed (entry_price x quantity).
Each period shows benchmark comparison vs Euro STOXX 50 index (^STOXX50E).

Usage:
    python report.py                  # today
    python report.py --week
    python report.py --month
    python report.py --date 2026-03-17
    python report.py --all
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = "data/journal.db"
W = 76
INDEX_TICKER = "^STOXX50E"    # Euro STOXX 50 — our benchmark


# ── Benchmark (Euro STOXX 50) ──────────────────────────────────────────────────

def _fetch_index_return(start: date, end: date) -> tuple[float | None, float | None, float | None]:
    """
    Return (index_return_fraction, base_price, end_price) for the Euro STOXX 50
    over [start, end].

    Base price is always the last closing price BEFORE `start` (strictly less than),
    so daily reports (start==end==today) correctly show today's move vs yesterday's close.
    End price is the latest available close on or before `end`.

    Returns (None, None, None) if data is unavailable.
    """
    try:
        import yfinance as yf
        from datetime import timedelta as td
        fetch_start = start - td(days=10)  # 10-day buffer covers weekends/holidays
        fetch_end   = end   + td(days=1)
        df = yf.download(INDEX_TICKER, start=str(fetch_start), end=str(fetch_end),
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, None, None
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        if len(close) < 2:
            return None, None, None
        # Base = last close STRICTLY BEFORE start (prev trading day)
        base_candidates = close[close.index.date < start]
        end_candidates  = close[close.index.date <= end]
        if base_candidates.empty or end_candidates.empty:
            return None, None, None
        base_price = float(base_candidates.iloc[-1])
        end_price  = float(end_candidates.iloc[-1])
        if base_price == 0:
            return None, None, None
        return (end_price - base_price) / base_price, base_price, end_price
    except Exception:
        return None, None, None


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _sign(v: float, decimals: int = 2) -> str:
    fmt = f"+{{:.{decimals}f}}" if v >= 0 else f"{{:.{decimals}f}}"
    return fmt.format(v)


def _pct(v: float) -> str:
    """Format a ratio as a signed percentage string."""
    return _sign(v * 100, 2) + "%"


def _bar(value: float, max_val: float, width: int = 14) -> str:
    if max_val == 0:
        return ""
    filled = int(abs(value) / max_val * width)
    char = "+" if value >= 0 else "-"
    return char * min(filled, width)


def sep(char: str = "-") -> str:
    return char * W


def header(title: str) -> str:
    return f"\n{'=' * W}\n  {title}\n{'=' * W}"


# ── Capital deployed per trade ─────────────────────────────────────────────────

def _invested(t: dict) -> float:
    """Capital deployed = entry_price x quantity (gross position size)."""
    ep = t.get("entry_price") or 0
    qty = t.get("quantity") or 0
    return ep * qty


def _commission(t: dict) -> float:
    """Commission = pnl_gross - pnl_net."""
    gross = t.get("pnl_gross") or 0
    net   = t.get("pnl_net")
    if net is None:
        return 0.0
    return gross - net


def _return_pct(t: dict) -> float | None:
    """Net return as fraction of capital invested."""
    inv = _invested(t)
    net = t.get("pnl_net")
    if not inv or net is None:
        return None
    return net / inv


def _exit_label(t: dict) -> str:
    ep = t.get("exit_price") or 0
    tp = t.get("take_profit") or 0
    sl = t.get("stop_loss") or 0
    if tp and abs(ep - tp) / tp < 0.005:
        return "TP"
    if sl and abs(ep - sl) / sl < 0.005:
        return "SL"
    return "MKT"


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_trades(db: str, start: date, end: date) -> list[dict]:
    with _conn(db) as conn:
        rows = conn.execute("""
            SELECT t.*, s.action, s.alert_name, s.alert_direction,
                   s.failure_proba, s.crowding_score, s.conviction
            FROM trades t
            LEFT JOIN signals s ON t.signal_id = s.id
            WHERE t.exit_price IS NOT NULL
              AND t.pnl_net IS NOT NULL
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
            WHERE t.exit_price IS NULL
              AND t.status NOT IN ('cancelled','error')
            ORDER BY t.date DESC
        """).fetchall()
    return [dict(r) for r in rows]


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}

    completed = [t for t in trades if t.get("pnl_net") is not None]
    if not completed:
        return {}

    pnls        = [t["pnl_net"] for t in completed]
    rets        = [r for t in completed for r in [_return_pct(t)] if r is not None]
    commissions = [_commission(t) for t in completed]
    invested    = [_invested(t) for t in completed]

    wins   = [t for t in completed if t["pnl_net"] > 0]
    losses = [t for t in completed if t["pnl_net"] <= 0]

    best  = max(completed, key=lambda t: _return_pct(t) or -9e9)
    worst = min(completed, key=lambda t: _return_pct(t) or 9e9)

    gross_wins   = sum(t["pnl_net"] for t in wins)
    gross_losses = abs(sum(t["pnl_net"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    total_invested = sum(invested)
    total_pnl      = sum(pnls)
    total_comm     = sum(commissions)

    return {
        "n":                len(completed),
        "n_wins":           len(wins),
        "n_losses":         len(losses),
        "hit_rate":         len(wins) / len(completed),
        "total_pnl":        total_pnl,
        "total_commission": total_comm,
        "total_invested":   total_invested,
        "roic":             total_pnl / total_invested if total_invested else 0,
        "avg_pnl":          total_pnl / len(completed),
        "avg_return_pct":   sum(rets) / len(rets) if rets else 0,
        "avg_win_pct":      sum(_return_pct(t) or 0 for t in wins) / len(wins) if wins else 0,
        "avg_loss_pct":     sum(_return_pct(t) or 0 for t in losses) / len(losses) if losses else 0,
        "best_pnl":         best.get("pnl_net", 0),
        "worst_pnl":        worst.get("pnl_net", 0),
        "best_return_pct":  _return_pct(best) or 0,
        "worst_return_pct": _return_pct(worst) or 0,
        "best_trade":       best,
        "worst_trade":      worst,
        "profit_factor":    profit_factor,
    }


def by_ticker(trades: list[dict]) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for t in trades:
        groups[t["ticker"]].append(t)
    return {tk: compute_stats(ts) for tk, ts in sorted(groups.items())}


def by_alert(trades: list[dict]) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for t in trades:
        groups[t.get("alert_name") or "unknown"].append(t)
    return {k: compute_stats(v) for k, v in groups.items()}


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_benchmark(agent_roic: float,
                     index_ret: float | None,
                     base_price: float | None = None,
                     end_price: float | None = None) -> str:
    """Benchmark comparison block showing agent vs Euro STOXX 50."""
    agent_str = _pct(agent_roic)
    if index_ret is None:
        idx_str = "n/a (no market data)"
        return (f"  Agent return (ROIC): {agent_str}\n"
                f"  Euro STOXX 50      : {idx_str}")
    idx_str = _pct(index_ret)
    price_detail = (f"  ({base_price:.1f} → {end_price:.1f})"
                    if base_price and end_price else "")
    alpha = agent_roic - index_ret
    alpha_str = _pct(alpha)
    if alpha > 0.001:
        verdict = "BEAT INDEX"
    elif alpha < -0.001:
        verdict = "LAGGED INDEX"
    else:
        verdict = "MATCHED INDEX"
    return (f"  Agent return (ROIC): {agent_str}\n"
            f"  Euro STOXX 50      : {idx_str}{price_detail}\n"
            f"  Alpha vs index     : {alpha_str}  →  {verdict}")


def render_overview(stats: dict, open_trades: list[dict] | None = None,
                    index_ret: float | None = None,
                    base_price: float | None = None,
                    end_price: float | None = None) -> str:
    if not stats:
        idx_line = ""
        if index_ret is not None:
            price_detail = (f"  ({base_price:.1f} → {end_price:.1f})"
                            if base_price and end_price else "")
            idx_line = f"\n  Euro STOXX 50      : {_pct(index_ret)}{price_detail}  (no agent trades to compare)"
        open_line = f"\n  Currently open     : {len(open_trades)} position(s)" if open_trades else ""
        return f"  No completed trades in this period.{idx_line}{open_line}\n"

    pf = (f"{stats['profit_factor']:.2f}"
          if stats['profit_factor'] != float("inf") else "inf (no losses)")

    lines = [
        f"  Trades completed   : {stats['n']}  "
            f"({stats['n_wins']}W / {stats['n_losses']}L  "
            f"hit rate {stats['hit_rate']:.1%})",
        sep(),
        f"  Total P&L (net)    : {_sign(stats['total_pnl'])} EUR"
            f"   ({_pct(stats['roic'])} on capital deployed)",
        f"  Commissions paid   : -{stats['total_commission']:.2f} EUR"
            f"   ({stats['total_commission'] / stats['n']:.2f} EUR avg/trade)",
        f"  Capital deployed   : {stats['total_invested']:,.0f} EUR total",
        sep(),
        render_benchmark(stats['roic'], index_ret, base_price, end_price),
        sep(),
        f"  Avg return/trade   : {_pct(stats['avg_return_pct'])} net"
            f"   ({_sign(stats['avg_pnl'])} EUR)",
        f"  Avg win            : {_pct(stats['avg_win_pct'])}"
            f"   |  Avg loss: {_pct(stats['avg_loss_pct'])}",
        f"  Profit factor      : {pf}",
        sep(),
        f"  Best trade         : {stats['best_trade'].get('ticker','?'):10s}"
            f"  {_pct(stats['best_return_pct']):>8}  ({_sign(stats['best_pnl'])} EUR)",
        f"  Worst trade        : {stats['worst_trade'].get('ticker','?'):10s}"
            f"  {_pct(stats['worst_return_pct']):>8}  ({_sign(stats['worst_pnl'])} EUR)",
    ]
    if open_trades is not None:
        lines.append(f"  Currently open     : {len(open_trades)} position(s)")
    return "\n".join(lines)


def render_trades_table(trades: list[dict]) -> str:
    if not trades:
        return "  (none)"
    hdr = (f"  {'Date':10}  {'Ticker':8}  {'Alert':20}  "
           f"{'Invested':>10}  {'Net P&L':>9}  {'Return%':>8}  {'Exit':4}  Result")
    rows = [sep(), hdr, sep()]
    for t in trades:
        pnl  = t.get("pnl_net") or 0
        ret  = _return_pct(t)
        comm = _commission(t)
        inv  = _invested(t)
        result = "WIN " if pnl > 0 else "LOSS"
        ret_str = _pct(ret) if ret is not None else "  n/a  "
        rows.append(
            f"  {str(t.get('date',''))[:10]:10}  "
            f"{str(t.get('ticker','?'))[:8]:8}  "
            f"{str(t.get('alert_name',''))[:20]:20}  "
            f"{inv:>10,.0f}  "
            f"{_sign(pnl):>9}  "
            f"{ret_str:>8}  "
            f"{_exit_label(t):4}  "
            f"{result}  (comm {comm:.2f})"
        )
    rows.append(sep())
    # Commission total line
    total_comm = sum(_commission(t) for t in trades)
    total_pnl  = sum(t.get("pnl_net") or 0 for t in trades)
    total_inv  = sum(_invested(t) for t in trades)
    rows.append(
        f"  {'TOTAL':10}  {'':8}  {'':20}  "
        f"{total_inv:>10,.0f}  "
        f"{_sign(total_pnl):>9}  "
        f"{_pct(total_pnl / total_inv) if total_inv else '':>8}  "
        f"      comm total: {total_comm:.2f} EUR"
    )
    rows.append(sep())
    return "\n".join(rows)


def render_by_ticker(ticker_stats: dict[str, dict]) -> str:
    if not ticker_stats:
        return "  (none)"
    items = sorted(ticker_stats.items(),
                   key=lambda x: x[1].get("avg_return_pct", 0), reverse=True)
    max_abs = max(abs(s.get("avg_return_pct", 0)) for _, s in items) or 1
    lines = [sep(),
             f"  {'Ticker':10}  {'N':3}  {'W/L':6}  {'Hit%':5}  "
             f"{'Avg%net':>8}  {'Total EUR':>10}  {'TotalComm':>9}",
             sep()]
    for ticker, s in items:
        bar = _bar(s["avg_return_pct"], max_abs)
        lines.append(
            f"  {ticker:10}  {s['n']:3}  "
            f"{s['n_wins']}W/{s['n_losses']}L  "
            f"{s['hit_rate']:4.0%}  "
            f"{_pct(s['avg_return_pct']):>8}  "
            f"{_sign(s['total_pnl']):>10}  "
            f"{s['total_commission']:>9.2f}  {bar}"
        )
    lines.append(sep())
    return "\n".join(lines)


def render_by_alert(alert_stats: dict[str, dict]) -> str:
    if not alert_stats:
        return "  (none)"
    items = sorted(alert_stats.items(),
                   key=lambda x: x[1].get("avg_return_pct", 0), reverse=True)
    lines = [sep(),
             f"  {'Alert type':24}  {'N':3}  {'W/L':6}  {'Hit%':5}  "
             f"{'Avg%net':>8}  {'Total EUR':>10}  {'TotalComm':>9}",
             sep()]
    for alert, s in items:
        lines.append(
            f"  {alert[:24]:24}  {s['n']:3}  "
            f"{s['n_wins']}W/{s['n_losses']}L  "
            f"{s['hit_rate']:4.0%}  "
            f"{_pct(s['avg_return_pct']):>8}  "
            f"{_sign(s['total_pnl']):>10}  "
            f"{s['total_commission']:>9.2f}"
        )
    lines.append(sep())
    return "\n".join(lines)


def render_open_positions(open_trades: list[dict]) -> str:
    if not open_trades:
        return "  (none)"
    from datetime import datetime, timezone
    lines = [sep(),
             f"  {'Ticker':10}  {'Dir':4}  {'Entry':7}  {'SL':7}  {'TP':7}  "
             f"{'SL%':6}  {'TP%':6}  {'Size EUR':>9}  {'Open h':>6}  Status  Alert",
             sep()]
    now = datetime.now(timezone.utc)
    naked_count = 0
    for t in open_trades:
        ep  = t.get("entry_price") or 0
        sl  = t.get("stop_loss") or 0
        tp  = t.get("take_profit") or 0
        inv = _invested(t)
        direction = t.get("trade_direction", "BUY")
        status = t.get("status", "")
        sl_pct = (sl - ep) / ep if ep else 0
        tp_pct = (tp - ep) / ep if ep else 0
        try:
            entry_ts = datetime.fromisoformat(t["ts"]).replace(tzinfo=timezone.utc)
            hours_open = (now - entry_ts).total_seconds() / 3600
            hours_str = f"{hours_open:.1f}h"
        except Exception:
            hours_str = "  ?"
        alert = str(t.get("alert_name") or "")[:18]
        # Warn if stop is missing or position is pending_close
        if status == "pending_close":
            flag = "CLOSING"
        elif not sl:
            flag = "NO SL!"
            naked_count += 1
        else:
            flag = "active"
        lines.append(
            f"  {str(t.get('ticker','?'))[:10]:10}  "
            f"{direction[:4]:4}  "
            f"{ep:7.2f}  {sl:7.2f}  {tp:7.2f}  "
            f"{_pct(sl_pct):>6}  {_pct(tp_pct):>6}  "
            f"{inv:>9,.0f}  "
            f"{hours_str:>6}  "
            f"{flag:7}  {alert}"
        )
    lines.append(sep())
    total_exp = sum(_invested(t) for t in open_trades)
    summary = f"  Total exposure: {total_exp:,.0f} EUR  |  {len(open_trades)} open positions"
    if naked_count:
        summary += f"  *** {naked_count} position(s) WITHOUT stop-loss — monitor will re-place ***"
    lines.append(summary)
    lines.append(sep())
    return "\n".join(lines)


# ── Report assembly ────────────────────────────────────────────────────────────

def report_period(db: str, label: str, start: date, end: date,
                  show_open: bool = False) -> str:
    trades      = get_trades(db, start, end)
    stats       = compute_stats(trades)
    open_trades = get_open_trades(db) if show_open else None
    ticker_s    = by_ticker(trades)
    alert_s     = by_alert(trades)
    index_ret, base_px, end_px = _fetch_index_return(start, end)

    sections = [header(label)]

    sections.append("\n  OVERVIEW\n" + sep())
    sections.append(render_overview(stats, open_trades, index_ret, base_px, end_px))

    if show_open and open_trades:
        sections.append("\n  OPEN POSITIONS\n" + sep())
        sections.append(render_open_positions(open_trades))

    if trades:
        sections.append("\n  COMPLETED TRADES\n" + sep())
        sections.append(render_trades_table(trades))

        sections.append("\n  BY STOCK\n" + sep())
        sections.append(render_by_ticker(ticker_s))

        sections.append("\n  BY SIGNAL TYPE\n" + sep())
        sections.append(render_by_alert(alert_s))

        tickers_traded = sorted({t["ticker"] for t in trades})
        sections.append(f"\n  STOCKS TRADED: {', '.join(tickers_traded)}")

    return "\n".join(sections)


def report_daily(db: str, for_date: date | None = None) -> str:
    d = for_date or date.today()
    return report_period(db, f"DAILY REPORT  -  {d.strftime('%A, %d %B %Y')}",
                         d, d, show_open=True)


def report_weekly(db: str, ref: date | None = None) -> str:
    d   = ref or date.today()
    mon = d - timedelta(days=d.weekday())
    sun = mon + timedelta(days=6)
    return report_period(db, f"WEEKLY REPORT  -  {mon.strftime('%d %b')} to {sun.strftime('%d %b %Y')}",
                         mon, sun)


def report_monthly(db: str, ref: date | None = None) -> str:
    d     = ref or date.today()
    start = d.replace(day=1)
    end   = (d.replace(month=d.month + 1, day=1) - timedelta(days=1)
             if d.month < 12 else d.replace(day=31))
    return report_period(db, f"MONTHLY REPORT  -  {d.strftime('%B %Y')}", start, end)


def report_all(db: str) -> str:
    with _conn(db) as conn:
        first = conn.execute(
            "SELECT MIN(date) FROM trades WHERE exit_price IS NOT NULL AND pnl_net IS NOT NULL"
        ).fetchone()[0]
    if not first:
        return "No completed trades found."
    start = date.fromisoformat(first)
    return report_period(db, f"ALL-TIME REPORT  -  {start} to {date.today()}",
                         start, date.today())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading performance report")
    parser.add_argument("--today",  action="store_true")
    parser.add_argument("--week",   action="store_true")
    parser.add_argument("--month",  action="store_true")
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--date",   type=str, help="YYYY-MM-DD")
    parser.add_argument("--db",     type=str, default=DB_PATH)
    args = parser.parse_args()

    db = args.db
    if not Path(db).exists():
        print(f"Database not found: {db}")
        return

    if args.date:
        print(report_daily(db, for_date=date.fromisoformat(args.date)))
    elif args.week:
        print(report_weekly(db))
    elif args.month:
        print(report_monthly(db))
    elif args.all:
        print(report_all(db))
    else:
        print(report_daily(db))
    print()


if __name__ == "__main__":
    main()
