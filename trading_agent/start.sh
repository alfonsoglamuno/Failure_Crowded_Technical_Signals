#!/usr/bin/env bash
# ============================================================
# Trading Agent — Start Script (Linux / macOS)
# Account: set IBKR_ACCOUNT in .env (gitignored)
# ============================================================
set -e
cd "$(dirname "$0")"

# ── 1. Ensure virtual environment ───────────────────────────
if [ ! -f ".venv/bin/python" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv .venv
fi

# ── 2. Activate ─────────────────────────────────────────────
source .venv/bin/activate

# ── 3. Dependencies ─────────────────────────────────────────
echo "[SETUP] Checking dependencies..."
pip install -q -r requirements.txt

# ── 4. Bootstrap model if missing ───────────────────────────
if [ ! -f "data/model/xgboost_h3d.joblib" ]; then
    echo "[SETUP] Model not found — bootstrapping with yfinance..."
    python bootstrap_model.py --yfinance
fi

# ── 5. Route command ────────────────────────────────────────
MODE="${1:-paper}"

case "$MODE" in
    check)
        echo "[CHECK] Running pre-flight connection check..."
        python check_connection.py
        ;;
    dashboard)
        echo "[DASHBOARD] Launching live P&L monitor..."
        python dashboard.py --watch
        ;;
    status)
        python run_agent.py --status
        ;;
    once)
        echo "[AGENT] Running single paper trading cycle..."
        python run_agent.py --paper --once
        ;;
    live)
        echo "[WARNING] LIVE TRADING — real money!"
        python run_agent.py --live
        ;;
    *)
        echo "[AGENT] Scheduled paper trading — 09:15 CET weekdays (Ctrl+C to stop)"
        python run_agent.py --paper
        ;;
esac
