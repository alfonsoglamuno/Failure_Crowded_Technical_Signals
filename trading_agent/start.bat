@echo off
:: ============================================================
:: Trading Agent — Start Script (Windows)
:: Account: set IBKR_ACCOUNT in .env (gitignored)
:: ============================================================

setlocal
cd /d "%~dp0"

:: ── 1. Ensure virtual environment exists ────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create venv. Is Python 3.10+ installed?
        pause
        exit /b 1
    )
)

:: ── 2. Activate ─────────────────────────────────────────────
call .venv\Scripts\activate.bat

:: ── 3. Install / update dependencies ────────────────────────
echo [SETUP] Checking dependencies...
pip install -q -r requirements.txt

:: ── 4. Ensure model exists ───────────────────────────────────
if not exist "data\model\xgboost_h3d.joblib" (
    echo [SETUP] Model not found — running bootstrap with yfinance...
    python bootstrap_model.py --yfinance
    if errorlevel 1 (
        echo ERROR: Bootstrap failed. Check your internet connection.
        pause
        exit /b 1
    )
)

:: ── 5. Parse argument ────────────────────────────────────────
set MODE=%1
if "%MODE%"=="" set MODE=paper

if /i "%MODE%"=="check" (
    echo [CHECK] Running pre-flight connection check...
    python check_connection.py
    goto :end
)

if /i "%MODE%"=="dashboard" (
    echo [DASHBOARD] Launching live P^&L monitor...
    python dashboard.py --watch
    goto :end
)

if /i "%MODE%"=="status" (
    python run_agent.py --status
    goto :end
)

if /i "%MODE%"=="once" (
    echo [AGENT] Running single trading cycle (paper)...
    python run_agent.py --paper --once
    goto :end
)

if /i "%MODE%"=="live" (
    echo [WARNING] LIVE TRADING MODE — real money will be used!
    python run_agent.py --live
    goto :end
)

:: Default: scheduled paper trading
echo [AGENT] Starting scheduled paper trading (09:15 CET weekdays)...
echo         Press Ctrl+C to stop
echo.
python run_agent.py --paper

:end
endlocal
