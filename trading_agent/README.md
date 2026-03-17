# Failure of Crowded Technical Signals — Trading Agent

ML-driven paper trading agent for EURO STOXX 50 stocks.
Predicts when technical alerts will **fail** and trades the reversal.

> **Instrument scope: cash equities only.**
> This agent trades **shares (stocks) exclusively** using bracket orders (market entry + limit take-profit + stop-loss).
> Options, futures, CFDs, and all other derivatives are **not used and not required**. No options trading permissions are needed.

---

## How It Works

Every trading day at **09:15 CET**:

```
IBKR data ──► Alert Detection ──► Feature Engineering ──► XGBoost
                 (18 signal types)    (53 features)        P(failure)
                                                               │
                          P ≥ 0.60 ──► FADE ──► Contrarian bracket order
                          P ≤ 0.40 ──► FOLLOW ──► Momentum bracket order
                          between  ──► SKIP
                                               │
                                         Journal (SQLite)
                                               │
                                         Adaptive Learner
                                      (recalibrate thresholds,
                                       retrain model every 2 weeks)
```

### Model Performance (validation set, 3,159 events)

| Decile | P(failure) | Actual Fail % | Action |
|--------|-----------|---------------|--------|
| Top 10% | 0.75-0.99 | **82.0%** | FADE |
| Top 20% | 0.63-0.75 | 62.7% | FADE |
| Bottom 10% | 0.04-0.26 | 18.6% | FOLLOW |

> Break-even needed with 2,000 EUR position: **37.5%**. Model delivers **69.6%** on FADE signals.

---

## Prerequisites

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)
2. **IBKR account permissions needed:** stock trading on European exchanges only. **No options permissions required.**
3. **IB Gateway** — download at:
   `https://www.interactivebrokers.com/en/trading/ibgateway-stable.html`

   First-time IB Gateway setup:
   - Log in with your IBKR username and select **Paper Trading** account
   - Go to **Configure → Settings → API → Enable ActiveX and Socket Clients**
   - Set **Socket port** to `4002`
   - Tick **Allow connections from localhost only**
   - Click **OK** and restart IB Gateway

---

## Quick Start

### Step 0 — Configure credentials

Copy `.env.example` to `.env` and fill in your IBKR details:

```bash
cp .env.example .env
```

```ini
# .env  (gitignored — never commit this)
IBKR_ACCOUNT=U12345678      # your paper account ID
IBKR_PAPER_PORT=4002
IBKR_LIVE_PORT=4001
```

### Step 1 — Install dependencies

```bat
cd trading_agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Step 2 — Start trading (Windows)

```bat
start.bat              # scheduled paper trading at 09:15 CET (default)
start.bat once         # run one cycle right now
start.bat dashboard    # open live P&L dashboard
start.bat status       # print quick performance summary
start.bat live         # !! real money — confirm required !!
```

### Step 2 — Start trading (Linux / macOS)

```bash
chmod +x start.sh
./start.sh             # scheduled paper trading
./start.sh once        # run one cycle right now
./start.sh dashboard   # open live P&L dashboard
./start.sh status      # quick status
./start.sh live        # !! real money !!
```

### Manual commands (if you prefer)

```bash
# Bootstrap model (no IB Gateway needed)
python bootstrap_model.py --yfinance

# Run one paper trading cycle now
python run_agent.py --paper --once

# Start scheduled paper trading (09:15 CET weekdays)
python run_agent.py --paper

# Live P&L dashboard (refreshes every 30s, no IBKR needed)
python dashboard.py --watch

# Quick status snapshot
python run_agent.py --status
```

---

## Dashboard

```
python dashboard.py             # snapshot
python dashboard.py --watch     # live refresh every 30s
python dashboard.py --trades    # include full trade history
```

Shows:
- Total P&L, hit rate, profit factor
- Open positions with entry/SL/TP levels
- Daily P&L history (last 14 days)
- Per-alert-type performance breakdown
- Today's signals with probabilities

---

## Risk Configuration (`configs/config.yaml`)

Current settings for trial / paper account:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Capital | 100,000 EUR | Working allocation from paper account |
| Position size | 2% NAV | ~2,000 EUR per trade |
| Stop-loss | 2% | |
| Take-profit | 4% | 2:1 risk/reward |
| Max trades/day | 5 | FADE both long and short |
| Commission (est.) | 5 EUR | 0.05% × 2,000 EUR × 2 = ~2 EUR + exchange fees |
| Max daily loss | 300 EUR | 0.3% of capital |
| Fade threshold | 0.60 | P(failure) ≥ 0.60 → trade |
| Follow disabled | true | Enable after 50+ FADE trades |

Break-even: **37.5% hit rate** needed. Model delivers **69.6%** on FADE signals.

---

## The Adaptive Learner

The agent continuously learns from outcomes:

### Fast loop (every 10 trades)
- If FADE win rate drops below 50% → raise `fade_threshold` (be more selective)
- If FADE win rate exceeds 65% → lower `fade_threshold` (take more trades)
- Per-alert-type breakdown logged to identify which alerts work best

### Slow loop (every 14 days)
- Full XGBoost retrain on all live-traded data
- Model literally learns from its own trading outcomes
- Learner state persisted in `data/model/learner_state.json`

---

## File Structure

```
trading_agent/
├── start.bat               ← Windows launch script
├── start.sh                ← Linux/Mac launch script
├── run_agent.py            ← Main agent entry point
├── bootstrap_model.py      ← One-time model training
├── dashboard.py            ← Live P&L monitor
├── .env                    ← IBKR credentials (gitignored)
├── configs/
│   ├── config.yaml         ← All settings
│   └── ibkr_contracts.yaml ← EURO STOXX 50 symbol mappings
├── agent/
│   ├── alerts.py           ← 18 technical signal detectors
│   ├── features.py         ← 53-feature vector builder
│   ├── model.py            ← XGBoost loader/predictor
│   ├── strategy.py         ← FADE / FOLLOW / SKIP logic
│   ├── risk.py             ← Position sizing, daily limits
│   ├── executor.py         ← IBKR bracket orders
│   ├── journal.py          ← SQLite trade log
│   ├── learner.py          ← Threshold recalibration + retrain
│   ├── monitor.py          ← Detects closed bracket orders
│   └── data_feed.py        ← IBKR connection + OHLCV fetch
└── data/
    ├── model/              ← Trained model + feature list (gitignored)
    ├── cache/              ← OHLCV parquet cache (gitignored)
    └── journal.db          ← Trade database (gitignored)
```

---

## Decision Logic

| P(failure) | Action | Trade direction |
|-----------|--------|----------------|
| ≥ 0.60 | **FADE** | Opposite of alert (fade the crowd) |
| 0.40-0.60 | **SKIP** | No trade — model uncertain |
| ≤ 0.40 | **FOLLOW** | Same as alert (disabled by default) |

**FADE examples:**
- Bullish alert (e.g. RSI overbought) + P(failure)=0.75 → **SELL** (expect reversal)
- Bearish alert (e.g. break below 50d MA) + P(failure)=0.82 → **BUY** (expect bounce)

---

## Cron Automation (Linux/Mac)

```bash
crontab -e
# Add (adjust path):
15 9 * * 1-5 cd /path/to/trading_agent && .venv/bin/python run_agent.py --paper >> data/cron.log 2>&1
```

---

## Troubleshooting

**"Model not found"**
```bash
python bootstrap_model.py --yfinance
```

**"Could not connect to IB Gateway"**
- Make sure IB Gateway is open and you're logged into your paper account
- Check API is enabled on port 4002
- If firewall blocks it: Add IB Gateway to Windows Firewall exceptions

**"No alerts today"**
- Normal — some days have no qualifying signals
- Agent logs this and exits cleanly

**Check the log**
```bash
type data\agent.log     # Windows
tail -f data/agent.log  # Linux/Mac
```

---

## Go Live

After 2-4 weeks of paper trading with satisfactory results:

1. Switch IB Gateway to **Live** account
2. Verify funds and permissions for **European stock trading** (no options permissions needed)
3. Run: `start.bat live` (will ask for confirmation)

Required IBKR permissions for live:
- Trading permissions: **Stocks** on European exchanges (XETRA, Euronext, LSE, etc.)
- Market data: European equities delayed or real-time
- **Options trading is not used — do not enable it**

> Strongly recommended: maintain paper trading alongside live for comparison.
