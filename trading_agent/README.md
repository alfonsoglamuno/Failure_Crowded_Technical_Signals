# Failure of Crowded Technical Signals — Trading Agent

Predicts when technical alerts will **fail** and trades the reversal.
Runs live against IBKR, EURO STOXX 50 cash equities only.

> **Default mode: `h1d_longonly` — intraday long-only, EOD close.**
> No shorts required. No overnight risk. All positions close at 16:30 CET.

---

## Disclaimer

**This is a personal research project. It is not a registered investment product, not financial advice, and not intended for real trading.**

Algorithmic trading involves significant financial risk. You can lose part or all of the capital you deploy. Past model performance on historical data does not guarantee future results. Market conditions, liquidity, and execution costs can change unpredictably.

**The author accept no responsibility whatsoever for any financial loss, missed opportunity, or damage of any kind arising from the use of this software.** By using or running this agent you acknowledge that you act entirely at your own risk and on your own responsibility.

---

## How It Works

On each scan cycle (every 30 min during market hours):

```
IBKR data ──► Alert Detection ──► Feature Engineering ──► XGBoost (active variant)
             (18 signal types)    (80+ features, 9 blocks)    P(failure)
                                                                    │
                        P(failure) >= threshold ──► FADE ──► contrarian bracket order
                        P(failure) <= threshold ──► FOLLOW ──► momentum bracket order
                        otherwise              ──► SKIP
                                                            │
                                                      Journal (SQLite)
                                                            │
                                                      Adaptive Learner
                                               (recalibrate thresholds,
                                                retrain monthly or on
                                                performance degradation)
```

Between scans, a 5-minute monitor trails stop-losses and enforces time-based exits.

---

## Quick Start

```bash
# 1. Install dependencies
cd trading_agent && pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env    # fill in IBKR_ACCOUNT, ports

# 3. Train all model variants (~5 min, 7yr data)
python bootstrap_model.py --yfinance

# 4. Start paper trading — uses h1d_both from config (intraday, EOD close)
python run_agent.py --paper --no-menu

# 5. Emergency: close specific positions manually
python manage_positions.py --ticker ASML.AS
python manage_positions.py --eod-close   # flatten everything now
```

```bat
# Windows shortcuts
start.bat          # paper trading
start.bat live     # !! real money — requires double confirmation !!
```

When started with `--live`, the agent shows a full disclaimer screen and requires typing `yes` then `confirmed` before any orders are placed.

---

## Model Variants

Six models across 3 horizons x 2 direction modes. Select with `--variant` or the interactive menu.

| Variant | Hold | EOD close | SL | TP | Corr gate | When to use |
|---------|------|-----------|----|----|-----------|-------------|
| **`h1d_both`** | **intraday** | **YES (16:27)** | **1.5%** | **2.5%** | **0.85** | **Active default. Long-only with allow_short=false** |
| `h1d_longonly` | intraday | YES (16:27) | 1.5% | 2.5% | 0.85 | Pure long-only training; lower signal count |
| `h3d_longonly` | 3 days | NO | 2.5% | 4.0% | 0.70 | Swing fades, positions survive overnight |
| `h5d_longonly` | 5 days | NO | 3.5% | 6.0% | 0.65 | Multi-day positions, wider SL/TP |
| `h3d_both` | 3 days | NO | 2.5% | 4.0% | 0.70 (dir-adj) | Swing, allow_short=true required |
| `h5d_both` | 5 days | NO | 3.5% | 6.0% | 0.65 (dir-adj) | Position, allow_short=true required |

**Active variant**: `h1d_both` with `allow_short=false`.
`both` means the model was trained on all alert directions (more data, better calibration).
SELL entries are blocked by the executor when `allow_short=false` — only BUY entries execute.
Closing existing longs (SL/TP/EOD exits) is always allowed regardless of this setting.

**`longonly`**: trained only on bearish/neutral alerts (fewer training samples, slightly lower AUC).
Better theoretical match for a long-only book but in practice `h1d_both` outperforms with 7yr data.

**Hold horizon and EOD behaviour**:
- h1d: EOD close at 16:27 CET flattens all positions. SL/TP orders use DAY TIF (expire automatically).
- h3d/h5d: NO EOD close. Positions survive overnight. SL/TP use GTC TIF. Morning check re-evaluates each session.

The variant is set in `configs/config.yaml → model.variant`.

---

## Evaluating Model Performance

```bash
# Open the notebook (most complete evaluation):
notebooks/01_results_overview.ipynb

# Or run the standalone script:
python reports/evaluate_models.py --variant h1d_both
```

The notebook shows: AUC, precision/recall at multiple thresholds, temporal stability across
Pre-COVID / COVID / Recovery / Recent windows, feature importance, calibration curves, and break-even analysis.

**Active model results (h1d_both, 7yr data, chronological 80/20 hold-out)**:
- AUC = 0.605 | P@0.60 = 67% | P@0.65 = 69%
- Break-even precision (SL=1.5%, TP=2.5%, commission 0.10% rt): ~40%
- Temporal stability: P@0.60 improves from 50% (Pre-COVID) to 71% (Recent) — model adapts over time

---

## Feature Set (9 blocks)

| Block | Name | Key features |
|-------|------|-------------|
| A | Alert | type, direction, simultaneous count |
| B | Price state | returns 1–20d, MA distance, 20d range position |
| C | Volatility / cost | ATR, realized vol, atr_vs_commission (gate for tight stocks) |
| D | Volume / attention | volume z-score, returnxvolume interaction, streak counts |
| E | Market regime | index trend, beta, correlation, relative strength vs index |
| F | Calendar | day-of-week, month-end, week-start/end |
| G | Peer correlation | avg 20d pairwise corr with all STOXX50 peers |
| H | Intraday proxies | close-vs-range, gap fill, VWAP distance, intrabar reversal |
| I | Trend / momentum | EMA slope, RSI level+trend, MACD histogram, vol acceleration |

Block I was added to give the model sensitivity to *direction and rate-of-change* of momentum, not just levels. A bearish alert into a falling RSI/MACD is more reliable than one into a rising RSI.

---

## Correlation Gate

Before placing any new trade, the agent checks the 20-day return correlation between the candidate and all open positions. If the maximum **effective correlation** exceeds the threshold, the trade is skipped.

**Why it matters**: two concentrated same-direction bets on highly correlated stocks provide almost no diversification — you are effectively doubling a single macro bet.

**Threshold by horizon**: the gate is stricter for longer holds because positions overlap for more days.
- h1d: 0.85 — positions close EOD regardless; gate is a soft filter against intraday crowding
- h3d: 0.70 — 3-day overlap risk
- h5d: 0.65 — 5-day overlap risk

**Direction-adjusted for short-enabled modes**: in `both` mode, a LONG on A and a SHORT on B (where A and B are highly correlated) is a **partially hedged pair** — when A rises, B also tends to rise, partially offsetting the short loss. The effective correlation formula is:

```
effective_corr = sign(candidate_direction) x sign(open_direction) x raw_corr
```

- Same direction + high corr → effective_corr > 0 → concentrated → **blocked**
- Opposite directions + high corr → effective_corr < 0 → hedged → **allowed**

In `longonly` mode, all positions are longs, so the direction factor is always +1 and raw |corr| is used directly.

---

## Commission Model

IBKR tiered pricing: **0.05% of trade value, minimum 2 EUR per order**.

| Trade value | Commission (per side) |
|-------------|----------------------|
| 500 EUR | 2.00 EUR (minimum) |
| 2,000 EUR | 2.00 EUR (minimum) |
| 5,000 EUR | 2.50 EUR |
| 10,000 EUR | 5.00 EUR |

The `atr_vs_commission` feature (ATR / 0.10% round-trip cost) explicitly models whether a stock's typical daily move is large enough to make the trade worth entering. The `expected_gross < 2 x commission` check in `risk.py` rejects trades where the edge is too thin even before sizing.

---

## Risk Parameters

Shared limits apply to all models. SL/TP and trail settings are **per-horizon** — each model uses values calibrated to its expected hold duration and typical price range (wider for longer holds):

| Parameter | h1d (intraday) | h3d (swing) | h5d (position) |
|-----------|---------------|-------------|----------------|
| Stop-loss | 1.5% | 2.5% | 3.5% |
| Take-profit | 2.5% | 4.0% | 6.0% |
| Reward/risk | 1.67:1 | 1.60:1 | 1.71:1 |
| Trail trigger | 1.0% | 1.5% | 2.0% |
| Trail step | 0.5% | 0.8% | 1.0% |
| Time exit | 6.5h open | none | none |
| EOD flatten | YES 16:27 | NO | NO |
| SL/TP TIF | DAY | GTC | GTC |

| Shared parameter | Value | Notes |
|-----------------|-------|-------|
| Capital | 100,000 EUR | Working allocation |
| Position size | 2% NAV | ~2,000 EUR per trade |
| Max open positions | 10 | Slot cap |
| Max daily loss | 300 EUR | 0.3% of capital |
| Commission | 0.05% min 2 EUR | IBKR tiered |
| Fade threshold | 0.60 | Active (h1d_both) |
| Allow short | false | SELL entries blocked |

---

## Adaptive Learner

### Fast loop (every 10 completed trades)
Recalibrates `fade_threshold` based on recent win/loss rate:
- Win rate < 50% → raise threshold (more selective)
- Win rate > 65% → lower threshold (capture more signals)

### Slow loop (monthly + performance trigger)
Full XGBoost retrain on current universe data. Triggers when:
1. Model file is more than 30 days old (monthly schedule), **or**
2. Live hit-rate drops more than 10pp below the baseline recorded at last retrain (regime shift)

Training uses exponential recency weighting (252-day half-life) so recent data has more influence without discarding the full history.

---

## Overnight Position Handling

The agent is designed to close all positions intraday. The EOD close at 16:30 CET cancels all bracket orders and flattens everything.

If a position survives to the next morning (connectivity loss, manual intervention), the **morning check** runs before any new signal scan:

1. Cover any accidental shorts (MKT BUY)
2. Gap-through-SL: if overnight price moved past the original stop → close immediately
3. Stale thesis: if held more than `max_hold_days` (default 3) and still at a loss → close
4. Re-place missing SL orders at a smart level:
   - At ≥ 50% progress toward TP → break-even lock
   - Small profit → SL from current price
   - At a loss → restore to original SL level

---

## Short Safety

Two layers prevent accidental short entries:

1. **Strategy filter**: SELL-direction signals are dropped when `allow_short=false`
2. **Executor hard block**: if a SELL entry reaches `place_bracket()` and `allow_short=false`, the order is refused before touching IBKR

SL/TP exit orders (which are also SELL actions for longs) bypass this check — they are child orders of an existing bracket, not new entries.

```bash
python close_shorts.py --shorts-only   # cover accidental shorts
python close_shorts.py --emergency     # flatten everything
```

---

## Bootstrap / Retrain

```bash
# Train all 6 variants (recommended, ~3 min)
python bootstrap_model.py --yfinance

# Train one variant
python bootstrap_model.py --yfinance --variant h1d_longonly
```

Retrain after:
- Adding or changing features in `src/features/engineering.py`
- Changing `allow_short` mode (longonly → both or vice versa)
- Adding new tickers to the universe
- Updating `history_days` in config

---

## File Structure

```
trading_agent/
├── run_agent.py            ← Main agent loop
├── bootstrap_model.py      ← Trains all 6 model variants
├── dashboard.py            ← Live P&L dashboard
├── close_shorts.py         ← Emergency position management
├── configs/
│   ├── config.yaml         ← All settings (model.variant, risk, strategy)
│   └── ibkr_contracts.yaml ← EURO STOXX 50 symbol mappings
└── agent/
    ├── alerts.py           ← 18 technical signal detectors
    ├── features.py         ← 80+ feature vector builder (live)
    ├── model.py            ← XGBoost loader/predictor
    ├── strategy.py         ← FADE / FOLLOW / SKIP logic + crowding gate
    ├── risk.py             ← Position sizing, commission model, daily limits
    ├── executor.py         ← IBKR bracket orders + short safety block
    ├── journal.py          ← SQLite trade log
    ├── learner.py          ← Threshold recalibration + monthly retrain
    ├── monitor.py          ← Exit sync, trailing SL, SL health check
    └── data_feed.py        ← IBKR connection + OHLCV fetch
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Model not found" | `python bootstrap_model.py --yfinance` |
| "Feature mismatch" | Features changed — rerun bootstrap |
| Accidental short | `python close_shorts.py --shorts-only` |
| Flatten everything | `python close_shorts.py --emergency` |
| Check logs | `tail -f data/agent.log` (Linux) or `type data\agent.log` (Windows) |
