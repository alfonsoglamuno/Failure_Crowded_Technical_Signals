# Failure of Crowded Technical Signals — Trading Agent

ML-driven paper trading agent for EURO STOXX 50 stocks.
Predicts when technical alerts will **fail** and trades the reversal.

> **Instrument scope: cash equities only.**
> This agent trades **shares (stocks) exclusively** using bracket orders (market entry + limit take-profit + stop-loss).
> Options, futures, CFDs, and all other derivatives are **not used and not required**.

---

## How It Works

On each scan cycle (every 30 min during market hours):

```
IBKR data ──► Alert Detection ──► Feature Engineering ──► XGBoost (active variant)
                 (18 signal types)    (74 features)           P(failure)
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

### Model Performance (h3d_longonly, 2-year validation)

| Threshold | Signals | Precision |
|-----------|---------|-----------|
| 0.55 | 921 | 71.0% |
| 0.60 | 757 | **75.0%** ← active |
| 0.65 | 604 | 79.0% |
| 0.70 | 495 | 83.4% |

> Break-even with 2,000 EUR position: **37.5%**. Active threshold delivers **75.0%**.

---

## Prerequisites

1. **Python 3.10+**
2. **IB Gateway** — paper trading on port 4002, API enabled
3. **IBKR account** with European stock trading permissions (no options needed)

---

## Quick Start

```bash
# 1. Install dependencies
cd trading_agent && pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env    # fill in IBKR_ACCOUNT, ports

# 3. Train all model variants (first run ~3 min)
python bootstrap_model.py --yfinance

# 4. Start paper trading
python run_agent.py --paper
```

```bat
# Windows shortcuts
start.bat          # scheduled paper trading
start.bat once     # one cycle now
start.bat live     # !! real money !!
```

---

## Model Variants

Six models are trained, covering **3 horizons** × **2 direction modes**.
Select the active variant in `configs/config.yaml → model.variant`.

| Variant | Horizon | Direction Mode | When to Use |
|---------|---------|----------------|-------------|
| `h1d_longonly` | 1 day | Long-only | Very short intraday holds, allow_short=false |
| **`h3d_longonly`** | **3 days** | **Long-only** | **Default — swing fades, allow_short=false** |
| `h5d_longonly` | 5 days | Long-only | Multi-day holds, allow_short=false |
| `h1d_both` | 1 day | Both directions | Short-term + allow_short=true |
| `h3d_both` | 3 days | Both directions | Symmetric long/short, allow_short=true |
| `h5d_both` | 5 days | Both directions | Position trading, allow_short=true |

### Direction mode explained

- **longonly**: trained only on bearish/neutral alerts — optimised for FADE→BUY entries.
  Better calibration for long-only books because the training set matches the trades.
- **both**: trained on all alert directions including bullish alerts.
  Required when `allow_short=true` because the model must also score SELL-side signals.

### Switching variants

```yaml
# configs/config.yaml
model:
  variant: "h1d_longonly"                                # change this
  path: "data/model/xgboost_h1d_longonly.joblib"        # update to match
  feature_cols_path: "data/model/feature_cols_h1d_longonly.json"
```

Then restart the agent — no retraining needed (all variants are pre-trained).
Re-run bootstrap only when `allow_short` mode changes or features are updated.

---

## Feature Set (74 features across 8 blocks)

The model is trained on features derived from microstructure research principles.
Below is a description of each block and the design rationale.

### Block A — Alert features
`direction_*`, `alert_name_*`, `n_simultaneous_alerts`

The alert type and direction are the primary signals. Simultaneous alerts on the same
ticker amplify the crowding effect — the more signals fire at once, the more attention
a stock attracts and the more likely the signal is noise rather than real information
(Barber & Odean, 2008).

### Block B — Short-term price state
`ret_1d/3d/5d/10d/20d`, `dist_ma10/20/50/100/200`, `price_pos_20d`

How overextended is the stock? Momentum and distance from mean-reversion anchors
(moving averages) are the core "how stretched" features. A stock far from its 50d MA
on high RSI is a crowded position and a candidate for fade.

### Block C — Volatility state
`realvol_5d/10d/20d`, `atr_14`, `atr_norm_14`, `candle_range_norm`, `gap_size`,
`atr_vs_commission`

`atr_vs_commission`: ATR divided by round-trip commission cost (0.10%). A stock with
ATR >> commission offers room to profit; tight stocks get filtered. The model learns
that low ATR-to-commission names are high-cost opportunities and penalises them.
Intraday volatility regime affects the probability of signal success dramatically —
low-volatility markets give more reliable fades; high-volatility creates continuation.

### Block D — Volume and crowding state
`vol_zscore_5d/20d`, `ret_vol_interaction`, `consec_up/down`

Volume z-score captures attention surges. The interaction term (|return| × volume z-score)
is the "crowding intensity" feature — a large move on elevated volume is exactly the
retail-attention scenario where fades work best.

### Block E — Market regime (index-level)
`index_ret_1d/5d/20d`, `index_vol_20d`, `index_above_ma50/200`,
`index_corr_20d/60d`, `beta_20d/60d`, `rel_strength_5d/20d/1d`, `index_regime`

Market regime is the single strongest context variable.
- `index_above_ma200`: bear vs bull market distinction. Fade signals are more reliable
  in bear markets where retail longs pile in on every bounce and get stopped out.
- `index_regime`: +1 bull / 0 neutral / -1 bear based on 20-day index return.
- `rel_strength_1d/5d/20d`: stock return minus index return. A move that is
  idiosyncratic (high rel_strength) is more likely to mean-revert than one that
  is just following the tape.
- `beta_20d/60d`: how much of the stock's moves are systematic vs idiosyncratic.
  High-beta names in bear markets fade harder.

### Block F — Calendar / session features
`dow`, `month`, `is_month_end/start`, `is_week_start/end`

Intraday trading is not stationary through time. Monday openings show institutional
repositioning; Fridays show position squaring; month-end rebalancing creates
systematic flows. The model learns to be more or less aggressive at each session type.

### Block G — Inter-stock peer correlation
`avg_peer_corr_20d`

Average pairwise 20-day return correlation with all other stocks in the EURO STOXX 50
universe. High value = herding regime where all stocks move together = crowded signals
are less stock-specific and more likely to be systematic noise. The model uses this to
discount signals in herded environments and trust them more when stocks are trading
idiosyncratically.

### Block H — Intraday structure (daily OHLCV proxies)
`close_vs_range`, `open_to_close_ret`, `gap_pct`, `gap_is_filled`,
`vwap_distance`, `vol_vs_dow_baseline`, `reversal_intrabar`

These approximate microstructure features without requiring L2/tick data:

- `close_vs_range`: where in today's high-low range did price close? 0 = at low
  (sellers dominated), 1 = at high (buyers dominated). A bearish alert that closed
  near the daily high may still have buyers — the fade is weaker.
- `open_to_close_ret`: signed intraday move (open → close). Separates gap from
  intraday order-flow direction.
- `gap_pct`: signed overnight gap. Large gap-downs attract retail bargain-hunting
  (attention-driven buying); the model weights these differently from trending declines.
- `gap_is_filled`: 1 if price traded back through the prior close during the session.
  A filled gap-down is a strong reversal signal (buyers absorbed the selling).
- `vwap_distance`: close vs (H+L+2C)/4 typical price. Close above VWAP = buyers won
  the session; below = sellers. Standard institutional benchmark.
- `vol_vs_dow_baseline`: volume vs 40-session rolling average for the same day-of-week.
  Approximates "relative volume at this time of day" — a concept from intraday
  market microstructure (unusually high volume for a Tuesday vs. unusually high
  volume for a Friday have different implications).
- `reversal_intrabar`: after a gap-down opening, how much did price recover intraday?
  A full recovery indicates strong buying absorption; this is the daily OHLCV proxy
  for order-flow imbalance.

### Why no L2/tick features?

Full order-book imbalance, cancel-to-add ratio, and aggressor-side flow are the
strongest intraday microstructure predictors (see academic references in the root
README). However, they require historical L2 data that IBKR does not provide via
the standard API. The Block H features are the closest daily-OHLCV approximations.

**If you gain access to 1-min or 5-min IBKR bars**, the most valuable additions would be:
- 1m/5m/15m returns and range (short-horizon momentum)
- Volume vs same-minute-of-day baseline (true intraday relative volume)
- Distance to intraday VWAP computed from 1-min bar cumulative (V × P)
- Opening range breakout/failure (first 30-min high-low)
- Realized variance from 5-min returns (microstructure-based vol estimate)

---

## Intraday-Only Policy (no overnight positions)

The agent is strictly **intraday**: all positions are closed by EOD.

### Morning check (FIRST at market open)
Before any new signal scan, `_evaluate_overnight_positions()` runs automatically:
1. Covers any accidental short positions immediately (MKT BUY)
2. Re-places missing stop-loss orders for surviving long positions
3. Reconciles IBKR positions against journal

This is a safety net — under normal operation, EOD close handles everything.

### EOD close
Cancels all bracket orders, then flattens all positions (SELL longs + BUY shorts).

### Mid-session SL guard
The 5-minute monitor checks for missing SL orders and re-places them automatically.

---

## Short Safety

The agent has two layers of protection against accidental short entries:

1. **Strategy filter** (`filter_signals`): SELL signals are dropped when `allow_short=false`
2. **Executor hard block** (`place_bracket`): if `trade_direction=="SELL"` reaches the
   executor and `allow_short=false`, the order is **refused before reaching IBKR**.

SL/TP exit orders (which are also SELL actions) bypass this check because they are
not entries — they are set as child orders of an existing bracket, not standalone entries.

The `close_shorts.py` utility handles manual intervention if a short appears:
```bash
python close_shorts.py --shorts-only  # cover all accidental shorts
python close_shorts.py --emergency    # flatten everything
```

---

## Correlation Gate

Before placing a new trade, the agent checks the 20-day return correlation between
the candidate and all currently open positions. If the maximum correlation exceeds
**0.70**, the trade is skipped.

**Why**: two highly correlated positions provide nearly no diversification — you are
effectively doubling a single bet. The correlation gate prevents portfolio crowding,
especially during sector-wide rotations where multiple stocks may fire alerts
simultaneously (and all be subject to the same macro driver).

---

## Commission Model

Commissions are calculated proportionally: **0.05% of trade value, minimum 2 EUR**.

This reflects IBKR's tiered pricing for European equities. The flat `commission_per_trade_eur`
config value is only used as a fallback when trade value is unknown (e.g. pre-fill estimates).

| Trade value | Est. commission (per side) |
|-------------|---------------------------|
| 500 EUR | 2.00 EUR (minimum) |
| 2,000 EUR | 2.00 EUR (minimum) |
| 5,000 EUR | 2.50 EUR |
| 10,000 EUR | 5.00 EUR |
| 20,000 EUR | 10.00 EUR |

The `expected_gross < 2 × commission` check in `risk.py` uses the proportional estimate,
so illiquid or tiny trades are correctly rejected without blocking normally-sized ones.

Actual IBKR commission from `commissionReport` fills is always preferred over estimates
when available.

---

## Risk Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Capital | 100,000 EUR | Working allocation |
| Position size | 2% NAV | ~2,000 EUR per trade |
| Stop-loss | 1.5% | STOXX50 daily ATR ≈ 1.5% |
| Take-profit | 2.5% | 5:3 risk/reward |
| Max open positions | 10 | Slot cap |
| Commission | 0.05% min 2€ | Proportional to trade size |
| Max daily loss | 300 EUR | 0.3% of capital |
| Fade threshold | 0.60 | Active variant default |
| Follow disabled | true | Enable after 50+ FADE trades |
| Allow short | false | Long-only until short locates confirmed |

---

## Adaptive Learner

### Fast loop (every 10 trades)
- Win rate < 50% → raise `fade_threshold` (more selective)
- Win rate > 65% → lower `fade_threshold` (take more trades)

### Slow loop (every 14 days)
- Full XGBoost retrain on live-traded outcomes
- Uses the currently active model variant

---

## Bootstrap / Retrain

```bash
# Train all 6 variants (recommended — ~3 min)
python bootstrap_model.py --yfinance

# Train one specific variant
python bootstrap_model.py --yfinance --variant h1d_longonly

# Use IBKR data instead of yfinance
python bootstrap_model.py --paper
```

Run bootstrap again after:
- Adding or changing features in `src/features/engineering.py`
- Changing `history_days` in config (clears cache automatically)
- Adding new tickers to the universe
- Enabling/disabling short trading (mode change)

---

## File Structure

```
trading_agent/
├── run_agent.py            ← Main agent loop
├── bootstrap_model.py      ← Trains all 6 model variants
├── dashboard.py            ← Live P&L dashboard
├── close_shorts.py         ← Manual position management
├── configs/
│   ├── config.yaml         ← All settings (including model.variant)
│   └── ibkr_contracts.yaml ← EURO STOXX 50 symbol mappings
├── agent/
│   ├── alerts.py           ← 18 technical signal detectors
│   ├── features.py         ← 74-feature vector builder (live)
│   ├── model.py            ← XGBoost loader/predictor
│   ├── strategy.py         ← FADE / FOLLOW / SKIP logic + crowding gate
│   ├── risk.py             ← Position sizing, commission model, daily limits
│   ├── executor.py         ← IBKR bracket orders + short safety block
│   ├── journal.py          ← SQLite trade log
│   ├── learner.py          ← Threshold recalibration + periodic retrain
│   ├── monitor.py          ← Exit sync, trailing SL, SL re-placement
│   └── data_feed.py        ← IBKR connection + OHLCV fetch
└── data/
    ├── model/              ← 6 trained model files (gitignored)
    ├── cache/              ← OHLCV parquet cache (gitignored)
    └── journal.db          ← Trade database (gitignored)
```

---

## Troubleshooting

**"Model not found"** → `python bootstrap_model.py --yfinance`

**"Feature mismatch"** → features changed; rerun bootstrap

**Accidental shorts** → `python close_shorts.py --shorts-only`

**Flatten everything** → `python close_shorts.py --emergency`

**Check logs** → `type data\agent.log` (Windows) or `tail -f data/agent.log`
