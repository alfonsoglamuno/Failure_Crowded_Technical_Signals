# Predicting the Failure of Crowded Technical Signals for Short-Term Equity Reversals

## Overview

A machine learning framework to predict the short-term **failure** of popular technical alerts in the **EURO STOXX 50** universe. Rather than treating indicators as direct trade signals, this project models them as observable events whose reliability depends on surrounding market conditions.

**Core question:** Given a visible chart alert (breakout, RSI extreme, MACD cross, abnormal volume), and the current market context — what is the probability that this alert will *fade* rather than follow through over the next 1-5 sessions?

---

## Hypothesis

Technical alerts are not uniformly informative. Their short-term reliability depends on context. Highly visible signals occurring under crowded, attention-heavy, or exhausted market conditions are systematically more likely to fail and revert.

The framing is not: *"Will this stock go up in 5 minutes?"*

The framing is: *"Conditional on current liquidity, order-flow pressure, market regime, and relative move versus index/sector — is this move likely to continue or mean-revert enough to overcome costs?"*

---

## Project Structure

```
.
├── configs/                  # Configuration files
├── data/                     # Data storage (gitignored)
├── notebooks/                # Exploratory and reporting notebooks
├── src/                      # Core source code
│   ├── data/                 # Data download and preprocessing
│   ├── alerts/               # Alert detection engine (18 signal types)
│   ├── features/             # Feature engineering (74 features, 8 blocks)
│   ├── models/               # Training, evaluation, calibration
│   └── backtest/             # Strategy simulation
├── trading_agent/            # Live trading agent (IBKR)
│   └── README.md             ← agent-specific documentation
├── tests/
├── requirements.txt
└── README.md
```

---

## Alert Families

| Family | Alerts |
|--------|--------|
| **Trend** | N-day high/low breakout, MA crossovers |
| **Momentum** | RSI overbought/oversold, MACD cross, Stochastic reversal |
| **Volatility** | Bollinger-band breakout, ATR spike, large candle |
| **Volume / Attention** | Abnormal volume spike, gap moves, extreme 1-day returns |

---

## Label Design

For each alert event on day `t`:
- **Bullish alert failure**: forward return `r(t+1 : t+h) < -θ`
- **Bearish alert failure**: forward return `r(t+1 : t+h) > +θ`

Where `h ∈ {1, 3, 5}` days and `θ = 0.5%` minimum economically meaningful return.
Six model variants (see Model Variants section) cover all horizon × direction combinations.

---

## Feature Design Philosophy

This project follows a **tiered feature architecture** derived from intraday market microstructure research:

> "For intraday trading, the most useful features are microstructure, liquidity, and market-state features — not just classic daily-chart indicators."

### Tier 1 — Feasible now (daily OHLCV + market data)

These are computable from standard OHLCV bars and index data:

| Feature group | Examples | Block |
|---------------|----------|-------|
| Short-term price state | 1m/5m/15m returns proxy (daily lag returns), MA distance | B |
| Volatility state | Realized vol, ATR, candle range, gap size | C |
| Commission awareness | ATR vs round-trip cost — is the trade worth entering? | C |
| Volume / attention | Volume z-score, consec up/down, vol × return interaction | D |
| Market regime | Index trend, beta, correlation, relative strength | E |
| Calendar / session | Day-of-week, month-end, week-start/end (session regime) | F |
| Intraday structure | Close-vs-range, open-to-close direction, VWAP distance, gap fill | H |
| Peer correlation | Average 20d pairwise correlation with all EURO STOXX 50 peers | G |

### Tier 2 — Best if L2/tick data is available

These would be added if IBKR or a data vendor provides intraday bars or order-book snapshots:

| Feature | Why it matters |
|---------|---------------|
| Order-book imbalance (bid vs ask qty) | Strongest intraday predictor; real-time supply/demand balance |
| Aggressor-side volume ratio | Buy-initiated vs sell-initiated volume signals order flow |
| Cancel-to-add ratio | Rising cancellations = liquidity deterioration before a move |
| 1-min / 5-min / 15-min returns | True short-horizon momentum; much more predictive than daily |
| True intraday VWAP | Computed from 1-min (V×P) / cumulative V — institutional benchmark |
| Volume vs same-minute-of-day | True "relative volume now vs typical" baseline for intraday |
| Opening-range breakout / failure | First 30-min high-low: critical session-structure feature |
| Small-order intensity | Retail attention proxy without needing search/news data |

---

## Feature Blocks (74 features)

### Block A — Alert properties
`direction_*`, `alert_name_*`, `n_simultaneous_alerts`

Alert type, direction (bullish/bearish/neutral), and simultaneous alert count.
Multiple concurrent signals amplify crowding: more screens fire → more retail attention → higher fade probability (Barber & Odean, 2008).

### Block B — Short-term price state
`ret_1d/3d/5d/10d/20d`, `dist_ma10/20/50/100/200`, `price_pos_20d`

How overextended is the stock? MA distance and momentum quantify "crowdedness." A stock far above its 50d MA after a volume surge is a classic crowded long.

### Block C — Volatility and cost awareness
`realvol_5d/10d/20d`, `atr_14`, `atr_norm_14`, `candle_range_norm`, `gap_size`, `atr_vs_commission`, `vol_regime_pct`

- `atr_vs_commission` = ATR / round-trip commission cost (0.10%). When the stock barely moves relative to what the trade costs, the model should skip it.
- `vol_regime_pct` = volatility percentile over 60 sessions. High vol regimes change fade reliability dramatically.

### Block D — Volume and attention state
`vol_zscore_5d/20d`, `ret_vol_interaction`, `consec_up/down`

- `ret_vol_interaction` = |return| × volume z-score. The crowding intensity signal: a large move on elevated volume is exactly the retail-attention scenario.
- `consec_up/down` = streak of consecutive positive/negative days. Streaks attract attention and are prone to reversal.

### Block E — Market regime (EURO STOXX 50)
`index_ret_1d/5d/20d`, `index_vol_20d`, `index_above_ma50/200`,
`index_corr_20d/60d`, `beta_20d/60d`, `rel_strength_1d/5d/20d`, `index_regime`

Market regime is the single most important context variable:
- `index_above_ma200` — bear vs bull market. Fade signals are more reliable in bear markets where retail "buy the dip" fails more consistently.
- `index_regime` — +1 bull / 0 neutral / -1 bear (20d return threshold ±2%)
- `rel_strength_1d` — 1-day stock return minus index return. An idiosyncratic move is more likely to mean-revert; a move that just follows the tape may continue.
- `beta_20d/60d` — high-beta stocks amplify index moves; low-beta behave more independently. The model uses this to determine whether a signal is regime-driven or stock-specific.

### Block F — Calendar and session timing
`dow`, `month`, `is_month_end/start`, `is_week_start/end`

Intraday behavior is not stationary. Monday openings, Friday squaring, month-end rebalancing, and sector rotation events create predictable flow patterns. These are mandatory for any intraday model.

### Block G — Inter-stock peer correlation
`avg_peer_corr_20d`

Average 20-day pairwise return correlation with all other EURO STOXX 50 stocks.
- High value → stocks are moving together (herding/risk-on-off regime)
- Low value → idiosyncratic trading → signal is stock-specific and cleaner

This is the cross-sectional crowding feature: when everything is correlated, any individual signal has more systematic noise in it.

### Block H — Intraday structure proxies (daily OHLCV)
`close_vs_range`, `open_to_close_ret`, `gap_pct`, `gap_is_filled`, `vwap_distance`, `vol_vs_dow_baseline`, `reversal_intrabar`

These approximate microstructure signals without L2 data:

| Feature | Microstructure meaning |
|---------|----------------------|
| `close_vs_range` | 0=closed at low (sellers won), 1=closed at high (buyers won) |
| `open_to_close_ret` | Signed intraday order-flow direction |
| `gap_pct` | Overnight gap — informed flow or retail panic? |
| `gap_is_filled` | Did price trade back through the gap? Strong reversal confirmation |
| `vwap_distance` | Close vs (H+L+2C)/4 — standard institutional VWAP proxy |
| `vol_vs_dow_baseline` | Volume vs 8-week same-day-of-week rolling mean (session-normalized) |
| `reversal_intrabar` | Gap-down recovery fraction — buying absorption proxy |

**Important note on leakage:** All Block H features use only information available at the daily close (or during the day for same-day decisions). VWAP and range use only that day's OHLCV, not end-of-day session totals. Always ensure the bar is closed before using these features at any horizon.

---

## Model Variants

Six models are trained across two axes:

| Axis | Values | Meaning |
|------|--------|---------|
| **Horizon** | 1d, 3d, 5d | Forward return window for labels |
| **Mode** | `longonly`, `both` | Which alert directions are included |

```
h1d_longonly  h3d_longonly*  h5d_longonly
h1d_both      h3d_both       h5d_both
```
`*` = default active variant

**`longonly`**: trains only on bearish/neutral alert events — optimised for FADE→BUY entries.
Produces a better-calibrated model for a long-only book because training examples match the actual trades taken.

**`both`**: trains on all events including bullish alerts. Required when `allow_short=true`
because the model must also score SELL-side signal failures. Has more training data but
is noisier for long-only decisions.

**Choosing a variant:**
- Conservative paper trading → `h3d_longonly` (default)
- Shorter holds / intraday → `h1d_longonly`
- Longer multi-day holds → `h5d_longonly` or `h5d_both`
- Full long/short book → `h3d_both` or `h5d_both`

---

## Validation Framework

- Strict **chronological** train / validation / test splits (no shuffling)
- **Walk-forward** evaluation (rolling or expanding window)
- **Purging and embargo** to prevent label overlap leakage between adjacent events
- All features use only **past information** relative to the prediction timestamp
- Separate validation for each horizon × mode variant

---

## Key Design Decisions and Warnings

### Leakage is the biggest trap for intraday models

Common sources of lookahead in intraday feature engineering:
- Using today's high/low before the bar has closed
- Normalizing today's volume by end-of-day total (not available intraday)
- Using the full-session VWAP when predicting mid-session
- Regime features that depend on closing values used for intraday entry

This project uses daily bars and predicts at end-of-day, which avoids intraday leakage.
If moving to 1-min or 5-min bars, audit every feature for look-ahead with respect to
the prediction timestamp.

### Execution cost frames the strategy

A signal is only valuable if it survives execution:
- Spread and market impact absorb small predicted edges
- Commission must be recoverable within the predicted move
- The `atr_vs_commission` feature explicitly models this trade-off

The `risk.py` guard (`expected_gross < 2 × commission`) enforces it at the trade level.

### Attention spikes are double-edged

Volume and news spikes can mean:
- Crowd exhaustion → fade opportunity, OR
- Real information → continuation

The model combines attention (`vol_zscore`, `n_simultaneous_alerts`) with market regime
(`index_regime`, `index_above_ma200`) and relative move (`rel_strength_1d`) to
distinguish these cases. Volume alone is insufficient.

### Correlation is unstable intraday

Cross-stock correlation (`avg_peer_corr_20d`) is used as a context feature and a
portfolio-level gate (skip new trades that are >0.70 correlated with open positions).
However, intraday correlation strengthens near the close — use shorter windows for
intraday decisions and treat correlation as a risk-control input, not standalone alpha.

---

## Performance (Current Models)

| Variant | AUC | Precision@0.60 | Precision@0.70 |
|---------|-----|----------------|----------------|
| h1d_longonly | 0.766 | 78.2% | 83.0% |
| h3d_longonly | 0.760 | 75.0% | 83.4% |
| h5d_longonly | 0.758 | — | — |
| h1d_both | 0.672 | 70.4% | 80.0% |
| h3d_both | 0.696 | 70.5% | 81.6% |
| h5d_both | 0.717 | 79.5% | 89.3% |

Break-even hit rate with 2,000 EUR position at IBKR: ~37.5%.
Active variant (h3d_longonly) delivers **75.0%** at threshold 0.60.

---

## References

- Barber, B. M., & Odean, T. (2008). *All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors.* Review of Financial Studies.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance.
- Gould, M. D., et al. (2013). *Limit Order Books.* Quantitative Finance.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- Andersen, T. G., & Bollerslev, T. (1998). *Answering the Critics: Yes, ARCH Models Do Provide Good Volatility Forecasts.* International Economic Review.
