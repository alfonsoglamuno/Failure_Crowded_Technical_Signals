# Predicting the Failure of Crowded Technical Signals

## Core Question

> Given a visible chart alert (breakout, RSI extreme, MACD cross, abnormal volume) on a EURO STOXX 50 stock — and the surrounding market context — what is the probability that a trade on this alert will be **profitable** over the next 1-10 sessions?

The model does not predict price direction. It predicts **P(trade is profitable) conditioned on context**. A bearish signal that fires into a high-correlation, high-momentum, high-volume environment has very different profit odds than the same signal in an idiosyncratic, low-crowding environment.

**Tradeable consequence**: when the model assigns high probability of profit to a bearish alert, we enter a contrarian long (FADE → BUY). When a bullish alert scores high, we enter momentum (FOLLOW → BUY). Short entries require `allow_short=true` with a `_both` variant.

---

## Model Architecture

- **All models predict P(trade is profitable)** — never P(failure). This is always-profit mode.
- **8 variants**: h1d / h3d / h5d / h10d × longonly / both
- **Recommended default**: `h10d_longonly` — 10-day hold, long-only, gap_up alert family, Sharpe 0.89 on optimizer backtest
- **Optimizer output**: running `python optimize_strategy.py --yfinance` produces `strategy_recommendation.yaml` which auto-configures the agent for the best-performing alert+variant combination

---

## Hypothesis

Technical alerts are not uniformly informative. Their reliability depends on context.

Signals occurring under **crowded, attention-heavy, or exhausted** conditions are systematically more likely to fail and revert. This is consistent with the investor attention literature (Barber & Odean, 2008): when many participants act on the same visible signal, the trade is already priced in before it triggers, and the move fades.

The project operationalises this as a classification problem: train a model to predict, for each alert event, whether it belongs to the "failure" class based on measurable context features.

---

## Project Structure

```
.
├── configs/                  # Universe tickers, shared config
├── data/                     # Data storage (gitignored)
├── reports/                  # Model evaluation scripts and saved results
│   ├── evaluate_models.py    ← Run to evaluate all trained variants
│   └── results/              ← CSV outputs from evaluation runs
├── notebooks/                # Exploratory analysis
├── src/                      # Core research pipeline
│   ├── data/                 # Download and panel construction
│   ├── alerts/               # 18 technical signal detectors
│   ├── features/             # Feature engineering (80+ features, 9 blocks)
│   ├── models/               # Training, evaluation, calibration
│   └── backtest/             # Strategy simulation
└── trading_agent/            # Live IBKR agent (see trading_agent/README.md)
```

---

## Alert Families (18 signal types)

| Family | Signals |
|--------|---------|
| Trend | N-day high/low breakout, MA crossovers |
| Momentum | RSI overbought/oversold, MACD cross, Stochastic reversal |
| Volatility | Bollinger-band breakout, ATR spike, large candle |
| Volume / Attention | Abnormal volume spike, gap moves, extreme 1-day return |

Each alert fires with a direction (bullish / bearish / neutral). The model learns separately for each combination of alert type, direction, and context.

---

## Label Design

For each alert event on day `t`, the label is **P(trade is profitable)**:

- **Longonly mode** (all entries BUY): `label = 1` if `long_return > 0.2%` after commission
  - Bullish alert: `long_return = fwd_ret_{h}d` (FOLLOW momentum)
  - Bearish alert: `long_return = -fwd_ret_{h}d` (FADE contrarian BUY)
- **Both mode** (FADE entries, BUY + SELL): `label = 1` if FADE trade is profitable
  - Bullish alert FADE (SHORT): `label = 1` if `fwd_ret_{h}d < -0.2%`
  - Bearish alert FADE (BUY): `label = 1` if `fwd_ret_{h}d > +0.2%`

Where `h ∈ {1, 3, 5, 10}` days. Eight model variants cover all horizon × direction combinations (see below).

---

## Feature Blocks (9 blocks, 80+ features)

All features use only information available at end of day T. No forward-looking data.
See the leakage note at the end of this section.

### Block A — Alert properties
`direction_*`, `alert_name_*`, `n_simultaneous_alerts`

Alert type and direction are the primary inputs. Multiple concurrent alerts on the same ticker amplify the crowding signal (Barber & Odean, 2008).

### Block B — Short-term price state
`ret_1d/3d/5d/10d/20d`, `dist_ma10/20/50/100/200`, `price_pos_20d`

Distance from moving averages and recent momentum quantify overextension. A stock far above its 50d MA after a volume surge is the archetypal crowded long.

### Block C — Volatility and cost
`realvol_5d/10d/20d`, `atr_14`, `atr_norm_14`, `candle_range_norm`, `gap_size`, `atr_vs_commission`, `vol_regime_pct`

`atr_vs_commission`: ATR / round-trip cost (0.10%). The model learns to avoid tight, low-move stocks where commission consumes the entire predicted edge. `vol_regime_pct` captures whether the current volatility environment is historically elevated or compressed.

### Block D — Volume and attention
`vol_zscore_5d/20d`, `ret_vol_interaction`, `consec_up/down`

`ret_vol_interaction` = |return| x volume z-score: the crowding intensity signal. Consecutive up/down streaks attract retail attention and are historically prone to reversal.

### Block E — Market regime (EURO STOXX 50)
`index_ret_1d/5d/20d`, `index_vol_20d`, `index_above_ma50/200`, `index_corr_20d/60d`, `beta_20d/60d`, `rel_strength_1d/5d/20d`, `index_regime`

Market regime is the single strongest context variable. `index_above_ma200` distinguishes bear and bull markets; fade signals are more reliable in bear markets where retail "buy the dip" fails consistently. `rel_strength_1d` separates idiosyncratic moves (more likely to revert) from moves that just follow the index.

### Block F — Calendar and session
`dow`, `month`, `is_month_end/start`, `is_week_start/end`

Intraday and session-level trading behaviour is not stationary. Monday repositioning, Friday squaring, and month-end rebalancing create systematic flow patterns.

### Block G — Inter-stock peer correlation
`avg_peer_corr_20d`

Average 20-day pairwise correlation with all other EURO STOXX 50 stocks. When everything is correlated (herding regime), individual signals carry more systematic noise and fewer stock-specific information. The model discounts signals in high-herding environments.

### Block H — Intraday structure proxies (daily OHLCV)
`close_vs_range`, `open_to_close_ret`, `gap_pct`, `gap_is_filled`, `vwap_distance`, `vol_vs_dow_baseline`, `reversal_intrabar`

Closest approximations to microstructure signals without L2/tick data. `close_vs_range` = 0 means sellers dominated the session; 1 means buyers dominated. A bearish alert that closed near the daily high suggests residual buying — the fade signal is weaker.

### Block I — Trend and momentum (recency)
`ema_slope_5d/20d`, `rsi_14`, `rsi_trend_5d`, `macd_hist_norm`, `vol_accel`

Direction and rate-of-change of price dynamics, not just levels. A bearish alert into a falling RSI and negative MACD is more reliable than one into a rising RSI. `vol_accel` (short vol / long vol) flags regime transitions. Based on Jegadeesh & Titman (1993) and Lo & MacKinlay (1988).

### Leakage note

All features use only data available at the close of day T. Forward-looking columns (`ret_1d_lead`, `fwd_ret_{h}d`) are explicitly excluded from the training matrix. A full leakage audit found no contamination across all 9 blocks (see `reports/leakage_audit.md`).

---

## Model Variants

Eight models across two axes: horizon (1d / 3d / 5d / 10d) × direction mode (longonly / both).

| Variant | Horizon | Direction | Primary use case |
|---------|---------|-----------|-----------------|
| **`h10d_longonly`** | 10 days | Long-only | **Recommended default** — optimizer top pick (gap_up, Sharpe 0.89) |
| `h1d_longonly` | 1 day | Long-only | Intraday fades, EOD close, no shorts |
| `h3d_longonly` | 3 days | Long-only | Swing fades, 3-day holds, no shorts |
| `h5d_longonly` | 5 days | Long-only | Multi-day positions, no shorts |
| `h1d_both` | 1 day | Long + Short | Intraday, requires `allow_short=true` |
| `h3d_both` | 3 days | Long + Short | Swing, requires `allow_short=true` |
| `h5d_both` | 5 days | Long + Short | Position, requires `allow_short=true` |
| `h10d_both` | 10 days | Long + Short | Multi-week, requires `allow_short=true` |

**`longonly`**: trains on all directional alerts (bullish + bearish). All entries are BUY. Bullish alerts → FOLLOW (momentum), bearish alerts → FADE (contrarian BUY). Better calibrated for a long-only book.

**`both`**: trains on all alert directions, BUY + SELL entries. Model predicts P(FADE is profitable). Required when `allow_short=true`. Has more training data but noisier for pure long-only decisions.

**Recommended default is `h10d_longonly`** — optimizer-selected based on best risk-adjusted returns. Set via `model.variant` in `configs/config.yaml` or auto-configured by `optimize_strategy.py`.

### Correlation treatment by mode

In **longonly** mode, all positions are longs. High correlation between two open positions means doubled same-direction exposure → blocked at 0.85 for h1d, 0.70 for h3d, 0.65 for h5d.

In **both** mode, a LONG on stock A and SHORT on stock B that are highly correlated is actually a **hedged pair**: when A rises, B also rises (bad for the short), so the two P&Ls partially cancel. The correlation gate is direction-adjusted: `effective_corr = sign(candidate) x sign(open) x raw_corr`. Only same-direction concentrated bets are blocked.

---

## Validation Framework

- Strict **chronological** train/validation splits (no data shuffling)
- Walk-forward evaluation to test regime generalization
- Purging and embargo around adjacent events to prevent label-window leakage
- All features computed with information available at end of day T only
- Separate validation per horizon x mode variant

### Running evaluation

```bash
# Evaluate all variants (outputs to reports/results/)
python reports/evaluate_models.py

# One variant
python reports/evaluate_models.py --variant h1d_longonly

# Save to specific path
python reports/evaluate_models.py --output reports/results/my_run.csv
```

The script measures AUC, precision and recall at multiple thresholds, break-even analysis, and calibration error. Results accumulate in `reports/results/` as a historical record.

---

## Key Design Decisions

### Execution cost frames the strategy

A signal is only worth trading if the predicted edge survives execution. With a 2,000 EUR position:
- IBKR commission: ~2.00 EUR per side (0.05% x 2,000, min 2 EUR)
- Round-trip cost: ~4 EUR on a 2,000 EUR position = 0.20%
- SL = 1.5%, TP = 2.5% → break-even precision ≈ 40%

The `atr_vs_commission` feature and the `expected_gross < 2 x commission` guard in `risk.py` enforce this at the trade level.

### Attention signals are double-edged

A volume spike can mean crowd exhaustion (fade opportunity) or real information arrival (continuation). The model distinguishes these by combining attention features (`vol_zscore`, `n_simultaneous_alerts`) with regime context (`index_regime`, `index_above_ma200`) and relative move (`rel_strength_1d`). Volume alone is insufficient.

### Correlation is context-dependent

Cross-stock correlation changes meaning depending on hold duration and direction mode:
- Intraday (h1d): positions close EOD regardless — overlap risk is minimal. Softer gate (0.85).
- Swing/position (h3d/h5d): positions can overlap for multiple days. Tighter gate (0.70 / 0.65).
- Short-enabled (both): direction-adjusted. Correlated long+short pairs are partially hedged.

### Recency matters in training

Market microstructure and crowding behaviour shift over time. Training weights decay exponentially with a 252-day half-life (1 trading year), so recent data has more influence. The model retrains monthly, plus an early-trigger if live hit-rate drops more than 10pp from baseline.

---

## References

- Barber, B. M., & Odean, T. (2008). *All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors.* Review of Financial Studies.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Lo, A. W., & MacKinlay, A. C. (1988). *Stock Market Prices Do Not Follow Random Walks.* Review of Financial Studies.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance.
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- Andersen, T. G., & Bollerslev, T. (1998). *Answering the Critics: Yes, ARCH Models Do Provide Good Volatility Forecasts.* International Economic Review.
