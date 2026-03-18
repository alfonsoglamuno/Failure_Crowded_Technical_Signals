# Project Specification
## Predicting the Failure of Crowded Technical Signals — EURO STOXX 50

---

## 1. Folder Structure

```
Failure_Crowded_Technical_Signals/
│
├── configs/
│   ├── config.yaml                  # Shared parameters (universe, horizons, thresholds)
│   └── eurostoxx50_tickers.yaml     # 50 constituent Yahoo Finance tickers
│
├── data/                            # Gitignored — never committed
│   ├── raw/                         # Downloaded OHLCV parquet files, one per ticker
│   ├── processed/                   # Cleaned panel: panel.parquet (long format)
│   └── features/                    # Feature matrix + labeled events
│       ├── features.parquet
│       └── events_labeled.parquet
│
├── docs/
│   └── project_specification.md     # This document
│
├── notebooks/
│   └── 01_results_overview.ipynb    # End-to-end evaluation: alerts, models, calibration
│
├── reports/
│   ├── evaluate_models.py           # Standalone evaluation script for all 6 variants
│   └── results/                     # CSV evaluation outputs (timestamped)
│
├── src/
│   ├── data/
│   │   ├── download.py              # yfinance download + parquet caching
│   │   └── preprocess.py            # Cleaning, panel builder (long format)
│   ├── alerts/
│   │   └── engine.py                # 18 technical alerts across 4 families
│   ├── features/
│   │   ├── engineering.py           # 9 feature blocks, 80+ features (see §4)
│   │   └── labels.py                # Forward return computation + binary labels
│   ├── models/
│   │   ├── train.py                 # Walk-forward CV with purge/embargo
│   │   └── evaluate.py              # ML metrics + trading metrics
│   └── backtest/
│       └── strategy.py              # Strategy simulation
│
├── trading_agent/                   # Live IBKR agent (see trading_agent/README.md)
│   ├── run_agent.py                 # Main agent loop
│   ├── bootstrap_model.py           # Train all 6 model variants
│   ├── configs/
│   │   ├── config.yaml              # Agent settings (variant, risk, IBKR)
│   │   └── ibkr_contracts.yaml      # EURO STOXX 50 IBKR symbol mappings
│   └── agent/
│       ├── alerts.py / features.py / model.py / strategy.py
│       ├── risk.py / executor.py / journal.py / learner.py
│       ├── monitor.py / data_feed.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

## 2. Dataset Schema

### 2.1 Raw OHLCV (`data/raw/<TICKER>.parquet`)

| Column  | Type          | Description                        |
|---------|---------------|------------------------------------|
| date    | DatetimeIndex | Trading date (UTC, daily)          |
| open    | float64       | Adjusted open price                |
| high    | float64       | Adjusted high price                |
| low     | float64       | Adjusted low price                 |
| close   | float64       | Adjusted close price               |
| volume  | float64       | Share volume (adjusted for splits) |

- Source: `yfinance`, `auto_adjust=True`
- Coverage: 2010-01-01 → present
- Universe: 50 EURO STOXX 50 constituents (see `configs/eurostoxx50_tickers.yaml`)
- Minimum history: 252 trading days (1 year) before training begins

### 2.2 Processed Panel (`data/processed/panel.parquet`)

Long format, one row per (ticker, date).

### 2.3 Feature Table (`data/features/features.parquet`)

Panel augmented with all engineered columns from the 9 feature blocks (see §4).

### 2.4 Labeled Events (`data/features/events_labeled.parquet`)

One row per (ticker, date, alert_name) triplet. Merges alert events with features and labels.

| Column                | Type           | Description                                         |
|-----------------------|----------------|-----------------------------------------------------|
| date                  | datetime       | Alert date                                          |
| ticker                | str            | Stock ticker                                        |
| alert_name            | str            | Alert identifier (e.g. `rsi_overbought`)            |
| direction             | str            | `bullish` / `bearish` / `neutral`                   |
| n_simultaneous_alerts | int            | Count of alerts on same (date, ticker)              |
| fwd_ret_1d            | float64        | Forward 1-day return, close(t) to close(t+1)        |
| fwd_ret_3d            | float64        | Forward 3-day return                                |
| fwd_ret_5d            | float64        | Forward 5-day return                                |
| label_failure_1d      | float {0,1}    | 1 = alert failed at 1-day horizon                   |
| label_failure_3d      | float {0,1}    | 1 = alert failed at 3-day horizon                   |
| label_failure_5d      | float {0,1}    | 1 = alert failed at 5-day horizon                   |
| [feature columns]     | float64        | All 80+ features from the 9 blocks below            |

---

## 3. Label Definition

### Forward Return

```
fwd_ret_h(t) = close(t + h) / close(t) - 1
```

### Binary Failure Label

For an alert on day `t` with horizon `h` and threshold `θ = 0.005` (0.5%):

| Alert direction | Label = 1 (failure)       | Label = 0 (continuation) |
|-----------------|---------------------------|--------------------------|
| bullish         | `fwd_ret_h < −θ`          | `fwd_ret_h >= −θ`        |
| bearish         | `fwd_ret_h > +θ`          | `fwd_ret_h <= +θ`        |
| neutral         | `|fwd_ret_h| > θ`         | `|fwd_ret_h| <= θ`       |

**Why 0.5%?** Captures economically meaningful reversals after commission (0.10% round-trip), while filtering pure noise moves.

**Base rate**: ~37-47% failure rate depending on horizon. The model does not need a 50%+ base rate — it only needs to identify a subset where failure probability is meaningfully above base rate.

---

## 4. Feature Blocks (9 blocks, 80+ features)

All features use only information available at the end of day T. Forward-looking columns (`fwd_ret_*`, `ret_1d_lead`) are explicitly excluded from the training matrix.

### Block A — Alert Properties
| Feature                  | Description                                          |
|--------------------------|------------------------------------------------------|
| `direction_*`            | One-hot: bullish / bearish / neutral                |
| `alert_name_*`           | One-hot: alert type (18 signal types)               |
| `n_simultaneous_alerts`  | Count of other alerts on same (date, ticker)        |

Multiple concurrent alerts amplify the crowding signal (Barber & Odean, 2008).

### Block B — Short-Term Price State
| Feature         | Description                                    |
|-----------------|------------------------------------------------|
| `ret_1d/3d/5d/10d/20d` | Recent returns at multiple lookbacks  |
| `dist_ma10/20/50/100/200` | Distance from moving averages        |
| `price_pos_20d` | Position within 20-day high-low range [0, 1]  |

### Block C — Volatility and Cost
| Feature              | Description                                         |
|----------------------|-----------------------------------------------------|
| `realvol_5d/10d/20d` | Annualised realised volatility                      |
| `atr_14`             | 14-day Average True Range                           |
| `atr_norm_14`        | ATR / close                                         |
| `candle_range_norm`  | (high - low) / close                               |
| `gap_size`           | Overnight gap magnitude                             |
| `atr_vs_commission`  | ATR / round-trip cost (0.10%) — tradability gate   |
| `vol_regime_pct`     | Current vol percentile vs 60-day window            |

`atr_vs_commission` captures whether a stock's typical daily move is large enough to survive commission. The agent also enforces `expected_gross < 2 * commission` as a hard gate in `risk.py`.

### Block D — Volume and Attention
| Feature               | Description                                     |
|-----------------------|-------------------------------------------------|
| `vol_zscore_5d/20d`   | Volume z-score vs recent baseline               |
| `ret_vol_interaction` | `|return| x volume z-score` — crowding proxy    |
| `consec_up/down`      | Consecutive directional sessions before alert   |

`ret_vol_interaction` is the core crowding-intensity signal: high return + high volume = classic attention-driven exhaustion event.

### Block E — Market Regime (EURO STOXX 50)
| Feature                     | Description                                  |
|-----------------------------|----------------------------------------------|
| `index_ret_1d/5d/20d`       | Index momentum at multiple horizons          |
| `index_vol_20d`             | Index volatility (systemic stress proxy)     |
| `index_above_ma50/200`      | Binary bull/bear regime flags               |
| `index_corr_20d/60d`        | Index-stock rolling correlation              |
| `beta_20d/60d`              | Rolling beta vs index                        |
| `rel_strength_1d/5d/20d`    | Stock return minus index return              |
| `index_regime`              | Composite regime score                       |

`index_above_ma200` is the strongest single context variable: fade signals are more reliable in bear markets where retail "buy the dip" fails. `rel_strength_1d` separates idiosyncratic moves from index-driven moves.

### Block F — Calendar and Session
| Feature          | Description                        |
|------------------|------------------------------------|
| `dow`            | Day of week (0=Mon, 4=Fri)        |
| `month`          | Month (1-12)                       |
| `is_month_end`   | Month-end rebalancing flag        |
| `is_month_start` | Month-start repositioning flag    |
| `is_week_start/end` | Monday/Friday session flags    |

### Block G — Inter-Stock Peer Correlation
| Feature           | Description                                          |
|-------------------|------------------------------------------------------|
| `avg_peer_corr_20d` | Average 20d pairwise correlation with STOXX50 peers |

High peer correlation = herding regime. Signals in herding regimes carry more systematic noise and fewer stock-specific information content.

### Block H — Intraday Structure Proxies (daily OHLCV)
| Feature             | Description                                           |
|---------------------|-------------------------------------------------------|
| `close_vs_range`    | (close - low) / (high - low): session buyer strength |
| `open_to_close_ret` | Intraday return, open to close                       |
| `gap_pct`           | Open gap vs prior close                              |
| `gap_is_filled`     | Whether the gap was filled intraday                  |
| `vwap_distance`     | Close vs estimated VWAP                              |
| `vol_vs_dow_baseline` | Volume relative to same-weekday baseline          |
| `reversal_intrabar` | Intraday reversal signal                             |

`close_vs_range = 1` means buyers dominated; a bearish alert that closed near the daily high suggests residual buying — the fade signal is weaker.

### Block I — Trend and Momentum (Recency)
| Feature          | Description                                          |
|------------------|------------------------------------------------------|
| `ema_slope_5d`   | EMA10 rate of change over 5 days                    |
| `ema_slope_20d`  | EMA50 rate of change over 20 days                   |
| `rsi_14`         | 14-day RSI level                                    |
| `rsi_trend_5d`   | RSI change over 5 days                              |
| `macd_hist_norm` | MACD histogram normalized by close                  |
| `vol_accel`      | Short-vol / long-vol ratio (regime transition flag) |

Direction and rate-of-change of momentum, not just levels. A bearish alert into a falling RSI and negative MACD is more reliable than one into a rising RSI (Jegadeesh & Titman, 1993).

---

## 5. Model Variants

Six models covering all combinations of hold horizon × direction mode.

| Variant          | Horizon | Mode      | Correlation Gate | Training set          |
|------------------|---------|-----------|------------------|-----------------------|
| `h1d_longonly`   | 1 day   | Long-only | 0.85             | Bearish/neutral alerts only |
| `h3d_longonly`   | 3 days  | Long-only | 0.70             | Bearish/neutral alerts only |
| `h5d_longonly`   | 5 days  | Long-only | 0.65             | Bearish/neutral alerts only |
| `h1d_both`       | 1 day   | Long+Short | 0.85 (dir-adj)  | All alert directions  |
| `h3d_both`       | 3 days  | Long+Short | 0.70 (dir-adj)  | All alert directions  |
| `h5d_both`       | 5 days  | Long+Short | 0.65 (dir-adj)  | All alert directions  |

**Default: `h1d_longonly`** — intraday holds, EOD close at 16:30 CET, no short-selling risk.

### Longonly vs Both

- **`longonly`**: trained only on bearish/neutral events. Training examples exactly match the FADE→BUY trades that will be taken. Better calibrated for a long-only book.
- **`both`**: trained on all directions. Required when `allow_short=true` because the model must also score bullish-alert failure (FADE→SELL). More data, but noisier for pure long-only decisions.

### Correlation Gate

Before placing any trade, the agent checks the 20-day return correlation between the candidate and all open positions.

**Longonly mode**: all positions are longs, so any two correlated positions are additive exposure. Gate uses raw `|corr|`.

**Both mode** (direction-adjusted):
```
effective_corr = sign(candidate_direction) x sign(open_direction) x raw_corr
```
- Same direction + high corr → concentrated → **blocked**
- Opposite directions + high corr → hedged pair → **allowed**

Gate is stricter for longer holds (h3d/h5d) because positions can overlap for multiple days.

---

## 6. Label Design

```
fwd_ret_h(t) = close(t+h) / close(t) - 1

Bearish alert failure: fwd_ret > +theta  (price went up — expected move did not happen)
Bullish alert failure: fwd_ret < -theta  (price went down — expected continuation failed)
```

`theta = 0.005` (0.5%) is the minimum economically meaningful move after 0.10% round-trip commission. Six model variants cover all `h in {1, 3, 5}` × `{longonly, both}` combinations.

---

## 7. Training Pipeline

### Step 1: Data Ingestion
```
download_ohlcv(tickers, start, end)  →  data/raw/<ticker>.parquet
build_panel(raw_data)                →  data/processed/panel.parquet
```

### Step 2: Feature Engineering (9 blocks)
```
build_features(panel, index_close)   →  data/features/features.parquet
compute_forward_returns(panel, [1,3,5])
```

### Step 3: Alert Detection
```
run_alert_engine(panel)              →  events DataFrame (18 signal types)
add_alert_features(events, panel)    →  n_simultaneous_alerts, direction
```

### Step 4: Label Construction
```
assign_labels(events, features, horizons=[1,3,5], theta=0.005)
→  data/features/events_labeled.parquet
```

### Step 5: Model Training (XGBoost, per variant)
```
For each variant (horizon, mode):
    Filter events by mode (longonly → bearish/neutral only)
    X = feature_cols (no forward-looking columns)
    y = label_failure_{horizon}d
    Compute exponential sample weights (252-day half-life)
    Chronological train/validation split
    model.fit(X_train, y_train, sample_weight=w_train)
    Save model + feature_cols to data/model/
```

Trained via `trading_agent/bootstrap_model.py --yfinance`.

### Step 6: Evaluation
```
python reports/evaluate_models.py
→  reports/results/<timestamp>.csv
   AUC, precision@{0.50,0.55,0.60,0.65,0.70}, recall, calibration MAE
```

---

## 8. Validation Protocol

| Stage            | Method                                                          |
|------------------|-----------------------------------------------------------------|
| Split            | Chronological: 80% train / 20% validation (no shuffling)       |
| CV scheme        | Walk-forward expanding window                                   |
| Purge            | Drop training samples within 5 days of validation boundary     |
| Embargo          | 10-day gap after purge to prevent label-window leakage          |
| Leakage check    | All 9 feature blocks computed on day T data only               |
| Sample weighting | Exponential decay, 252-day half-life (recent data weighted more)|

---

## 9. Evaluation Metrics

### ML Metrics
- ROC-AUC (primary discrimination metric)
- Precision at thresholds 0.50, 0.55, 0.60, 0.65, 0.70
- Recall at same thresholds
- Calibration MAE (mean |predicted - actual| per probability bin)

### Break-Even Analysis
- Commission: 0.05% per side (min 2 EUR), round-trip = 0.10%
- SL = 1.5%,  TP = 2.5%
- Break-even precision: `SL / (SL + TP)` = 1.5 / 4.0 = **37.5%** gross; ~**40%** after commission
- Any threshold where measured precision >= 40% is profitable in expectation

---

## 10. Adaptive Learning

### Fast loop (every 10 completed trades)
Recalibrates `fade_threshold` based on recent win/loss rate:
- Win rate < 50% → raise threshold (more selective)
- Win rate > 65% → lower threshold (capture more signals)

Steps: +0.02 on losses, −0.01 on wins (asymmetric: easier to lose edge than gain it back).

### Slow loop (monthly + performance trigger)
Full XGBoost retrain on current universe data. Triggers when:
1. Model file is more than 30 days old (monthly schedule), **or**
2. Live hit-rate drops more than 10pp below the baseline recorded at last retrain

Training uses exponential recency weighting (252-day half-life) so recent market structure has more influence without discarding historical context.

---

## 11. Commission Model

IBKR tiered pricing: **0.05% of trade value, minimum 2 EUR per order**.

| Trade value | Commission (per side) | Round-trip |
|-------------|----------------------|------------|
| 500 EUR     | 2.00 EUR (minimum)   | 4.00 EUR   |
| 2,000 EUR   | 2.00 EUR (minimum)   | 4.00 EUR   |
| 5,000 EUR   | 2.50 EUR             | 5.00 EUR   |
| 10,000 EUR  | 5.00 EUR             | 10.00 EUR  |

The `atr_vs_commission` feature (ATR / 0.10%) explicitly models tradability. The `expected_gross < 2 * commission` guard in `risk.py` rejects trades where the statistical edge is too thin even before sizing.

---

## 12. Key Parameters

| Parameter                    | Value  | Notes                                               |
|------------------------------|--------|-----------------------------------------------------|
| `theta` (label threshold)    | 0.5%   | Minimum economically meaningful reversal            |
| `horizons`                   | 1,3,5  | Days; 6 variants cover all combinations            |
| `fade_threshold`             | 0.60   | Default — recalibrated by adaptive learner          |
| `follow_threshold`           | 0.40   | Below this → FOLLOW (disabled by default)           |
| `stop_loss_pct`              | 1.5%   | ~1 STOXX50 daily ATR                               |
| `take_profit_pct`            | 2.5%   | ~1.7:1 reward/risk ratio                           |
| `max_position_pct`           | 2%     | ~2,000 EUR per trade on 100K capital               |
| `max_open_positions`         | 10     | Slot cap                                            |
| `max_daily_loss_eur`         | 300    | 0.3% of capital — stops trading for the day        |
| `retrain_frequency_days`     | 30     | Monthly full retrain                                |
| `retrain_perf_trigger_pct`   | 0.10   | Also retrain if live hit-rate drops 10pp            |
| `sample_weight_halflife_days`| 252    | 1-year exponential decay on training samples        |
| `purge_days`                 | 5      | Days purged around train/val boundary               |
| `embargo_days`               | 10     | Additional gap after purge                          |

---

## 13. References

- Barber, B. M., & Odean, T. (2008). *All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors.* Review of Financial Studies.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Lo, A. W., & MacKinlay, A. C. (1988). *Stock Market Prices Do Not Follow Random Walks.* Review of Financial Studies.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance.
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
- Andersen, T. G., & Bollerslev, T. (1998). *Answering the Critics: Yes, ARCH Models Do Provide Good Volatility Forecasts.* International Economic Review.
