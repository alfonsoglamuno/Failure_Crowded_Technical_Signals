# Project Specification
## Predicting the Failure of Crowded Technical Signals — EURO STOXX 50

---

## 1. Folder Structure

```
Failure_Crowded_Technical_Signals/
│
├── configs/
│   ├── config.yaml                  # All parameters (universe, horizons, thresholds, model)
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
│   ├── 01_data_exploration.ipynb    # Data quality, coverage, missing values
│   ├── 02_alert_engine.ipynb        # Alert frequency, co-occurrence, base rates
│   ├── 03_feature_engineering.ipynb # Feature distributions, correlations, importance
│   ├── 04_modeling.ipynb            # Walk-forward training, OOS evaluation
│   └── 05_backtesting.ipynb         # Strategy comparison, net-of-cost results
│
├── results/
│   ├── models/                      # Serialized model artifacts (.joblib)
│   └── reports/                     # HTML/CSV evaluation summaries
│
├── src/
│   ├── data/
│   │   ├── download.py              # yfinance download + parquet caching
│   │   └── preprocess.py            # Cleaning, panel builder (long format)
│   ├── alerts/
│   │   └── engine.py                # 18 technical alerts across 4 families
│   ├── features/
│   │   ├── engineering.py           # 6 feature blocks (see §4)
│   │   └── labels.py                # Forward return computation + binary labels
│   ├── models/
│   │   ├── train.py                 # Walk-forward CV with purge/embargo
│   │   └── evaluate.py              # ML metrics + trading metrics
│   └── backtest/
│       └── strategy.py              # 3-way strategy comparison
│
├── tests/                           # Unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## 2. Dataset Schema

### 2.1 Raw OHLCV (`data/raw/<TICKER>.parquet`)

| Column  | Type    | Description                          |
|---------|---------|--------------------------------------|
| date    | DatetimeIndex | Trading date (UTC, daily)     |
| Open    | float64 | Adjusted open price                  |
| High    | float64 | Adjusted high price                  |
| Low     | float64 | Adjusted low price                   |
| Close   | float64 | Adjusted close price                 |
| Volume  | float64 | Share volume (adjusted for splits)   |

- Source: `yfinance`, `auto_adjust=True`
- Coverage: 2010-01-01 → present
- Universe: 50 EURO STOXX 50 constituents (see `configs/eurostoxx50_tickers.yaml`)
- Minimum history: 252 trading days (configurable)

### 2.2 Processed Panel (`data/processed/panel.parquet`)

Long format, one row per (ticker, date).

| Column  | Type     | Description                |
|---------|----------|----------------------------|
| date    | datetime | Trading date               |
| ticker  | str      | Yahoo Finance ticker       |
| open    | float64  | Adjusted open              |
| high    | float64  | Adjusted high              |
| low     | float64  | Adjusted low               |
| close   | float64  | Adjusted close             |
| volume  | float64  | Adjusted volume            |

### 2.3 Feature Table (`data/features/features.parquet`)

Panel + all engineered columns. See §4 for full feature list.

### 2.4 Labeled Events (`data/features/events_labeled.parquet`)

One row per (ticker, date, alert_name) triplet. Merges alert events with features and labels.

| Column             | Type     | Description                                         |
|--------------------|----------|-----------------------------------------------------|
| date               | datetime | Alert date                                          |
| ticker             | str      | Stock ticker                                        |
| alert_name         | str      | Alert identifier (e.g. `rsi_overbought`)            |
| direction          | str      | `bullish` / `bearish` / `neutral`                   |
| n_simultaneous_alerts | int   | Count of other alerts on same (date, ticker)        |
| fwd_ret_1d         | float64  | Forward 1-day return from close(t) to close(t+1)   |
| fwd_ret_3d         | float64  | Forward 3-day return                               |
| fwd_ret_5d         | float64  | Forward 5-day return                               |
| label_failure_1d   | float {0,1} | 1 = alert failed at 1-day horizon               |
| label_failure_3d   | float {0,1} | 1 = alert failed at 3-day horizon               |
| label_failure_5d   | float {0,1} | 1 = alert failed at 5-day horizon               |
| [all feature cols] | float64  | Feature block columns (see §4)                      |

---

## 3. Exact Label Definition

### Forward Return

```
fwd_ret_h(t) = close(t + h) / close(t) - 1
```

Computed per ticker, shifted backward by h so the value sits on the alert date row.

### Binary Failure Label

For an alert on day `t` with horizon `h` and threshold `θ = 0.005` (0.5%):

| Alert direction | Label = 1 (failure)         | Label = 0 (continuation) |
|-----------------|-----------------------------|--------------------------|
| bullish         | `fwd_ret_h < −θ`            | `fwd_ret_h ≥ −θ`         |
| bearish         | `fwd_ret_h > +θ`            | `fwd_ret_h ≤ +θ`         |
| neutral         | `|fwd_ret_h| > θ`           | `|fwd_ret_h| ≤ θ`        |

**Why 0.5%?** Captures economically meaningful reversals while filtering noise. Tune in sensitivity analysis.

### Base Rate (empirical)

From smoke test on 3 tickers (2018-2024), h=3: ~46% failure rate. Expect similar across full universe — roughly balanced, which is healthy for classification.

---

## 4. Feature List (Version 1)

### Block A — Alert Features
| Feature                  | Description                                          |
|--------------------------|------------------------------------------------------|
| `alert_name`             | Categorical identifier (one-hot encode for model)   |
| `direction`              | bullish / bearish / neutral                          |
| `n_simultaneous_alerts`  | Number of alerts active on same (date, ticker)       |

### Block B — Short-Term Price State
| Feature         | Description                                    |
|-----------------|------------------------------------------------|
| `ret_1d`        | 1-day return                                   |
| `ret_3d`        | 3-day return                                   |
| `ret_5d`        | 5-day return                                   |
| `ret_10d`       | 10-day return                                  |
| `ret_20d`       | 20-day return                                  |
| `dist_ma10`     | (close − MA10) / MA10                          |
| `dist_ma20`     | (close − MA20) / MA20                          |
| `dist_ma50`     | (close − MA50) / MA50                          |
| `dist_ma100`    | (close − MA100) / MA100                        |
| `dist_ma200`    | (close − MA200) / MA200                        |
| `price_pos_20d` | Position within 20-day high-low range [0, 1]   |

### Block C — Volatility State
| Feature          | Description                                       |
|------------------|---------------------------------------------------|
| `realvol_5d`     | 5-day annualised realised volatility              |
| `realvol_10d`    | 10-day annualised realised volatility             |
| `realvol_20d`    | 20-day annualised realised volatility             |
| `atr_14`         | 14-day Average True Range                         |
| `atr_norm_14`    | ATR / close (normalised)                          |
| `candle_range_norm` | (high − low) / close                          |
| `gap_size`       | |open / close(t-1) − 1|                          |
| `vol_regime_pct` | Percentile rank of current vol vs. 60-day window  |

### Block D — Volume & Crowding State
| Feature               | Description                                   |
|-----------------------|-----------------------------------------------|
| `vol_zscore_5d`       | Volume z-score vs. 5-day window               |
| `vol_zscore_20d`      | Volume z-score vs. 20-day window              |
| `ret_vol_interaction` | |return| x volume z-score (attention proxy)   |
| `consec_up`           | Consecutive up sessions before alert          |
| `consec_down`         | Consecutive down sessions before alert        |

### Block E — Regime Features
| Feature             | Description                                     |
|---------------------|-------------------------------------------------|
| `index_ret_5d`      | EURO STOXX 50 index 5-day return                |
| `index_vol_20d`     | Index 20-day annualised volatility              |
| `index_above_ma50`  | Binary: index > 50-day MA                      |

### Block F — Calendar Features
| Feature          | Description          |
|------------------|----------------------|
| `dow`            | Day of week (0-4)    |
| `month`          | Month (1-12)         |
| `is_month_end`   | Binary               |
| `is_month_start` | Binary               |

**Total: ~36 raw features** (before one-hot encoding of `alert_name` / `direction`)

---

## 5. Baseline Pipeline

### Step 1: Data Ingestion
```
download_ohlcv(tickers, start, end) → data/raw/<ticker>.parquet
build_panel(raw_data)               → data/processed/panel.parquet
```

### Step 2: Feature Engineering
```
build_features(panel, index_close)  → data/features/features.parquet
compute_forward_returns(panel, [1,3,5])
```

### Step 3: Alert Detection
```
run_alert_engine(panel)             → events DataFrame
add_alert_features(events, panel)   → enriched events
```

### Step 4: Label Construction
```
assign_labels(events, features, horizons=[1,3,5], theta=0.005)
→ data/features/events_labeled.parquet
```

### Step 5: Baseline Model (XGBoost, h=3)
```
X = events[feature_cols].fillna(0)
y = events["label_failure_3d"]
dates = events["date"]

results = train_evaluate(X, y, dates, model_name="xgboost",
    cfg=WalkForwardConfig(n_splits=5, purge_days=5, embargo_days=10))
```

### Step 6: Evaluation
```
ml_metrics(results["y_true"], results["y_pred_proba"])
→ ROC-AUC, PR-AUC, precision, recall, F1, top-decile precision
```

### Step 7: Strategy Backtest
```
events["failure_proba"] = results["y_pred_proba"]
compare_strategies(events, horizon=3, confidence_threshold=0.6)
→ follow_alert | blind_inverse | ml_filtered returns

strategy_metrics(ml_filtered_returns, cost_bps=10)
→ Sharpe, Sortino, hit rate, max drawdown
```

---

## 6. Validation Protocol

| Stage       | Method                                      |
|-------------|---------------------------------------------|
| Split       | Chronological: 70% train / 10% val / 20% test |
| CV scheme   | Walk-forward expanding window, 5 splits     |
| Purge       | Drop training samples within 5 days of test boundary |
| Embargo     | Additional 10-day gap after purge           |
| Feature leakage check | All features computed on `t` data only; forward returns shifted correctly |
| No lookahead | `pct_change`, `rolling`, `shift` all use only past data |

---

## 7. Evaluation Metrics

### ML Metrics
- ROC-AUC (primary ranking metric)
- PR-AUC (handles class imbalance)
- Top-decile precision (most actionable)
- F1, precision, recall at 0.5 threshold

### Trading Metrics (net of 10bps round-trip)
- Hit rate
- Mean trade return
- Sharpe ratio (annualised)
- Sortino ratio
- Maximum drawdown
- Total return

### Primary comparison
`ml_filtered` vs. `follow_alert` vs. `blind_inverse` — on identical test periods.

---

## 8. Key Parameters to Tune (Version 2)

| Parameter         | Default | Range to explore       |
|-------------------|---------|------------------------|
| `theta`           | 0.005   | 0.002 - 0.015          |
| `horizon`         | 3       | 1, 3, 5                |
| `confidence_threshold` | 0.60 | 0.55 - 0.75         |
| `purge_days`      | 5       | 3 - 10                 |
| `embargo_days`    | 10      | 5 - 20                 |
| `xgb max_depth`   | 4       | 3 - 6                  |
| `vol_spike_z_threshold` | 2.0 | 1.5 - 3.0           |
