# Predicting the Failure of Crowded Technical Signals for Short-Term Equity Reversals

## Overview

A machine learning framework to predict the short-term **failure** of popular technical alerts in the **EURO STOXX 50** universe. Rather than treating indicators as direct trade signals, this project models them as observable events whose reliability depends on surrounding market conditions.

**Core question:** Given a visible chart alert (breakout, RSI extreme, MACD cross, abnormal volume), and the current market context — what is the probability that this alert will *fade* rather than follow through over the next 1–5 sessions?

---

## Hypothesis

Technical alerts are not uniformly informative. Their short-term reliability depends on context. Highly visible signals occurring under crowded, attention-heavy, or exhausted market conditions are systematically more likely to fail and revert.

---

## Project Structure

```
.
├── configs/                  # Configuration files
│   └── config.yaml           # Main config (universe, horizons, thresholds)
├── data/                     # Data storage (gitignored)
│   ├── raw/                  # Original downloaded OHLCV data
│   ├── processed/            # Cleaned and aligned data
│   └── features/             # Engineered feature tables
├── notebooks/                # Exploratory and reporting notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_alert_engine.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_backtesting.ipynb
├── results/                  # Outputs (models, reports, plots)
│   ├── models/
│   └── reports/
├── src/                      # Core source code
│   ├── data/                 # Data download and preprocessing
│   ├── alerts/               # Alert detection engine
│   ├── features/             # Feature engineering
│   ├── models/               # Training, evaluation, calibration
│   └── backtest/             # Strategy simulation
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

---

## Alert Families

| Family      | Alerts |
|-------------|--------|
| **Trend**   | N-day high/low breakout, MA crossovers |
| **Momentum**| RSI overbought/oversold, MACD cross, Stochastic reversal |
| **Volatility** | Bollinger-band breakout, ATR spike, large candle |
| **Volume / Attention** | Abnormal volume spike, gap moves, extreme 1-day returns |

---

## Label Design

For each alert event on day `t`:
- **Bullish alert failure**: forward return `r(t+1 : t+h) < -θ`
- **Bearish alert failure**: forward return `r(t+1 : t+h) > +θ`

Where `h ∈ {1, 3, 5}` days and `θ` is a minimum economically meaningful threshold.

---

## Models

| Model | Purpose |
|-------|---------|
| Logistic Regression | Interpretability and calibration baseline |
| Random Forest | Nonlinear interaction baseline |
| XGBoost / LightGBM | Main production candidate |

---

## Validation

- Strict chronological train / validation / test splits
- Walk-forward (rolling or expanding window)
- Purging and embargo to prevent label overlap leakage
- No future information in features

---

## Benchmark Strategies

1. **Follow-the-alert** — trade in the signal's direction
2. **Blind inverse** — always fade the alert
3. **ML-filtered contrarian** — only fade when model confidence exceeds threshold

---

## Scope (Version 1)

| Parameter | Value |
|-----------|-------|
| Universe | EURO STOXX 50 constituents (daily) |
| Horizons | 1, 3, 5 days |
| Main model | XGBoost |
| Validation | Walk-forward with purging/embargo |
| Primary output | Ranked contrarian opportunity scores |

---

## Setup

```bash
git clone https://github.com/alfonsoglamuno/Failure_Crowded_Technical_Signals.git
cd Failure_Crowded_Technical_Signals
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## References

- Barber, B. M., & Odean, T. (2008). *All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors.* Review of Financial Studies.
- Prado, M. L. de (2018). *Advances in Financial Machine Learning.* Wiley.
