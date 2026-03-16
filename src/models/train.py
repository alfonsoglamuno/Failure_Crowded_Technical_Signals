"""
Model training with walk-forward cross-validation and purging/embargo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    purge_days: int = 5
    embargo_days: int = 10
    test_fraction: float = 0.2


def get_model(name: str, random_state: int = 42, scale_pos_weight: float = 1.0, **kwargs) -> Any:
    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=random_state,
                class_weight="balanced", **kwargs
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1,
            class_weight="balanced", **kwargs
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            **kwargs,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models)}")
    return models[name]


def make_walk_forward_splits(
    dates: pd.Series,
    cfg: WalkForwardConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, test_idx) pairs with purge and embargo.

    Purge: drop training samples whose label horizon overlaps with test period.
    Embargo: gap between end of training set and start of test set.
    """
    sorted_dates = dates.sort_values().values
    n = len(sorted_dates)
    test_size = int(n * cfg.test_fraction)
    step = test_size

    splits = []
    for split_i in range(cfg.n_splits):
        test_end_idx = n - split_i * step
        test_start_idx = test_end_idx - test_size

        if test_start_idx <= 0:
            break

        test_start_date = sorted_dates[test_start_idx]
        test_end_date = sorted_dates[test_end_idx - 1]

        # Purge: remove training samples whose label horizon bleeds into test period.
        # embargo_days acts as a calendar gap; purge_days should equal the label horizon.
        cutoff_date = test_start_date - np.timedelta64(cfg.embargo_days + cfg.purge_days, "D")
        train_mask = dates < cutoff_date
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[::-1]  # chronological order


def train_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_name: str = "xgboost",
    cfg: WalkForwardConfig | None = None,
) -> dict:
    """
    Run walk-forward cross-validation and return OOS predictions + metrics.
    """
    if cfg is None:
        cfg = WalkForwardConfig()

    splits = make_walk_forward_splits(dates, cfg)
    logger.info("Walk-forward splits: %d", len(splits))

    all_preds = []
    all_true = []
    all_dates = []
    all_orig_idx = []   # original DataFrame row indices for precise backtest mapping
    all_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        d_test = dates.iloc[test_idx]

        # Drop NaN labels
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test, y_test = X_test[valid_test], y_test[valid_test]
        d_test = d_test[valid_test]

        if len(y_train) == 0 or len(y_test) == 0:
            logger.warning("Fold %d: empty split, skipping.", fold_i)
            continue

        # Scale pos weight to handle class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        model = get_model(model_name, scale_pos_weight=spw)
        model.fit(X_train.fillna(0), y_train)

        proba = model.predict_proba(X_test.fillna(0))[:, 1]
        all_preds.extend(proba)
        all_true.extend(y_test.values)
        all_dates.extend(d_test.values)
        all_orig_idx.extend(X_test.index.tolist())   # preserve original row index
        logger.info("Fold %d: train=%d, test=%d  spw=%.2f", fold_i, len(y_train), len(y_test), spw)

        # Collect feature importances (XGBoost and RF only)
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "feature_importances_"):
            all_importances.append(
                pd.Series(clf.feature_importances_, index=X_train.columns)
            )

    return {
        "dates": all_dates,
        "y_true": all_true,
        "y_pred_proba": all_preds,
        "orig_idx": all_orig_idx,
        "feature_importance": (
            pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
            if all_importances else None
        ),
    }
