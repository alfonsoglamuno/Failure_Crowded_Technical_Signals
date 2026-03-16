"""
Model evaluation: ML metrics + trading metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


def ml_metrics(y_true: list | np.ndarray, y_pred_proba: list | np.ndarray) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)
    y_bin = (y_pred >= 0.5).astype(int)

    # Top-decile precision
    threshold = np.percentile(y_pred, 90)
    top_decile_mask = y_pred >= threshold

    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
        "precision": precision_score(y_true, y_bin, zero_division=0),
        "recall": recall_score(y_true, y_bin, zero_division=0),
        "f1": f1_score(y_true, y_bin, zero_division=0),
        "top_decile_precision": (
            precision_score(y_true[top_decile_mask], y_bin[top_decile_mask], zero_division=0)
            if top_decile_mask.sum() > 0 else np.nan
        ),
        "n_samples": len(y_true),
        "base_rate": y_true.mean(),
    }


def strategy_metrics(
    returns: pd.Series,
    cost_bps: float = 10,
) -> dict:
    """
    Compute trading performance metrics from a series of trade returns.
    cost_bps: round-trip transaction cost in basis points.
    """
    cost = cost_bps / 10_000
    net_returns = returns - cost

    cum = (1 + net_returns).cumprod()
    total_ret = cum.iloc[-1] - 1 if len(cum) else np.nan

    ann_factor = 252
    mean_ret = net_returns.mean()
    std_ret = net_returns.std()

    sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else np.nan

    downside = net_returns[net_returns < 0].std()
    sortino = (mean_ret / downside * np.sqrt(ann_factor)) if downside > 0 else np.nan

    roll_max = cum.expanding().max()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    return {
        "n_trades": len(returns),
        "hit_rate": (returns > 0).mean(),
        "mean_gross_ret": returns.mean(),
        "mean_net_ret": mean_ret,
        "total_net_ret": total_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }
