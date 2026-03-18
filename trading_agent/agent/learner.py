"""
Adaptive learner — two feedback loops:

  Fast  (every N completed trades):
    Recalibrate fade/follow thresholds based on recent win/loss patterns.
    Also identifies which alert types are performing vs. failing.

  Slow  (every retrain_frequency_days):
    Full XGBoost retrain on all accumulated universe data.
    The model literally learns from repeated live exposure.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PARENT = Path(__file__).resolve().parents[2]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

log = logging.getLogger(__name__)


class AdaptiveLearner:
    def __init__(self, cfg: dict, journal, predictor):
        self.cfg = cfg
        self.journal = journal
        self.predictor = predictor

        mcfg = cfg["model"]
        scfg = cfg["strategy"]

        self.fade_threshold   = scfg["fade_threshold"]
        self.follow_threshold = scfg["follow_threshold"]
        self.follow_disabled  = scfg.get("follow_disabled", True)

        self._min_trades       = mcfg["min_trades_for_recalibration"]
        self._retrain_days     = mcfg["retrain_frequency_days"]
        self._perf_trigger_pct = mcfg.get("retrain_perf_trigger_pct", 0.10)
        self._halflife_days    = mcfg.get("sample_weight_halflife_days", 252)
        self._step_up     = mcfg.get("threshold_step_up", 0.02)
        self._step_down   = mcfg.get("threshold_step_down", 0.01)
        self._win_target  = mcfg.get("win_rate_target", 0.55)

        # Baseline hit-rate recorded at last retrain — used for degradation trigger
        self._baseline_hit_rate: float | None = None

        # Per-alert-type win rate tracking
        self._alert_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})

        # Persist threshold state across sessions
        self._state_path = Path(cfg["model"]["path"]).parent / "learner_state.json"
        self._load_state()

    # ── State persistence ────────────────────────────────────────────────────

    def _load_state(self):
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                self.fade_threshold   = state.get("fade_threshold",   self.fade_threshold)
                self.follow_threshold = state.get("follow_threshold", self.follow_threshold)
                self._alert_stats     = defaultdict(
                    lambda: {"wins": 0, "total": 0},
                    state.get("alert_stats", {}),
                )
                log.info("Learner state loaded: fade=%.3f  follow=%.3f",
                         self.fade_threshold, self.follow_threshold)
            except Exception as e:
                log.warning("Could not load learner state: %s", e)

    def _save_state(self):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump({
                "fade_threshold":   round(self.fade_threshold, 4),
                "follow_threshold": round(self.follow_threshold, 4),
                "alert_stats":      dict(self._alert_stats),
                "updated":          datetime.utcnow().isoformat(),
            }, f, indent=2)

    # ── Outcome recording ────────────────────────────────────────────────────

    def is_alert_suppressed(self, action: str, alert_name: str) -> bool:
        """
        Return True if this (action, alert_name) combination has a win rate
        low enough to suppress new signals.

        Gate:
          - Requires at least min_trades_for_recalibration samples before blocking.
          - Blocks when win rate < suppress_win_rate_threshold (default 0.25).
          - Only applies to action types that are currently enabled
            (FADE always eligible; FOLLOW only when not follow_disabled).
        """
        if action == "FOLLOW" and self.follow_disabled:
            return False  # already disabled globally — no need for per-alert check
        min_n     = self._min_trades
        threshold = self.cfg["model"].get("suppress_win_rate_threshold", 0.25)
        key       = f"{action}:{alert_name}"
        stats     = self._alert_stats.get(key)
        if not stats or stats.get("total", 0) < min_n:
            return False   # not enough data yet
        win_rate = stats["wins"] / stats["total"]
        suppressed = win_rate < threshold
        if suppressed:
            log.info(
                "Alert suppressed: %s  win=%.0f%%  n=%d  (threshold=%.0f%%)",
                key, win_rate * 100, stats["total"], threshold * 100,
            )
        return suppressed

    def record_outcome(self, alert_name: str, action: str, pnl_net: float):
        """
        Call when a bracket order closes (TP or SL hit).
        Immediately updates per-alert statistics.
        """
        won = pnl_net > 0
        key = f"{action}:{alert_name}"
        self._alert_stats[key]["total"] += 1
        self._alert_stats[key]["wins"]  += int(won)
        self._save_state()
        log.info("Outcome: %s %s  pnl=%.2f EUR  won=%s", action, alert_name, pnl_net, won)

    # ── Fast loop ────────────────────────────────────────────────────────────

    def maybe_recalibrate(self):
        """Recalibrate thresholds after each cycle if enough data exists."""
        trades = self.journal.get_recent_trades(n=self._min_trades * 4)
        completed = [t for t in trades if t.get("pnl_net") is not None]

        if len(completed) < self._min_trades:
            log.info("Recalibration pending: %d/%d completed trades",
                     len(completed), self._min_trades)
            return

        changed = False

        # ── Fade threshold ────────────────────────────────────────────────
        fade_trades = [t for t in completed if t.get("action") == "FADE"]
        if len(fade_trades) >= self._min_trades:
            wins = sum(1 for t in fade_trades if (t.get("pnl_net") or 0) > 0)
            win_rate = wins / len(fade_trades)
            old = self.fade_threshold

            if win_rate < self._win_target - 0.05:
                # Losing → be more selective (raise threshold)
                self.fade_threshold = round(min(old + self._step_up, 0.85), 3)
            elif win_rate > self._win_target + 0.10:
                # Winning well → loosen slightly to get more trades
                self.fade_threshold = round(max(old - self._step_down, 0.55), 3)

            if self.fade_threshold != old:
                log.info("FADE threshold: %.3f → %.3f  (win_rate=%.1f%%  n=%d)",
                         old, self.fade_threshold, win_rate * 100, len(fade_trades))
                changed = True

            self._log_alert_breakdown(fade_trades, "FADE")

        # ── Follow threshold ──────────────────────────────────────────────
        if not self.follow_disabled:
            follow_trades = [t for t in completed if t.get("action") == "FOLLOW"]
            if len(follow_trades) >= self._min_trades:
                wins = sum(1 for t in follow_trades if (t.get("pnl_net") or 0) > 0)
                win_rate = wins / len(follow_trades)
                old = self.follow_threshold

                if win_rate < self._win_target - 0.05:
                    self.follow_threshold = round(max(old - self._step_up, 0.10), 3)
                elif win_rate > self._win_target + 0.10:
                    self.follow_threshold = round(min(old + self._step_down, 0.45), 3)

                if self.follow_threshold != old:
                    log.info("FOLLOW threshold: %.3f → %.3f  (win_rate=%.1f%%)",
                             old, self.follow_threshold, win_rate * 100)
                    changed = True

        if changed:
            self._save_state()

        self._print_summary(completed)

    def _log_alert_breakdown(self, trades: list[dict], action: str):
        by_alert: dict[str, list] = defaultdict(list)
        for t in trades:
            pnl = t.get("pnl_net")
            if pnl is not None:
                by_alert[t.get("alert_name", "unknown")].append(pnl > 0)

        rows = sorted(
            [(a, len(o), sum(o) / len(o)) for a, o in by_alert.items()],
            key=lambda r: r[2], reverse=True,
        )
        lines = "\n".join(f"    {a:30s}  n={n}  win={w:.0%}" for a, n, w in rows)
        log.info("%s per-alert breakdown:\n%s", action, lines or "    (no data)")

    def _print_summary(self, completed: list[dict]):
        n = len(completed)
        if n == 0:
            return
        wins = sum(1 for t in completed if (t.get("pnl_net") or 0) > 0)
        total_pnl = sum(t.get("pnl_net") or 0 for t in completed)
        log.info(
            "--- Learner: %d trades  hit=%.1f%%  pnl=%.2f EUR"
            "  fade_thr=%.3f  follow_thr=%.3f ---",
            n, wins / n * 100, total_pnl,
            self.fade_threshold, self.follow_threshold,
        )

    # ── Slow loop: full retrain ───────────────────────────────────────────────

    def maybe_retrain(self, universe_data: dict, index_close: pd.Series | None = None):
        """
        Retrain model if the scheduled window has passed OR performance has degraded.

        Triggers:
          1. Time-based  : model file is older than retrain_frequency_days (default 30 = monthly).
          2. Perf-based  : live hit-rate has dropped more than retrain_perf_trigger_pct (default 10pp)
                           versus the baseline recorded at the previous retrain.
                           Rationale: market regime shifts do not follow a calendar. A sudden drop
                           in hit-rate (e.g. crowding behaviour changes) should trigger an immediate
                           refresh rather than waiting for the monthly window.
        """
        model_path = Path(self.cfg["model"]["path"])

        perf = self.journal.get_performance_summary()
        n_trades = perf.get("n_trades", 0)
        if n_trades < 10:
            log.info("Retrain skipped — need ≥10 trades, have %d", n_trades)
            return

        time_due = True
        if model_path.exists():
            age = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).days
            time_due = age >= self._retrain_days
            if time_due:
                log.info("Model %d days old (>= %d) — scheduled monthly retrain",
                         age, self._retrain_days)
            else:
                log.info("Model %d days old — next scheduled retrain in %d days",
                         age, self._retrain_days - age)

        # Performance degradation check
        perf_due = False
        if self._baseline_hit_rate is not None and n_trades >= 20:
            live_hit_rate = perf.get("hit_rate", 0.0)
            drop = self._baseline_hit_rate - live_hit_rate
            if drop >= self._perf_trigger_pct:
                log.warning(
                    "Hit-rate dropped %.1fpp (baseline=%.1f%%  current=%.1f%%) "
                    ">= trigger %.1fpp — early retrain triggered",
                    drop * 100, self._baseline_hit_rate * 100,
                    live_hit_rate * 100, self._perf_trigger_pct * 100,
                )
                perf_due = True

        if not time_due and not perf_due:
            return

        log.info("Starting model retrain (time_due=%s  perf_due=%s)...", time_due, perf_due)
        self._retrain(universe_data, index_close)

        # Record new baseline after successful retrain
        self._baseline_hit_rate = perf.get("hit_rate", 0.0)
        log.info("Retrain baseline hit-rate set to %.1f%%", self._baseline_hit_rate * 100)

    def _retrain(self, universe_data: dict, index_close: pd.Series | None):
        try:
            from src.data.preprocess import build_panel
            from src.features.engineering import build_features, add_alert_features
            from src.features.labels import compute_forward_returns, assign_labels
            from src.alerts.engine import run_alert_engine
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score

            raw_dict = {}
            for ticker, df in universe_data.items():
                d = df.rename(columns=str.title)
                d.index = pd.to_datetime(d["Date"] if "Date" in d else d.index)
                raw_dict[ticker] = d

            panel = build_panel(raw_dict, min_history_days=self.cfg["model"]["min_history_days"])
            feat_panel = build_features(panel, index_close=index_close)
            feat_panel = compute_forward_returns(feat_panel, horizons=[3])

            events = run_alert_engine(panel)
            events = add_alert_features(events, panel)
            labeled = assign_labels(events, feat_panel, horizons=[3], theta=0.005)

            EXCLUDE = {"date","ticker","open","high","low","close","volume",
                       "ret_1d_lead","fwd_ret_1d","fwd_ret_3d","fwd_ret_5d",
                       "alert_name","direction","n_simultaneous_alerts","_dir_raw"}
            price_cols = [c for c in feat_panel.columns if c not in EXCLUDE]

            labeled["_dir_raw"] = labeled["direction"]
            labeled = pd.get_dummies(labeled, columns=["direction", "alert_name"])
            labeled = labeled.merge(
                feat_panel[["date","ticker"] + price_cols].drop_duplicates(["date","ticker"]),
                on=["date","ticker"], how="left",
            )

            feat_cols = self.predictor.feature_cols
            label_col = "label_failure_3d"
            valid = labeled[label_col].notna()
            X = labeled.loc[valid, [c for c in feat_cols if c in labeled.columns]].fillna(-1)
            y = labeled.loc[valid, label_col]

            if len(y) < 100:
                log.warning("Not enough samples for retrain (%d)", len(y))
                return

            split = int(len(X) * 0.85)
            neg, pos = (y.iloc[:split] == 0).sum(), (y.iloc[:split] == 1).sum()
            spw = neg / pos if pos > 0 else 1.0

            # Recency sample weights — exponential decay so recent data matters more.
            # Half-life from config (default 252 days ≈ 1 trading year).
            sample_weight = None
            if "date" in labeled.columns:
                try:
                    import numpy as _np
                    dates_all = pd.to_datetime(labeled.loc[valid, "date"])
                    ts = dates_all.values.astype("datetime64[D]").astype(float)
                    age_days = ts.max() - ts
                    decay = _np.log(2) / self._halflife_days
                    w = _np.exp(-decay * age_days)
                    w = w / w.sum() * len(w)
                    sample_weight = w[:split]
                    log.info("  Retrain sample weights: halflife=%dd  min=%.3f  max=%.3f",
                             self._halflife_days, sample_weight.min(), sample_weight.max())
                except Exception as we:
                    log.debug("Could not compute sample weights: %s", we)

            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=-1,
                scale_pos_weight=spw,
            )
            model.fit(X.iloc[:split], y.iloc[:split],
                      sample_weight=sample_weight, verbose=False)

            val_auc = roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:, 1])
            log.info("Retrained model: val_ROC-AUC=%.4f  n_samples=%d", val_auc, len(y))
            self.predictor.save(model, feat_cols)

        except Exception as e:
            log.error("Retrain failed: %s", e, exc_info=True)
