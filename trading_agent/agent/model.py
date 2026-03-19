"""
ML model loader and predictor.
Loads the XGBoost model trained by the research pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FailurePredictor:
    def __init__(self, model_path: str, feature_cols_path: str):
        self.model_path = Path(model_path)
        self.feature_cols_path = Path(feature_cols_path)
        self._model = None
        self._feature_cols: list[str] = []

    def load(self) -> bool:
        if not self.model_path.exists():
            log.error("Model not found at %s — run bootstrap_model.py first", self.model_path)
            return False
        if not self.feature_cols_path.exists():
            log.error("Feature cols not found at %s", self.feature_cols_path)
            return False

        self._model = joblib.load(self.model_path)
        from agent.features import load_feature_cols
        self._feature_cols = load_feature_cols(str(self.feature_cols_path))
        log.info("Model loaded from %s (%d features)", self.model_path, len(self._feature_cols))
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Predict P(trade is profitable) for each row.
        Returns a Series of probabilities indexed the same as feature_df.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = feature_df[self._feature_cols].fillna(-1)
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=feature_df.index, name="failure_proba")

    def save(self, model, feature_cols: list[str]):
        """Save a newly trained model."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        from agent.features import save_feature_cols
        save_feature_cols(feature_cols, str(self.feature_cols_path))
        self._model = model
        self._feature_cols = feature_cols
        log.info("Model saved to %s", self.model_path)
