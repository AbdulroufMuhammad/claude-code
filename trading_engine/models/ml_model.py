from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from trading_engine.config import Config
from trading_engine.features.feature_engine import FeatureSet
from trading_engine.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MLModel(BaseModel):
    """Logistic Regression (or XGBoost) classifier for directional prediction.

    During warm-up returns 0.5. Auto-trains once MIN_SAMPLES outcomes are
    available and persists the model to disk via joblib.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._clf = None
        self._samples: List[Tuple[np.ndarray, int]] = []
        self._fitted = False
        self._load_if_exists()

    # ------------------------------------------------------------------
    # BaseModel
    # ------------------------------------------------------------------

    def predict(self, features: FeatureSet) -> float:
        if not self._fitted or self._clf is None:
            return 0.5
        x = self._to_vector(features).reshape(1, -1)
        try:
            prob = float(self._clf.predict_proba(x)[0][1])
            return max(0.0, min(1.0, prob))
        except Exception as exc:
            logger.warning("MLModel predict error: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def add_sample(self, features: FeatureSet, label: int) -> None:
        """label = 1 if next candle closed UP, 0 otherwise."""
        self._samples.append((self._to_vector(features), label))

    def maybe_fit(self) -> bool:
        """Fit or refit if enough samples are available. Returns True if fitted."""
        if len(self._samples) < self._cfg.ML_TRAIN_MIN_SAMPLES:
            return False
        X = np.array([s[0] for s in self._samples])
        y = np.array([s[1] for s in self._samples])
        if len(np.unique(y)) < 2:
            return False
        try:
            clf = self._build_classifier()
            clf.fit(X, y)
            self._clf = clf
            self._fitted = True
            self._save()
            logger.info("MLModel fitted on %d samples", len(self._samples))
            return True
        except Exception as exc:
            logger.warning("MLModel fit failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            import joblib
            os.makedirs(os.path.dirname(self._cfg.ML_MODEL_PATH), exist_ok=True)
            joblib.dump(self._clf, self._cfg.ML_MODEL_PATH)
        except Exception as exc:
            logger.warning("MLModel save failed: %s", exc)

    def _load_if_exists(self) -> None:
        if not os.path.exists(self._cfg.ML_MODEL_PATH):
            return
        try:
            import joblib
            self._clf = joblib.load(self._cfg.ML_MODEL_PATH)
            self._fitted = True
            logger.info("MLModel loaded from %s", self._cfg.ML_MODEL_PATH)
        except Exception as exc:
            logger.warning("MLModel load failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector(features: FeatureSet) -> np.ndarray:
        return np.array([
            features.ma20,
            features.ma50,
            features.volatility,
            features.momentum,
            features.drift,
            features.vol_zscore,
        ], dtype=float)

    def _build_classifier(self):
        if self._cfg.ML_USE_XGBOOST:
            try:
                from xgboost import XGBClassifier  # type: ignore
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )
            except ImportError:
                logger.warning("xgboost not installed, falling back to LogisticRegression")

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
