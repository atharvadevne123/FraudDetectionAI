"""
Unsupervised anomaly detection layer.
Uses Isolation Forest + Local Outlier Factor as a pre-filter before the supervised ensemble.
Anomaly scores feed into the ensemble as additional features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class AnomalyDetector:
    """
    Ensemble of three unsupervised detectors.
    Outputs a composite anomaly score in [0, 1] per transaction.
    Higher score = more anomalous.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iforest = IForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.lof = LOF(
            n_neighbors=20,
            contamination=contamination,
        )
        self.ocsvm = OCSVM(
            kernel="rbf",
            contamination=contamination,
        )
        self._fitted = False
        # Score range stats learned at fit time for stable single-sample inference
        self._score_stats: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X) -> "AnomalyDetector":
        X_arr = self._to_array(X)
        X_scaled = self.scaler.fit_transform(X_arr)
        logger.info("Fitting anomaly detectors on {} samples × {} features.", *X_scaled.shape)
        self.iforest.fit(X_scaled)
        self.lof.fit(X_scaled)
        self.ocsvm.fit(X_scaled)
        # Store training score ranges so single-sample inference normalises correctly
        self._score_stats = {
            "if": self._minmax(self.iforest.decision_function(X_scaled)),
            "lof": self._minmax(self.lof.decision_function(X_scaled)),
            "ocsvm": self._minmax(self.ocsvm.decision_function(X_scaled)),
        }
        self._fitted = True
        logger.success("Anomaly detectors fitted.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, X) -> np.ndarray:
        """Return composite anomaly score per sample, range [0, 1]."""
        self._check_fitted()
        X_scaled = self.scaler.transform(self._to_array(X))
        # Normalise using training-time ranges so single-sample inference works
        s_if = self._normalise_fixed(
            self.iforest.decision_function(X_scaled), **self._score_stats["if"]
        )
        s_lof = self._normalise_fixed(
            self.lof.decision_function(X_scaled), **self._score_stats["lof"]
        )
        s_ocsvm = self._normalise_fixed(
            self.ocsvm.decision_function(X_scaled), **self._score_stats["ocsvm"]
        )
        composite = 0.5 * s_if + 0.3 * s_lof + 0.2 * s_ocsvm
        return np.clip(composite, 0.0, 1.0)

    def predict(self, X) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        scores = self.score(X)
        threshold = 1 - self.contamination
        return (scores >= threshold).astype(int)

    def score_and_predict(self, X) -> pd.DataFrame:
        X_arr = self._to_array(X)
        scores = self.score(X_arr)
        labels = (scores >= (1 - self.contamination)).astype(int)
        return pd.DataFrame({"anomaly_score": scores, "is_anomaly": labels})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "anomaly_detector.joblib"
        joblib.dump(self, path)
        logger.info("AnomalyDetector saved → {}", path)
        return path

    @classmethod
    def load(cls, path=None) -> "AnomalyDetector":
        path = Path(path) if path else ARTIFACT_DIR / "anomaly_detector.joblib"
        obj = joblib.load(path)
        logger.info("AnomalyDetector loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(scores: np.ndarray) -> dict:
        return {"s_min": float(scores.min()), "s_max": float(scores.max())}

    @staticmethod
    def _normalise_fixed(scores: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    @staticmethod
    def _to_array(X) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Fit the AnomalyDetector before calling score/predict.")
