"""Tests for AnomalyDetector.predict_proba and score_and_predict."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def fitted_detector():
    from models.anomaly.anomaly_detector import AnomalyDetector
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))
    det = AnomalyDetector(contamination=0.05, n_estimators=30, random_state=0)
    det.fit(X)
    return det, X


class TestPredictProba:
    def test_shape(self, fitted_detector):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_rows_sum_to_one(self, fitted_detector):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_values_in_range(self, fitted_detector):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_second_column_equals_score(self, fitted_detector):
        det, X = fitted_detector
        scores = det.score(X)
        proba = det.predict_proba(X)
        np.testing.assert_allclose(proba[:, 1], scores, atol=1e-6)


class TestScoreAndPredict:
    def test_returns_dataframe(self, fitted_detector):
        det, X = fitted_detector
        result = det.score_and_predict(X)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, fitted_detector):
        det, X = fitted_detector
        result = det.score_and_predict(X)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_is_anomaly_binary(self, fitted_detector):
        det, X = fitted_detector
        result = det.score_and_predict(X)
        assert set(result["is_anomaly"].unique()).issubset({0, 1})

    def test_score_in_range(self, fitted_detector):
        det, X = fitted_detector
        result = det.score_and_predict(X)
        assert result["anomaly_score"].min() >= 0.0
        assert result["anomaly_score"].max() <= 1.0


class TestPredictProbaParametrized:
    @pytest.mark.parametrize("n", [1, 5, 50])
    def test_proba_shape_various_sizes(self, fitted_detector, n):
        det, X = fitted_detector
        rng = __import__("numpy").random.default_rng(77)
        X_test = rng.standard_normal((n, X.shape[1]))
        proba = det.predict_proba(X_test)
        assert proba.shape == (n, 2)

    @pytest.mark.parametrize("idx", [0, 1])
    def test_proba_column_in_range(self, fitted_detector, idx):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        assert proba[:, idx].min() >= 0.0
        assert proba[:, idx].max() <= 1.0

    def test_normal_samples_low_anomaly_score(self, fitted_detector):
        det, X_train = fitted_detector
        rng = __import__("numpy").random.default_rng(88)
        normal = rng.standard_normal((20, X_train.shape[1])) * 0.1
        scores = det.score(normal)
        assert scores.mean() < 0.9
