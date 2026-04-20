from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly.anomaly_detector import AnomalyDetector


@pytest.fixture()
def detector():
    return AnomalyDetector(contamination=0.05, n_estimators=10, random_state=42)


@pytest.fixture()
def X_train():
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 5))


class TestFit:
    def test_fit_returns_self(self, detector, X_train):
        assert detector.fit(X_train) is detector

    def test_fit_sets_fitted_flag(self, detector, X_train):
        assert detector._fitted is False
        detector.fit(X_train)
        assert detector._fitted is True

    def test_fit_stores_score_stats(self, detector, X_train):
        detector.fit(X_train)
        for key in ("if", "lof", "ocsvm"):
            assert key in detector._score_stats
            assert "s_min" in detector._score_stats[key]
            assert "s_max" in detector._score_stats[key]

    def test_score_raises_before_fit(self, detector, X_train):
        with pytest.raises(RuntimeError, match="Fit the AnomalyDetector"):
            detector.score(X_train)


class TestScore:
    def test_score_output_shape(self, detector, X_train):
        detector.fit(X_train)
        assert detector.score(X_train).shape == (len(X_train),)

    def test_score_range_zero_to_one(self, detector, X_train):
        detector.fit(X_train)
        scores = detector.score(X_train)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_score_single_sample(self, detector, X_train):
        detector.fit(X_train)
        score = detector.score(X_train[:1])
        assert score.shape == (1,)
        assert 0.0 <= score[0] <= 1.0

    def test_score_batch_vs_single_in_range(self, detector, X_train):
        detector.fit(X_train)
        for i in range(5):
            s = detector.score(X_train[i:i+1])
            assert 0.0 <= s[0] <= 1.0

    def test_score_dataframe_input(self, detector, X_train):
        detector.fit(X_train)
        scores = detector.score(pd.DataFrame(X_train))
        assert scores.shape == (len(X_train),)

    def test_predict_binary_output(self, detector, X_train):
        detector.fit(X_train)
        labels = detector.predict(X_train)
        assert set(labels).issubset({0, 1})
