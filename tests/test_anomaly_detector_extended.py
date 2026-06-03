"""Extended tests for AnomalyDetector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 5))


@pytest.fixture()
def fitted_detector(sample_data):
    from models.anomaly.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(contamination=0.05, n_estimators=50, random_state=42)
    det.fit(sample_data)
    return det


class TestAnomalyDetectorInit:
    def test_not_fitted_on_init(self):
        from models.anomaly.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det._fitted is False

    def test_score_raises_before_fit(self):
        from models.anomaly.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        with pytest.raises(RuntimeError, match="Fit"):
            det.score(np.zeros((1, 5)))

    def test_custom_contamination(self):
        from models.anomaly.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(contamination=0.10)
        assert det.contamination == 0.10

    @pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10, 0.20])
    def test_various_contamination_rates(self, sample_data, contamination):
        from models.anomaly.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(contamination=contamination, n_estimators=30, random_state=0)
        det.fit(sample_data)
        scores = det.score(sample_data)
        assert scores.shape == (len(sample_data),)


class TestAnomalyDetectorScore:
    def test_score_shape(self, fitted_detector, sample_data):
        scores = fitted_detector.score(sample_data)
        assert scores.shape == (len(sample_data),)

    def test_score_in_range(self, fitted_detector, sample_data):
        scores = fitted_detector.score(sample_data)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_score_accepts_dataframe(self, fitted_detector, sample_data):
        df = pd.DataFrame(sample_data, columns=[f"f{i}" for i in range(sample_data.shape[1])])
        scores = fitted_detector.score(df)
        assert scores.shape == (len(df),)

    def test_score_single_sample(self, fitted_detector):
        sample = np.zeros((1, 5))
        scores = fitted_detector.score(sample)
        assert scores.shape == (1,)


class TestAnomalyDetectorPredict:
    def test_predict_binary(self, fitted_detector, sample_data):
        labels = fitted_detector.predict(sample_data)
        assert set(labels).issubset({0, 1})

    def test_predict_shape(self, fitted_detector, sample_data):
        labels = fitted_detector.predict(sample_data)
        assert labels.shape == (len(sample_data),)


class TestAnomalyDetectorPersistence:
    def test_save_and_load(self, fitted_detector, sample_data, tmp_path):
        path = tmp_path / "detector.joblib"
        fitted_detector.save(path)
        assert path.exists()
        loaded = fitted_detector.__class__.load(path)
        assert loaded._fitted is True
        orig_scores = fitted_detector.score(sample_data)
        load_scores = loaded.score(sample_data)
        np.testing.assert_allclose(orig_scores, load_scores, atol=1e-6)


class TestAnomalyDetectorExtendedParametrized:
    @pytest.mark.parametrize("n_rows", [1, 10, 50])
    def test_score_and_predict_row_counts(self, fitted_detector, n_rows):
        rng = __import__("numpy").random.default_rng(99)
        X = rng.standard_normal((n_rows, 5))
        result = fitted_detector.score_and_predict(X)
        assert len(result) == n_rows
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    @pytest.mark.parametrize("contamination", [0.01, 0.05, 0.20])
    def test_predict_proba_sums_to_one(self, contamination, sample_data):
        det = __import__(
            "models.anomaly.anomaly_detector",
            fromlist=["AnomalyDetector"],
        ).AnomalyDetector(contamination=contamination, n_estimators=10)
        det.fit(sample_data)
        proba = det.predict_proba(sample_data[:5])
        assert __import__("numpy").allclose(proba.sum(axis=1), 1.0)
