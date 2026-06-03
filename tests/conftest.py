from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def app():
    """Return a Flask test application with rate limiting disabled."""
    from api.app import app as flask_app

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    return flask_app


@pytest.fixture()
def client(app):
    """Return a test client for the Flask application."""
    return app.test_client()


@pytest.fixture()
def mock_ensemble():
    """Return a lightweight mock FraudEnsemble that always returns 0.85 fraud probability."""

    class _MockEnsemble:
        _shap_explainer = None
        _fitted = True

        def predict_proba(self, X):
            n = len(X) if hasattr(X, "__len__") else 1
            return np.column_stack([np.full(n, 0.15), np.full(n, 0.85)])

        def predict(self, X, threshold: float = 0.5):
            proba = self.predict_proba(X)[:, 1]
            return (proba >= threshold).astype(int)

        def explain(self, X):
            return [{"shap_values": {"amount_zscore": 0.42}, "base_value": 0.1}]

    return _MockEnsemble()


@pytest.fixture()
def mock_anomaly_detector():
    """Return a lightweight mock AnomalyDetector that always returns 0.72 anomaly score."""

    class _MockAnomalyDetector:
        _fitted = True
        contamination = 0.02

        def score(self, X):
            n = len(X) if hasattr(X, "__len__") else 1
            return np.full(n, 0.72)

        def predict(self, X):
            return np.zeros(len(X) if hasattr(X, "__len__") else 1, dtype=int)

    return _MockAnomalyDetector()


@pytest.fixture()
def patch_models(monkeypatch, mock_ensemble, mock_anomaly_detector):
    """Monkeypatch the global model objects in api.app with lightweight mocks."""
    import api.app as app_module
    monkeypatch.setattr(app_module, "_ensemble", mock_ensemble)
    monkeypatch.setattr(app_module, "_anomaly_detector", mock_anomaly_detector)
    monkeypatch.setattr(app_module, "_feature_engineer", None)
    monkeypatch.setattr(app_module, "_rag_explainer", None)
    monkeypatch.setattr(app_module, "_feature_cols", [])


@pytest.fixture()
def valid_transaction():
    return {
        "user_id": 12345,
        "amount": 4999.99,
        "merchant_category": "crypto",
        "payment_method": "wire",
        "device_type": "mobile",
        "channel": "online",
        "account_age_days": 30,
        "credit_utilization": 0.85,
        "prior_fraud_count": 1,
        "explain": False,
    }


@pytest.fixture()
def clean_transaction():
    return {
        "user_id": 99999,
        "amount": 12.50,
        "merchant_category": "restaurant",
        "payment_method": "debit",
        "device_type": "mobile",
        "channel": "in-store",
        "account_age_days": 2000,
        "credit_utilization": 0.05,
        "prior_fraud_count": 0,
        "explain": False,
    }
