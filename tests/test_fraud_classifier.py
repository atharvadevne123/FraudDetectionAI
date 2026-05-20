"""Tests for FraudEnsemble — mocked to avoid heavy training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # 94 normal + 6 fraud: fraud rate ~6.4%, enough for SMOTE with k=5
    n = 100
    return pd.DataFrame({
        "amount_zscore": rng.normal(0, 1, n),
        "amount_log": rng.uniform(1, 10, n),
        "credit_utilization": rng.uniform(0, 1, n),
        "account_age_days": rng.integers(1, 3650, n).astype(float),
    })


@pytest.fixture()
def fitted_ensemble(feature_df):
    from models.ensemble.fraud_classifier import FraudEnsemble
    y = pd.Series([0] * 94 + [1] * 6)
    ens = FraudEnsemble(random_state=0)
    ens.fit(feature_df, y, mlflow_run=False)
    return ens


class TestFraudEnsembleInit:
    def test_not_fitted_on_init(self):
        from models.ensemble.fraud_classifier import FraudEnsemble
        ens = FraudEnsemble()
        assert ens._fitted is False

    def test_shap_explainer_none_on_init(self):
        from models.ensemble.fraud_classifier import FraudEnsemble
        ens = FraudEnsemble()
        assert ens._shap_explainer is None

    def test_predict_raises_before_fit(self, feature_df):
        from models.ensemble.fraud_classifier import FraudEnsemble
        ens = FraudEnsemble()
        with pytest.raises(RuntimeError, match="Fit"):
            ens.predict_proba(feature_df)


class TestFraudEnsembleFit:
    def test_fit_sets_fitted_flag(self, fitted_ensemble):
        assert fitted_ensemble._fitted is True

    def test_fit_stores_feature_names(self, fitted_ensemble, feature_df):
        assert fitted_ensemble.feature_names == list(feature_df.columns)

    def test_fit_returns_self(self, feature_df):
        from models.ensemble.fraud_classifier import FraudEnsemble
        y = pd.Series([0] * 94 + [1] * 6)
        ens = FraudEnsemble(random_state=0)
        result = ens.fit(feature_df, y, mlflow_run=False)
        assert result is ens


class TestFraudEnsemblePredict:
    def test_predict_proba_shape(self, fitted_ensemble, feature_df):
        proba = fitted_ensemble.predict_proba(feature_df)
        assert proba.shape == (len(feature_df), 2)

    def test_predict_proba_sums_to_one(self, fitted_ensemble, feature_df):
        proba = fitted_ensemble.predict_proba(feature_df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, fitted_ensemble, feature_df):
        proba = fitted_ensemble.predict_proba(feature_df)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_binary_output(self, fitted_ensemble, feature_df):
        labels = fitted_ensemble.predict(feature_df, threshold=0.5)
        assert set(labels).issubset({0, 1})

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_predict_threshold_effect(self, fitted_ensemble, feature_df, threshold):
        labels = fitted_ensemble.predict(feature_df, threshold=threshold)
        proba = fitted_ensemble.predict_proba(feature_df)[:, 1]
        expected = (proba >= threshold).astype(int)
        np.testing.assert_array_equal(labels, expected)


class TestFraudEnsemblePersistence:
    def test_save_and_load(self, fitted_ensemble, feature_df, tmp_path):
        path = tmp_path / "ensemble.joblib"
        fitted_ensemble.save(path)
        assert path.exists()
        loaded = fitted_ensemble.__class__.load(path)
        assert loaded._fitted is True
        proba_orig = fitted_ensemble.predict_proba(feature_df)
        proba_load = loaded.predict_proba(feature_df)
        np.testing.assert_allclose(proba_orig, proba_load, atol=1e-6)
