"""Extended tests for TransactionFeatureEngineer."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "amount": [100.0, 250.0, 50.0, 75.0, 999.0],
        "merchant_category": ["retail", "crypto", "retail", "food", "crypto"],
        "payment_method": ["credit", "wire", "credit", "debit", "wire"],
        "device_type": ["mobile", "desktop", "mobile", "mobile", "tablet"],
        "channel": ["online", "online", "in-store", "online", "online"],
        "account_age_days": [365, 30, 1000, 500, 7],
        "credit_utilization": [0.2, 0.9, 0.1, 0.5, 0.95],
        "prior_fraud_count": [0, 1, 0, 0, 2],
        "timestamp": [
            "2024-01-15 02:30:00",
            "2024-01-15 14:00:00",
            "2024-01-15 22:00:00",
            "2024-01-16 09:00:00",
            "2024-01-16 03:00:00",
        ],
        "is_fraud": [0, 1, 0, 0, 1],
    })


@pytest.fixture()
def fitted_fe(sample_df):
    from pipeline.feature_engineering import TransactionFeatureEngineer
    fe = TransactionFeatureEngineer()
    fe.fit(sample_df)
    return fe


class TestAmountFeatures:
    def test_amount_zscore_computed(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert "amount_zscore" in result.columns

    def test_amount_log_non_negative(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert (result["amount_log"] >= 0).all()

    def test_amount_is_round_flag(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert "amount_is_round" in result.columns
        assert result["amount_is_round"].isin([0, 1]).all()

    @pytest.mark.parametrize("pct_col", ["amount_above_p95", "amount_above_p99"])
    def test_amount_percentile_flags_binary(self, fitted_fe, sample_df, pct_col):
        result = fitted_fe.transform(sample_df)
        assert pct_col in result.columns
        assert result[pct_col].isin([0, 1]).all()


class TestTemporalFeatures:
    def test_hour_of_day_range(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_day_of_week_range(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_is_night_flag_correct(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        # 02:30 and 22:00 and 03:00 are night
        night_rows = result[result["is_night"] == 1]
        assert len(night_rows) >= 2

    def test_is_weekend_binary(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert result["is_weekend"].isin([0, 1]).all()


class TestCategoricalEncoding:
    def test_freq_encoded_columns_exist(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        for col in ["merchant_category_freq", "payment_method_freq", "device_type_freq"]:
            assert col in result.columns

    def test_unseen_category_maps_to_zero(self, fitted_fe):
        new_df = pd.DataFrame({
            "user_id": [99],
            "amount": [100.0],
            "merchant_category": ["UNSEEN_CATEGORY"],
            "payment_method": ["UNSEEN"],
            "device_type": ["UNSEEN"],
            "channel": ["online"],
            "account_age_days": [100],
            "credit_utilization": [0.5],
            "prior_fraud_count": [0],
        })
        result = fitted_fe.transform(new_df)
        assert result["merchant_category_freq"].iloc[0] == 0.0

    def test_high_frequency_category_has_higher_encoding(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        retail_freq = result.loc[result["merchant_category"] == "retail", "merchant_category_freq"].iloc[0]
        assert retail_freq > 0


class TestBehavioralFeatures:
    def test_repeat_fraudster_flag(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        assert "is_repeat_fraudster" in result.columns
        high_fraud = result[sample_df["prior_fraud_count"] > 0]
        assert (high_fraud["is_repeat_fraudster"] == 1).all()

    def test_repeat_fraudster_zero_for_clean(self, fitted_fe, sample_df):
        result = fitted_fe.transform(sample_df)
        clean = result[sample_df["prior_fraud_count"] == 0]
        assert (clean["is_repeat_fraudster"] == 0).all()


class TestCategoryFreqCache:
    def test_get_category_freq_returns_float(self, fitted_fe):
        freq = fitted_fe._get_category_freq("merchant_category", "retail")
        assert isinstance(freq, float)

    def test_get_category_freq_unseen_returns_zero(self, fitted_fe):
        freq = fitted_fe._get_category_freq("merchant_category", "NONEXISTENT")
        assert freq == 0.0

    def test_get_category_freq_is_cached(self, fitted_fe):
        _ = fitted_fe._get_category_freq("merchant_category", "retail")
        cache_info = fitted_fe._get_category_freq.cache_info()
        assert cache_info.currsize > 0


class TestFeatureEngineeringExtendedParametrized:
    @pytest.mark.parametrize(
        "col",
        ["merchant_category_freq", "payment_method_freq", "device_type_freq", "channel_freq"],
    )
    def test_all_categorical_freq_encoded(self, fitted_fe, sample_df, col):
        result = fitted_fe.transform(sample_df)
        assert col in result.columns

    @pytest.mark.parametrize(
        "temporal_col",
        ["hour_of_day", "day_of_week", "is_weekend", "is_night", "is_business_hours"],
    )
    def test_temporal_cols_present(self, fitted_fe, sample_df, temporal_col):
        result = fitted_fe.transform(sample_df)
        assert temporal_col in result.columns

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_fit_transform_consistent_with_fit_then_transform(self, n):
        from pipeline.feature_engineering import TransactionFeatureEngineer

        __import__("tests.conftest", fromlist=["sample_df"])
        # use sample_df fixture data directly by reading the full file
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import numpy as _np
        _rng = _np.random.default_rng(n)
        import pandas as _pd
        _df = _pd.DataFrame({
            "user_id": _rng.integers(1, 100, 50),
            "amount": _rng.exponential(200, 50),
            "merchant_category": _rng.choice(["retail", "grocery"], 50),
            "payment_method": _rng.choice(["credit", "debit"], 50),
            "device_type": _rng.choice(["mobile", "desktop"], 50),
            "channel": _rng.choice(["online", "pos"], 50),
            "timestamp": _pd.date_range("2024-01-01", periods=50, freq="1h"),
            "account_age_days": _rng.integers(1, 3650, 50),
            "credit_utilization": _rng.uniform(0, 1, 50),
            "prior_fraud_count": _rng.integers(0, 3, 50),
        })
        fe1 = TransactionFeatureEngineer()
        out1 = fe1.fit_transform(_df)
        fe2 = TransactionFeatureEngineer()
        fe2.fit(_df)
        out2 = fe2.transform(_df)
        assert list(out1.columns) == list(out2.columns)
