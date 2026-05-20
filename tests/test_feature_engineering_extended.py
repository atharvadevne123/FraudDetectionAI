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
