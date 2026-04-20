from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_engineering import TransactionFeatureEngineer


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "user_id":           [1, 1, 2, 2, 3],
        "amount":            [100.0, 200.0, 50.0, 75.0, 5000.0],
        "account_age_days":  [365, 365, 730, 730, 12],
        "credit_utilization":[0.3, 0.4, 0.2, 0.5, 0.95],
        "prior_fraud_count": [0, 0, 0, 1, 2],
        "merchant_category": ["retail", "grocery", "retail", "crypto", "crypto"],
        "payment_method":    ["credit", "debit", "credit", "wire", "wire"],
        "device_type":       ["mobile", "desktop", "mobile", "mobile", "mobile"],
        "channel":           ["online", "pos", "online", "online", "online"],
        "timestamp":         pd.date_range("2024-01-01", periods=5, freq="h"),
    })


class TestFit:
    def test_fit_returns_self(self, sample_df):
        fe = TransactionFeatureEngineer()
        assert fe.fit(sample_df) is fe

    def test_fit_sets_fitted_flag(self, sample_df):
        fe = TransactionFeatureEngineer()
        assert fe._fitted is False
        fe.fit(sample_df)
        assert fe._fitted is True

    def test_fit_learns_category_maps(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        assert "merchant_category" in fe._category_maps
        assert "retail" in fe._category_maps["merchant_category"]

    def test_fit_learns_global_stats(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        assert "amount" in fe._global_stats
        stats = fe._global_stats["amount"]
        assert "mean" in stats and "std" in stats and "p95" in stats

    def test_transform_raises_before_fit(self, sample_df):
        fe = TransactionFeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(sample_df)


class TestTransform:
    def test_transform_adds_amount_features(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        out = fe.transform(sample_df)
        for col in ("amount_zscore", "amount_log", "amount_is_round",
                    "amount_above_p95", "amount_above_p99"):
            assert col in out.columns, f"Missing: {col}"

    def test_transform_adds_temporal_features(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        out = fe.transform(sample_df)
        for col in ("hour_of_day", "day_of_week", "is_weekend", "is_night"):
            assert col in out.columns

    def test_transform_adds_categorical_freq(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        out = fe.transform(sample_df)
        assert "merchant_category_freq" in out.columns

    def test_transform_does_not_mutate_input(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        original_cols = set(sample_df.columns)
        fe.transform(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_fit_transform_equivalent(self, sample_df):
        fe1 = TransactionFeatureEngineer()
        out1 = fe1.fit_transform(sample_df)
        fe2 = TransactionFeatureEngineer()
        fe2.fit(sample_df)
        out2 = fe2.transform(sample_df)
        pd.testing.assert_frame_equal(out1, out2)


class TestEdgeCases:
    def test_zero_amount_does_not_raise(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        zero_df = sample_df.copy()
        zero_df["amount"] = 0.0
        result = fe.transform(zero_df)
        assert "amount_log" in result.columns
        assert (result["amount_log"] == 0.0).all()

    def test_missing_timestamp_skips_temporal(self, sample_df):
        df_no_ts = sample_df.drop(columns=["timestamp"])
        fe = TransactionFeatureEngineer()
        fe.fit(df_no_ts)
        out = fe.transform(df_no_ts)
        assert "hour_of_day" not in out.columns

    def test_unseen_category_maps_to_zero(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        new_df = sample_df.copy()
        new_df["merchant_category"] = "unknown_new_category"
        out = fe.transform(new_df)
        assert (out["merchant_category_freq"] == 0.0).all()

    def test_single_row_transform(self, sample_df):
        fe = TransactionFeatureEngineer()
        fe.fit(sample_df)
        result = fe.transform(sample_df.iloc[:1].copy())
        assert len(result) == 1
