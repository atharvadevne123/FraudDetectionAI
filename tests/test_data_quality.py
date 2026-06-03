"""Tests for scripts/check_data_quality helper functions."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.check_data_quality import (
    REQUIRED_COLS,
    check_duplicate_rows,
    check_missing_values,
    check_numeric_bounds,
    check_required_columns,
)


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "amount": [100.0, 250.0, 50.0],
        "merchant_category": ["retail", "crypto", "food"],
        "payment_method": ["credit", "wire", "debit"],
        "credit_utilization": [0.2, 0.9, 0.1],
        "account_age_days": [365, 30, 1000],
        "prior_fraud_count": [0, 1, 0],
    })


class TestRequiredColumns:
    def test_all_present_no_issues(self, valid_df):
        assert check_required_columns(valid_df) == []

    def test_missing_amount(self, valid_df):
        df = valid_df.drop(columns=["amount"])
        issues = check_required_columns(df)
        assert any("amount" in i for i in issues)

    def test_empty_df_returns_issues(self):
        issues = check_required_columns(pd.DataFrame())
        assert len(issues) >= 1

    def test_all_required_cols_defined(self):
        assert len(REQUIRED_COLS) >= 2


class TestMissingValues:
    def test_no_missing_no_issues(self, valid_df):
        assert check_missing_values(valid_df, threshold=0.05) == []

    def test_column_with_all_missing(self, valid_df):
        valid_df["amount"] = None
        issues = check_missing_values(valid_df, threshold=0.05)
        assert any("amount" in i for i in issues)

    def test_threshold_zero_catches_any_missing(self):
        df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
        issues = check_missing_values(df, threshold=0.0)
        assert any("a" in i for i in issues)

    def test_high_threshold_allows_missing(self):
        df = pd.DataFrame({"a": [None, None, 1], "b": [1, 2, 3]})
        issues = check_missing_values(df, threshold=1.0)
        assert issues == []


class TestNumericBounds:
    def test_valid_values_no_issues(self, valid_df):
        assert check_numeric_bounds(valid_df) == []

    def test_negative_amount_flagged(self, valid_df):
        valid_df.loc[0, "amount"] = -100.0
        issues = check_numeric_bounds(valid_df)
        assert any("amount" in i for i in issues)

    def test_utilization_above_one_flagged(self, valid_df):
        valid_df.loc[0, "credit_utilization"] = 1.5
        issues = check_numeric_bounds(valid_df)
        assert any("credit_utilization" in i for i in issues)

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"user_id": [1, 2]})
        assert check_numeric_bounds(df) == []


class TestDuplicateRows:
    def test_no_duplicates_no_issues(self, valid_df):
        assert check_duplicate_rows(valid_df) == []

    def test_duplicate_rows_flagged(self, valid_df):
        df_dup = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        issues = check_duplicate_rows(df_dup)
        assert len(issues) == 1
        assert "1 duplicate" in issues[0]


class TestDataQualityParametrized:
    @pytest.mark.parametrize(
        "col,bad_value",
        [
            ("amount", -1.0),
            ("credit_utilization", 1.5),
            ("account_age_days", -5),
        ],
    )
    def test_out_of_range_flagged(self, valid_df, col, bad_value):
        from scripts.check_data_quality import check_numeric_bounds

        df = valid_df.copy()
        if col in df.columns:
            df.loc[0, col] = bad_value
            issues = check_numeric_bounds(df)
            assert any(col in i for i in issues)

    @pytest.mark.parametrize("threshold", [0.0, 0.05, 0.5])
    def test_zero_missing_always_passes_threshold(self, valid_df, threshold):
        from scripts.check_data_quality import check_missing_values

        issues = check_missing_values(valid_df, threshold)
        assert issues == []

    @pytest.mark.parametrize("n_dupes", [1, 3, 10])
    def test_n_duplicates_flagged_correctly(self, valid_df, n_dupes):
        from scripts.check_data_quality import check_duplicate_rows

        extras = pd.concat([valid_df.iloc[[0]]] * n_dupes, ignore_index=True)
        df = pd.concat([valid_df, extras], ignore_index=True)
        issues = check_duplicate_rows(df)
        assert len(issues) == 1
        assert str(n_dupes) in issues[0]
