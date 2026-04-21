"""
Transaction feature engineering for fraud detection.
Generates behavioral, velocity, and statistical features from raw transaction streams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TransactionFeatureEngineer:
    """
    Builds a rich feature set from raw transaction records.
    Designed to operate on both historical batches and sliding windows for streaming.
    """

    CATEGORICAL_COLS = ["merchant_category", "payment_method", "device_type", "channel"]
    NUMERIC_COLS = ["amount", "account_age_days", "credit_utilization", "prior_fraud_count"]

    def __init__(self, velocity_windows: list = None):
        self.velocity_windows = velocity_windows if velocity_windows is not None else [1, 7, 30]
        self._fitted = False
        self._category_maps = {}
        self._global_stats = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureEngineer":
        """Learn encoding maps and global statistics from training data."""
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                self._category_maps[col] = freq.to_dict()

        for col in self.NUMERIC_COLS:
            if col in df.columns:
                self._global_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std() + 1e-9,
                    "p95": df[col].quantile(0.95),
                    "p99": df[col].quantile(0.99),
                }
        self._fitted = True
        logger.info("FeatureEngineer fitted on {:,} records.", len(df))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations. Returns enriched DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        df = self._amount_features(df)
        df = self._temporal_features(df)
        df = self._velocity_features(df)
        df = self._categorical_encoding(df)
        df = self._behavioral_features(df)
        df = self._geo_features(df)
        df = self._device_features(df)
        logger.debug("Transformed {:,} transactions → {} features.", len(df), df.shape[1])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = self._global_stats.get("amount", {"mean": 1, "std": 1, "p95": 1, "p99": 1})
        df["amount_zscore"] = (df["amount"] - stats["mean"]) / stats["std"]
        df["amount_log"] = np.log1p(df["amount"])
        df["amount_is_round"] = (df["amount"] % 1 == 0).astype(int)
        df["amount_above_p95"] = (df["amount"] > stats["p95"]).astype(int)
        df["amount_above_p99"] = (df["amount"] > stats["p99"]).astype(int)
        # Amount deviation from user's own median
        if "user_id" in df.columns:
            user_medians = df.groupby("user_id")["amount"].transform("median")
            df["amount_vs_user_median"] = df["amount"] / (user_medians + 1e-9)
        return df

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            return df
        ts = pd.to_datetime(df["timestamp"])
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
        df["is_business_hours"] = (
            (df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17) & (df["is_weekend"] == 0)
        ).astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction counts and total amounts within rolling windows per user."""
        if "user_id" not in df.columns or "timestamp" not in df.columns:
            return df
        df = df.sort_values("timestamp")
        ts = pd.to_datetime(df["timestamp"])
        for window in self.velocity_windows:
            cutoff = ts - pd.Timedelta(days=window)
            counts, totals = [], []
            for i, (uid, t) in enumerate(zip(df["user_id"], ts)):
                mask = (df["user_id"] == uid) & (ts >= cutoff.iloc[i]) & (ts < t)
                counts.append(mask.sum())
                totals.append(df.loc[mask, "amount"].sum())
            df[f"txn_count_{window}d"] = counts
            df[f"txn_total_{window}d"] = totals
        return df

    def _categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Frequency-encode categoricals; fallback to 0 for unseen values."""
        for col, freq_map in self._category_maps.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
        return df

    def _behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "user_id" not in df.columns:
            return df
        if "merchant_id" in df.columns:
            user_merchant = df.groupby("user_id")["merchant_id"].transform("nunique")
            df["unique_merchants"] = user_merchant
        if "ip_address" in df.columns:
            user_ips = df.groupby("user_id")["ip_address"].transform("nunique")
            df["unique_ips"] = user_ips
        if "prior_fraud_count" in df.columns:
            df["is_repeat_fraudster"] = (df["prior_fraud_count"] > 0).astype(int)
        return df

    def _geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag impossible velocity (same user, different country in short time)."""
        if "country" not in df.columns or "user_id" not in df.columns:
            return df
        user_countries = df.groupby("user_id")["country"].transform("nunique")
        df["multi_country_session"] = (user_countries > 1).astype(int)
        return df

    def _device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "device_fingerprint" not in df.columns or "user_id" not in df.columns:
            return df
        user_devices = df.groupby("user_id")["device_fingerprint"].transform("nunique")
        df["unique_devices"] = user_devices
        df["new_device"] = (user_devices > 3).astype(int)
        return df
