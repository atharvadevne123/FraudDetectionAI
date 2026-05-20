"""Tests for config.constants."""
from __future__ import annotations

import pytest

from config.constants import (
    API_VERSION,
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    ENSEMBLE_WEIGHTS,
    FEATURE_COUNT,
    MAX_BATCH_SIZE,
    RISK_TIERS,
    SMOTE_TARGET_RATIO,
    VELOCITY_WINDOWS,
)


class TestRiskTiers:
    def test_all_tiers_present(self):
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"):
            assert tier in RISK_TIERS

    def test_tiers_cover_full_range(self):
        lo = min(lo for lo, _ in RISK_TIERS.values())
        hi = max(hi for _, hi in RISK_TIERS.values())
        assert lo == pytest.approx(0.0)
        assert hi >= 1.0

    def test_tier_bounds_are_floats(self):
        for lo, hi in RISK_TIERS.values():
            assert isinstance(lo, float)
            assert isinstance(hi, float)

    def test_critical_threshold_highest(self):
        assert RISK_TIERS["CRITICAL"][0] >= RISK_TIERS["HIGH"][0]

    def test_clean_threshold_lowest(self):
        assert RISK_TIERS["CLEAN"][0] == pytest.approx(0.0)


class TestDefaults:
    def test_contamination_in_range(self):
        assert 0.0 < DEFAULT_CONTAMINATION < 0.5

    def test_n_estimators_positive(self):
        assert DEFAULT_N_ESTIMATORS > 0

    def test_smote_ratio_in_range(self):
        assert 0.0 < SMOTE_TARGET_RATIO < 1.0

    def test_feature_count_positive(self):
        assert FEATURE_COUNT > 0

    def test_max_batch_size_positive(self):
        assert MAX_BATCH_SIZE > 0


class TestEnsembleWeights:
    def test_weights_sum_to_one(self):
        assert sum(ENSEMBLE_WEIGHTS) == pytest.approx(1.0)

    def test_all_weights_positive(self):
        assert all(w > 0 for w in ENSEMBLE_WEIGHTS)

    def test_three_models(self):
        assert len(ENSEMBLE_WEIGHTS) == 3


class TestVelocityWindows:
    def test_windows_sorted(self):
        assert VELOCITY_WINDOWS == sorted(VELOCITY_WINDOWS)

    def test_all_windows_positive(self):
        assert all(w > 0 for w in VELOCITY_WINDOWS)

    def test_api_version_is_string(self):
        assert isinstance(API_VERSION, str)
        parts = API_VERSION.split(".")
        assert len(parts) == 3
