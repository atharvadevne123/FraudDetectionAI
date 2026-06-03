"""Tests for scripts/batch_predict.py helper functions."""
from __future__ import annotations

import pytest

from scripts.batch_predict import assign_risk_tier


class TestAssignRiskTier:
    @pytest.mark.parametrize("score,expected", [
        (0.95, "CRITICAL"),
        (0.90, "CRITICAL"),
        (0.75, "HIGH"),
        (0.70, "HIGH"),
        (0.55, "MEDIUM"),
        (0.50, "MEDIUM"),
        (0.35, "LOW"),
        (0.30, "LOW"),
        (0.10, "CLEAN"),
        (0.00, "CLEAN"),
    ])
    def test_score_maps_to_correct_tier(self, score, expected):
        assert assign_risk_tier(score) == expected

    def test_score_above_one_returns_clean(self):
        result = assign_risk_tier(2.0)
        assert isinstance(result, str)

    def test_all_tiers_reachable(self):
        tiers = {assign_risk_tier(s) for s in [0.95, 0.75, 0.55, 0.35, 0.10]}
        assert tiers == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"}

    def test_boundary_critical_low(self):
        assert assign_risk_tier(0.90) == "CRITICAL"

    def test_boundary_high_low(self):
        assert assign_risk_tier(0.70) == "HIGH"


class TestAssignRiskTierEdgeCases:
    @pytest.mark.parametrize("score", [0.0, 0.29999, 0.30, 0.49999, 0.50, 0.69999, 0.70, 0.89999, 0.90, 1.0])
    def test_all_boundaries_return_valid_tier(self, score):
        result = assign_risk_tier(score)
        assert result in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"}

    def test_negative_score_returns_clean(self):
        result = assign_risk_tier(-0.1)
        assert isinstance(result, str)

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_tier_string_is_uppercase(self, score):
        tier = assign_risk_tier(score)
        assert tier == tier.upper()
