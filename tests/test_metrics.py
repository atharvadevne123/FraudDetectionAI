"""Tests for utils.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import (
    average_precision,
    detection_rate,
    false_positive_rate,
    precision_at_k,
)


class TestPrecisionAtK:
    def test_all_fraud_in_top_k(self):
        y = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert precision_at_k(y, scores, k=2) == 1.0

    def test_no_fraud_in_top_k(self):
        y = np.array([0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert precision_at_k(y, scores, k=2) == 0.0

    def test_k_larger_than_n(self):
        y = np.array([1, 0])
        scores = np.array([0.9, 0.1])
        result = precision_at_k(y, scores, k=100)
        assert 0.0 <= result <= 1.0

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(np.array([1]), np.array([0.5]), k=0)

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_result_in_range(self, k):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=10)
        s = rng.random(10)
        r = precision_at_k(y, s, k=k)
        assert 0.0 <= r <= 1.0


class TestAveragePrecision:
    def test_perfect_ranking(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.3, 0.1])
        assert average_precision(y, scores) == pytest.approx(1.0)

    def test_no_positives(self):
        y = np.array([0, 0, 0])
        scores = np.array([0.9, 0.5, 0.1])
        assert average_precision(y, scores) == 0.0

    def test_result_in_range(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=50)
        s = rng.random(50)
        assert 0.0 <= average_precision(y, s) <= 1.0


class TestFalsePositiveRate:
    def test_all_correct(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 0, 1, 1])
        assert false_positive_rate(y, p) == 0.0

    def test_all_predicted_positive(self):
        y = np.array([0, 0, 1])
        p = np.array([1, 1, 1])
        assert false_positive_rate(y, p) == 1.0

    def test_no_negatives(self):
        y = np.array([1, 1])
        p = np.array([1, 1])
        assert false_positive_rate(y, p) == 0.0


class TestDetectionRate:
    def test_all_caught(self):
        y = np.array([1, 1, 0])
        p = np.array([1, 1, 0])
        assert detection_rate(y, p) == 1.0

    def test_none_caught(self):
        y = np.array([1, 1, 0])
        p = np.array([0, 0, 0])
        assert detection_rate(y, p) == 0.0

    def test_no_positives(self):
        y = np.array([0, 0, 0])
        p = np.array([0, 0, 0])
        assert detection_rate(y, p) == 0.0


class TestMetricsParametrized:
    @pytest.mark.parametrize(
        "k,expected",
        [(1, 0.0), (2, 0.5), (4, 0.5)],
    )
    def test_precision_at_k_parametrized(self, k, expected):
        from utils.metrics import precision_at_k

        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert precision_at_k(y, scores, k) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "frac_fraud",
        [0.0, 0.1, 0.5, 1.0],
    )
    def test_average_precision_bounds(self, frac_fraud):
        from utils.metrics import average_precision

        rng = __import__("numpy").random.default_rng(0)
        n = 100
        n_pos = int(n * frac_fraud)
        y = __import__("numpy").array([1] * n_pos + [0] * (n - n_pos))
        scores = rng.uniform(0, 1, n)
        ap = average_precision(y, scores)
        assert 0.0 <= ap <= 1.0

    @pytest.mark.parametrize("k", [1, 5, 10, 50])
    def test_precision_at_k_never_exceeds_one(self, k):
        from utils.metrics import precision_at_k

        rng = __import__("numpy").random.default_rng(1)
        y = rng.integers(0, 2, 100)
        scores = rng.uniform(0, 1, 100)
        assert 0.0 <= precision_at_k(y, scores, k) <= 1.0
