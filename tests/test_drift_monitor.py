"""Tests for the DriftMonitor class."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Stub heavy optional dependencies so tests run without evidently installed
for mod_name in ["evidently", "evidently.metrics", "evidently.report"]:
    if mod_name not in sys.modules:
        stub = ModuleType(mod_name)
        stub.ColumnMapping = MagicMock()
        stub.DataDriftTable = MagicMock()
        stub.DatasetDriftMetric = MagicMock()
        stub.DatasetMissingValuesMetric = MagicMock()
        stub.ClassificationQualityMetric = MagicMock()
        stub.Report = MagicMock()
        sys.modules[mod_name] = stub


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "amount": rng.exponential(200, 100).tolist(),
        "credit_utilization": rng.uniform(0, 1, 100).tolist(),
        "account_age_days": rng.integers(1, 3650, 100).tolist(),
        "fraud_score": rng.uniform(0, 1, 100).tolist(),
    })


@pytest.fixture()
def monitor(tmp_path):
    with patch("monitoring.drift_monitor.REFERENCE_PATH", tmp_path / "ref.parquet"), \
         patch("monitoring.drift_monitor.MONITOR_DIR", tmp_path):
        from monitoring.drift_monitor import DriftMonitor
        return DriftMonitor(drift_threshold=0.15, score_shift_threshold=0.05)


class TestDriftMonitorSetReference:
    def test_set_reference_stores_df(self, monitor, sample_df):
        monitor.set_reference(sample_df)
        assert monitor._reference is not None
        assert len(monitor._reference) == len(sample_df)

    def test_set_reference_does_not_mutate_input(self, monitor, sample_df):
        original_len = len(sample_df)
        monitor.set_reference(sample_df)
        assert len(sample_df) == original_len


class TestDriftMonitorRun:
    def test_run_without_reference_sets_reference(self, monitor, sample_df):
        result = monitor.run(sample_df)
        assert result["drift_detected"] is False
        assert "reason" in result

    def test_run_returns_required_keys(self, monitor, sample_df):
        monitor.set_reference(sample_df)
        result = monitor.run(sample_df)
        for key in ("run_timestamp", "batch_size", "drift_detected", "drifted_features"):
            assert key in result

    def test_run_batch_size_matches_input(self, monitor, sample_df):
        monitor.set_reference(sample_df)
        result = monitor.run(sample_df)
        assert result["batch_size"] == len(sample_df)


class TestDriftMonitorThresholds:
    def test_default_drift_threshold(self, monitor):
        assert monitor.drift_threshold == 0.15

    def test_custom_thresholds(self, tmp_path):
        with patch("monitoring.drift_monitor.REFERENCE_PATH", tmp_path / "ref.parquet"), \
             patch("monitoring.drift_monitor.MONITOR_DIR", tmp_path):
            from monitoring.drift_monitor import DriftMonitor
            m = DriftMonitor(drift_threshold=0.25, score_shift_threshold=0.10)
        assert m.drift_threshold == 0.25
        assert m.score_shift_threshold == 0.10
