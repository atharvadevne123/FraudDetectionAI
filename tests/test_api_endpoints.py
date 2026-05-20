"""Tests for /version, /readiness, and model info endpoints."""
from __future__ import annotations

import pytest


class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        r = client.get("/version")
        assert r.status_code == 200

    def test_version_has_version_field(self, client):
        data = client.get("/version").get_json()
        assert "version" in data

    def test_version_has_api_field(self, client):
        data = client.get("/version").get_json()
        assert "api" in data

    def test_version_has_python_version(self, client):
        data = client.get("/version").get_json()
        assert "python_version" in data

    def test_version_is_string(self, client):
        data = client.get("/version").get_json()
        assert isinstance(data["version"], str)


class TestReadinessEndpoint:
    def test_readiness_returns_dict(self, client):
        r = client.get("/readiness")
        assert r.status_code in (200, 503)
        data = r.get_json()
        assert "ready" in data

    def test_readiness_models_not_loaded_is_not_ready(self, client):
        r = client.get("/readiness")
        data = r.get_json()
        assert data["ready"] is False
        assert r.status_code == 503

    def test_readiness_with_mock_models(self, client, patch_models):
        r = client.get("/readiness")
        data = r.get_json()
        assert data["ready"] is True
        assert r.status_code == 200


class TestModelInfoEndpoint:
    @pytest.mark.parametrize("key", ["ensemble_loaded", "anomaly_loaded", "rag_loaded", "feature_count", "version"])
    def test_model_info_has_key(self, client, key):
        data = client.get("/model/info").get_json()
        assert key in data

    def test_model_info_feature_count_is_int(self, client):
        data = client.get("/model/info").get_json()
        assert isinstance(data["feature_count"], int)

    def test_model_info_no_models_loaded(self, client):
        data = client.get("/model/info").get_json()
        assert data["ensemble_loaded"] is False
        assert data["anomaly_loaded"] is False
