"""Tests for /health endpoint and API structural guarantees."""
from __future__ import annotations


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_json(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert data is not None

    def test_health_has_status_field(self, client):
        data = client.get("/health").get_json()
        assert "status" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_health_has_models_loaded_field(self, client):
        data = client.get("/health").get_json()
        assert "models_loaded" in data

    def test_health_models_loaded_is_bool(self, client):
        data = client.get("/health").get_json()
        assert isinstance(data["models_loaded"], bool)


class TestVersionEndpointValues:
    def test_version_field_has_semver_format(self, client):
        data = client.get("/version").get_json()
        parts = data["version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_api_field_is_non_empty_string(self, client):
        data = client.get("/version").get_json()
        assert isinstance(data["api"], str)
        assert len(data["api"]) > 0

    def test_python_version_format(self, client):
        data = client.get("/version").get_json()
        pv = data["python_version"]
        assert isinstance(pv, str)
        assert pv[0].isdigit()


class TestModelInfoExtended:
    def test_rag_loaded_is_bool(self, client):
        data = client.get("/model/info").get_json()
        assert isinstance(data["rag_loaded"], bool)

    def test_version_is_semver(self, client):
        data = client.get("/model/info").get_json()
        parts = data["version"].split(".")
        assert len(parts) == 3

    def test_feature_count_non_negative(self, client):
        data = client.get("/model/info").get_json()
        assert data["feature_count"] >= 0
