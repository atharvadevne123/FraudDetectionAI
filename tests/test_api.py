from __future__ import annotations

import json

import pytest


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_models_not_loaded(self, client):
        r = client.get("/health")
        assert r.get_json()["models_loaded"] is False


class TestModelInfo:
    def test_model_info_structure(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.get_json()
        for key in ("ensemble_loaded", "anomaly_loaded", "rag_loaded", "feature_count", "version"):
            assert key in data

    def test_model_info_version(self, client):
        r = client.get("/model/info")
        assert r.get_json()["version"] == "1.0.0"


class TestPredict:
    def test_predict_valid(self, client, patch_models, valid_transaction):
        r = client.post("/predict", data=json.dumps(valid_transaction),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "fraud_score" in data
        assert "risk_tier" in data
        assert 0.0 <= data["fraud_score"] <= 1.0
        assert data["risk_tier"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN")

    def test_predict_missing_user_id(self, client, patch_models):
        r = client.post("/predict", data=json.dumps({"amount": 100.0}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_amount(self, client, patch_models):
        r = client.post("/predict", data=json.dumps({"user_id": 1}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_amount_zero_rejected(self, client, patch_models):
        r = client.post("/predict", data=json.dumps({"user_id": 1, "amount": 0.0}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_amount_negative_rejected(self, client, patch_models):
        r = client.post("/predict", data=json.dumps({"user_id": 1, "amount": -50.0}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_credit_util_out_of_range(self, client, patch_models):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": 100.0, "credit_utilization": 1.5}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_account_age_negative(self, client, patch_models):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": 100.0, "account_age_days": -1}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_predict_response_has_request_id_header(self, client, patch_models, valid_transaction):
        r = client.post("/predict", data=json.dumps(valid_transaction),
                        content_type="application/json")
        assert "X-Request-ID" in r.headers

    def test_predict_empty_body_returns_400(self, client, patch_models):
        r = client.post("/predict", data="", content_type="application/json")
        assert r.status_code == 400


class TestPredictBatch:
    def test_batch_valid(self, client, patch_models, valid_transaction):
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": [valid_transaction, valid_transaction]}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_transactions(self, client, patch_models):
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": []}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_batch_missing_key(self, client, patch_models):
        r = client.post("/predict/batch", data=json.dumps({}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_batch_over_limit(self, client, patch_models, valid_transaction):
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": [valid_transaction] * 501}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_batch_invalid_txn_in_list(self, client, patch_models):
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": [{"amount": 100}]}),
                        content_type="application/json")
        assert r.status_code == 400


class TestMetrics:
    def test_metrics_returns_prometheus_format(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert b"fraud_api" in r.data or b"#" in r.data


class TestFeedback:
    def test_feedback_valid(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl"))
        import api.app as app_module
        monkeypatch.setattr(app_module, "FEEDBACK_LOG_PATH",
                            __import__("pathlib").Path(str(tmp_path / "feedback.jsonl")))
        r = client.post("/feedback",
                        data=json.dumps({
                            "transaction_id": "TXN-0001-AA",
                            "predicted_tier": "HIGH",
                            "actual_label": 1,
                            "analyst_id": "analyst@example.com",
                        }),
                        content_type="application/json")
        assert r.status_code == 200
        assert r.get_json()["status"] == "logged"

    def test_feedback_missing_transaction_id(self, client):
        r = client.post("/feedback",
                        data=json.dumps({"predicted_tier": "HIGH", "actual_label": 1}),
                        content_type="application/json")
        assert r.status_code == 400


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        r = client.get("/nonexistent-endpoint-xyz")
        assert r.status_code == 404
        data = r.get_json()
        assert "error" in data


class TestPredictParametrized:
    @pytest.mark.parametrize("amount", [0.01, 100.0, 9999.99, 1_000_000.0])
    def test_predict_valid_amounts(self, client, patch_models, amount):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": amount}),
                        content_type="application/json")
        assert r.status_code == 200

    @pytest.mark.parametrize("invalid_amount", [0.0, -1.0, -0.01])
    def test_predict_invalid_amounts_rejected(self, client, patch_models, invalid_amount):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": invalid_amount}),
                        content_type="application/json")
        assert r.status_code == 400

    @pytest.mark.parametrize("credit_util", [0.0, 0.5, 1.0])
    def test_predict_valid_credit_utilization(self, client, patch_models, credit_util):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": 100.0, "credit_utilization": credit_util}),
                        content_type="application/json")
        assert r.status_code == 200

    @pytest.mark.parametrize("channel", ["online", "in-store", "mobile", "atm"])
    def test_predict_various_channels(self, client, patch_models, channel):
        r = client.post("/predict",
                        data=json.dumps({"user_id": 1, "amount": 50.0, "channel": channel}),
                        content_type="application/json")
        assert r.status_code == 200

    def test_predict_response_has_all_fields(self, client, patch_models, valid_transaction):
        r = client.post("/predict", data=json.dumps(valid_transaction),
                        content_type="application/json")
        data = r.get_json()
        for field in ("fraud_score", "anomaly_score", "fraud_label", "risk_tier", "latency_ms"):
            assert field in data

    def test_predict_fraud_label_is_binary(self, client, patch_models, valid_transaction):
        r = client.post("/predict", data=json.dumps(valid_transaction),
                        content_type="application/json")
        data = r.get_json()
        assert data["fraud_label"] in (0, 1)


class TestBatchPredictParametrized:
    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_batch_various_sizes(self, client, patch_models, valid_transaction, batch_size):
        txns = [valid_transaction] * batch_size
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": txns}),
                        content_type="application/json")
        assert r.status_code == 200
        assert r.get_json()["count"] == batch_size

    def test_batch_exactly_at_limit(self, client, patch_models, valid_transaction):
        txns = [valid_transaction] * 500
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": txns}),
                        content_type="application/json")
        assert r.status_code == 200

    def test_batch_one_over_limit_rejected(self, client, patch_models, valid_transaction):
        txns = [valid_transaction] * 501
        r = client.post("/predict/batch",
                        data=json.dumps({"transactions": txns}),
                        content_type="application/json")
        assert r.status_code == 400


class TestVersionAndReadinessEndpoints:
    def test_version_endpoint_returns_200(self, client):
        r = client.get("/version")
        assert r.status_code == 200

    def test_version_contains_version_key(self, client):
        r = client.get("/version")
        assert "version" in r.get_json()

    def test_readiness_returns_json(self, client):
        r = client.get("/readiness")
        assert r.is_json

    def test_readiness_has_ready_key(self, client):
        r = client.get("/readiness")
        assert "ready" in r.get_json()

    @pytest.mark.parametrize("endpoint", ["/health", "/version", "/readiness"])
    def test_get_endpoints_respond(self, client, endpoint):
        r = client.get(endpoint)
        assert r.status_code in (200, 503)


class TestPredictInputValidation:
    @pytest.mark.parametrize(
        "payload,expected_status",
        [
            ({"user_id": 1, "amount": 100.0}, 200),  # minimal valid (merchant_category, etc. have defaults)
            ({}, 400),  # missing required fields
            ({"user_id": 1, "amount": -5.0}, 400),  # negative amount
            ({"user_id": 1, "amount": 100.0, "credit_utilization": 2.0}, 400),  # util out of range
        ],
    )
    def test_predict_input_validation(self, client, patch_models, payload, expected_status):
        r = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert r.status_code == expected_status

    def test_predict_missing_user_id_returns_400(self, client, patch_models):
        r = client.post(
            "/predict",
            data=json.dumps({"amount": 200.0}),
            content_type="application/json",
        )
        assert r.status_code == 400
