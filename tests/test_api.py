from __future__ import annotations

import json


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
