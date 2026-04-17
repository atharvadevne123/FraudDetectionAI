"""
Flask microservice: Fraud Detection API
Endpoints:
  POST /predict         — score a single transaction
  POST /predict/batch   — score a batch of transactions
  GET  /health          — liveness probe
  GET  /metrics         — Prometheus metrics
  GET  /model/info      — model metadata
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from loguru import logger
from marshmallow import Schema, fields as ma_fields, ValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ─── App bootstrap ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version="1.0",
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring with explainable AI",
    doc="/docs",
)

ns = api.namespace("", description="Fraud detection endpoints")

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter("fraud_api_requests_total", "Total API requests", ["endpoint", "status"])
PREDICTION_LATENCY = Histogram(
    "fraud_api_prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
FRAUD_SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))
_ensemble = None
_anomaly_detector = None
_feature_engineer = None
_rag_explainer = None
_feature_cols: list[str] = []


def _load_models():
    global _ensemble, _anomaly_detector, _feature_engineer, _rag_explainer, _feature_cols
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import joblib
    from models.ensemble.fraud_classifier import FraudEnsemble
    from models.anomaly.anomaly_detector import AnomalyDetector
    from models.rag.rag_explainer import RAGExplainer
    from pipeline.feature_engineering import TransactionFeatureEngineer

    ensemble_path = MODEL_DIR / "fraud_ensemble.joblib"
    anomaly_path = MODEL_DIR / "anomaly_detector.joblib"
    fe_path = MODEL_DIR / "feature_engineer.joblib"
    cols_path = MODEL_DIR / "feature_cols_augmented.json"
    if not cols_path.exists():
        cols_path = MODEL_DIR / "feature_cols.json"

    if ensemble_path.exists():
        _ensemble = FraudEnsemble.load(ensemble_path)
        logger.success("Ensemble model loaded.")
    else:
        logger.warning("Ensemble model not found at {}. Returning mock scores.", ensemble_path)

    if anomaly_path.exists():
        _anomaly_detector = AnomalyDetector.load(anomaly_path)
        logger.success("Anomaly detector loaded.")

    if fe_path.exists():
        _feature_engineer = joblib.load(fe_path)
        logger.success("Feature engineer loaded.")

    if cols_path.exists():
        _feature_cols = json.loads(cols_path.read_text())

    _rag_explainer = RAGExplainer()
    logger.success("RAG explainer initialised.")


# ─── Input validation ─────────────────────────────────────────────────────────

class TransactionSchema(Schema):
    transaction_id = ma_fields.Str(load_default=None)
    user_id = ma_fields.Int(required=True)
    amount = ma_fields.Float(required=True)
    merchant_category = ma_fields.Str(load_default="unknown")
    payment_method = ma_fields.Str(load_default="unknown")
    device_type = ma_fields.Str(load_default="unknown")
    channel = ma_fields.Str(load_default="online")
    timestamp = ma_fields.Str(load_default=None)
    account_age_days = ma_fields.Int(load_default=365)
    credit_utilization = ma_fields.Float(load_default=0.3)
    prior_fraud_count = ma_fields.Int(load_default=0)
    explain = ma_fields.Bool(load_default=False)


_schema = TransactionSchema()
_batch_schema = TransactionSchema(many=True)

# ─── Swagger models ───────────────────────────────────────────────────────────

transaction_model = api.model("Transaction", {
    "user_id": fields.Integer(required=True, example=12345),
    "amount": fields.Float(required=True, example=4999.99),
    "merchant_category": fields.String(example="crypto"),
    "payment_method": fields.String(example="credit"),
    "device_type": fields.String(example="mobile"),
    "channel": fields.String(example="online"),
    "timestamp": fields.String(example="2024-01-15T02:30:00"),
    "account_age_days": fields.Integer(example=180),
    "credit_utilization": fields.Float(example=0.87),
    "prior_fraud_count": fields.Integer(example=0),
    "explain": fields.Boolean(example=True),
})

prediction_response = api.model("PredictionResponse", {
    "transaction_id": fields.String(),
    "fraud_score": fields.Float(),
    "anomaly_score": fields.Float(),
    "fraud_label": fields.Integer(),
    "risk_tier": fields.String(),
    "explanation": fields.String(),
    "latency_ms": fields.Float(),
})


# ─── Core scoring logic ───────────────────────────────────────────────────────

def _score_transaction(txn: dict) -> dict:
    """Run anomaly + ensemble scoring and optionally RAG explanation."""
    start = time.perf_counter()
    df = pd.DataFrame([txn])

    # Feature engineering
    if _feature_engineer is not None:
        df = _feature_engineer.transform(df)

    # Base feature columns (without anomaly_score)
    base_cols = [c for c in (_feature_cols or []) if c != "anomaly_score"]
    if base_cols:
        X_base = pd.DataFrame(0.0, index=df.index, columns=base_cols)
        for col in base_cols:
            if col in df.columns:
                X_base[col] = df[col].fillna(0)
    else:
        exclude = {"is_fraud", "transaction_id", "user_id", "timestamp",
                   "merchant_id", "ip_address", "device_fingerprint", "country", "explain"}
        feat_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        X_base = df[feat_cols].fillna(0)

    # Anomaly score (computed on base features)
    anomaly_score = 0.0
    if _anomaly_detector is not None:
        anomaly_score = float(_anomaly_detector.score(X_base)[0])

    # Augment with anomaly_score if ensemble expects it
    if _feature_cols and "anomaly_score" in _feature_cols:
        X = X_base.copy()
        X["anomaly_score"] = anomaly_score
        X = X[_feature_cols]  # enforce correct column order
    else:
        X = X_base

    # Fraud score
    fraud_score = 0.0
    if _ensemble is not None:
        fraud_score = float(_ensemble.predict_proba(X)[0, 1])
    else:
        import random
        fraud_score = round(random.uniform(0.01, 0.95), 4)

    # Risk tier
    if fraud_score >= 0.9:
        risk_tier = "CRITICAL"
    elif fraud_score >= 0.7:
        risk_tier = "HIGH"
    elif fraud_score >= 0.5:
        risk_tier = "MEDIUM"
    elif fraud_score >= 0.3:
        risk_tier = "LOW"
    else:
        risk_tier = "CLEAN"

    # SHAP + RAG explanation
    explanation = ""
    if txn.get("explain") and _rag_explainer is not None:
        shap_contribs = {}
        if _ensemble and _ensemble._shap_explainer:
            try:
                shap_result = _ensemble.explain(X)
                shap_contribs = shap_result[0].get("shap_values", {}) if shap_result else {}
            except Exception:
                pass
        explanation = _rag_explainer.explain(
            transaction=txn,
            fraud_score=fraud_score,
            anomaly_score=anomaly_score,
            shap_contributions=shap_contribs,
        )

    latency_ms = (time.perf_counter() - start) * 1000
    FRAUD_SCORE_HISTOGRAM.observe(fraud_score)

    return {
        "transaction_id": txn.get("transaction_id"),
        "fraud_score": round(fraud_score, 6),
        "anomaly_score": round(anomaly_score, 6),
        "fraud_label": int(fraud_score >= 0.5),
        "risk_tier": risk_tier,
        "explanation": explanation,
        "latency_ms": round(latency_ms, 2),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {"status": "ok", "models_loaded": _ensemble is not None}, 200


@ns.route("/metrics")
class Metrics(Resource):
    def get(self):
        from flask import Response
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@ns.route("/model/info")
class ModelInfo(Resource):
    def get(self):
        return {
            "ensemble_loaded": _ensemble is not None,
            "anomaly_loaded": _anomaly_detector is not None,
            "rag_loaded": _rag_explainer is not None,
            "feature_count": len(_feature_cols),
            "version": "1.0.0",
        }


@ns.route("/predict")
class Predict(Resource):
    @ns.expect(transaction_model)
    @ns.marshal_with(prediction_response)
    def post(self):
        try:
            data = _schema.load(request.get_json(force=True) or {})
        except ValidationError as e:
            REQUEST_COUNT.labels(endpoint="predict", status="400").inc()
            return {"error": str(e.messages)}, 400

        with PREDICTION_LATENCY.time():
            result = _score_transaction(data)

        REQUEST_COUNT.labels(endpoint="predict", status="200").inc()
        return result, 200


@ns.route("/predict/batch")
class PredictBatch(Resource):
    def post(self):
        payload = request.get_json(force=True) or {}
        transactions = payload.get("transactions", [])
        if not transactions:
            return {"error": "No transactions provided."}, 400
        if len(transactions) > 500:
            return {"error": "Batch size limited to 500."}, 400

        try:
            data_list = _batch_schema.load(transactions)
        except ValidationError as e:
            REQUEST_COUNT.labels(endpoint="predict_batch", status="400").inc()
            return {"error": str(e.messages)}, 400

        results = []
        with PREDICTION_LATENCY.time():
            for txn in data_list:
                results.append(_score_transaction(txn))

        REQUEST_COUNT.labels(endpoint="predict_batch", status="200").inc()
        return {"results": results, "count": len(results)}, 200


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
