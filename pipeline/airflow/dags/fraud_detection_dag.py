"""
Airflow DAG: Real-Time Fraud Detection Pipeline

Schedule: Every 15 minutes (near real-time batch micro-processing)
Flow:
  ingest_transactions → feature_engineering → anomaly_scoring
      → ensemble_scoring → rag_explanation → store_results → drift_check
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import boto3
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from loguru import logger


# ---------------------------------------------------------------------------
# DAG defaults
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "fraud-ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=12),
}

S3_BUCKET = os.getenv("S3_BUCKET", "fraud-detection-bucket")
S3_RAW_PREFIX = "transactions/raw/"
S3_PROCESSED_PREFIX = "transactions/processed/"
S3_RESULTS_PREFIX = "transactions/results/"
ENSEMBLE_ARTIFACT = "/opt/airflow/models/fraud_ensemble.joblib"
ANOMALY_ARTIFACT = "/opt/airflow/models/anomaly_detector.joblib"
FEATURE_COLS_PATH = "/opt/airflow/models/feature_cols.json"


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="fraud_detection_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end fraud detection: ingest → features → score → explain → store",
    schedule_interval="*/15 * * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "ml", "production"],
) as dag:

    # -----------------------------------------------------------------------
    # Task 1: Ingest raw transactions from S3
    # -----------------------------------------------------------------------

    @task(task_id="ingest_transactions")
    def ingest_transactions(**context) -> str:
        """Pull the latest transaction batch from S3 into /tmp."""
        s3 = boto3.client("s3")
        execution_date = context["execution_date"].strftime("%Y%m%d_%H%M")
        key = f"{S3_RAW_PREFIX}batch_{execution_date}.parquet"
        local_path = f"/tmp/raw_batch_{execution_date}.parquet"

        try:
            s3.download_file(S3_BUCKET, key, local_path)
            df = pd.read_parquet(local_path)
            logger.info("Ingested {:,} transactions from s3://{}/{}", len(df), S3_BUCKET, key)
        except Exception:
            logger.warning("S3 key {} not found. Generating synthetic batch for demo.", key)
            df = _generate_synthetic_batch()
            df.to_parquet(local_path, index=False)

        return local_path

    # -----------------------------------------------------------------------
    # Task 2: Feature engineering
    # -----------------------------------------------------------------------

    @task(task_id="feature_engineering")
    def feature_engineering(raw_path: str) -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from pipeline.feature_engineering import TransactionFeatureEngineer
        import joblib

        df = pd.read_parquet(raw_path)
        fe_path = "/opt/airflow/models/feature_engineer.joblib"

        if Path(fe_path).exists():
            fe = joblib.load(fe_path)
        else:
            fe = TransactionFeatureEngineer()
            fe.fit(df)
            joblib.dump(fe, fe_path)

        df_feat = fe.transform(df)
        out_path = raw_path.replace("raw_batch", "feat_batch")
        df_feat.to_parquet(out_path, index=False)
        logger.info("Feature engineering done: {} columns.", df_feat.shape[1])
        return out_path

    # -----------------------------------------------------------------------
    # Task 3: Anomaly scoring
    # -----------------------------------------------------------------------

    @task(task_id="anomaly_scoring")
    def anomaly_scoring(feat_path: str) -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from models.anomaly.anomaly_detector import AnomalyDetector

        df = pd.read_parquet(feat_path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        exclude = ["is_fraud", "transaction_id"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

        if Path(ANOMALY_ARTIFACT).exists():
            detector = AnomalyDetector.load(ANOMALY_ARTIFACT)
        else:
            detector = AnomalyDetector(contamination=0.02)
            detector.fit(df[feature_cols].fillna(0))
            detector.save(ANOMALY_ARTIFACT)

        result = detector.score_and_predict(df[feature_cols].fillna(0))
        df["anomaly_score"] = result["anomaly_score"].values
        df["is_anomaly"] = result["is_anomaly"].values

        out_path = feat_path.replace("feat_batch", "anomaly_batch")
        df.to_parquet(out_path, index=False)
        logger.info("Anomaly scoring done. Anomalies detected: {:,}.", df["is_anomaly"].sum())
        return out_path

    # -----------------------------------------------------------------------
    # Task 4: Ensemble fraud scoring
    # -----------------------------------------------------------------------

    @task(task_id="ensemble_scoring")
    def ensemble_scoring(anomaly_path: str) -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from models.ensemble.fraud_classifier import FraudEnsemble

        df = pd.read_parquet(anomaly_path)
        exclude = {"is_fraud", "transaction_id", "user_id", "timestamp",
                   "merchant_id", "ip_address", "device_fingerprint", "country"}

        if Path(FEATURE_COLS_PATH).exists():
            feature_cols = json.loads(Path(FEATURE_COLS_PATH).read_text())
        else:
            feature_cols = [c for c in df.select_dtypes(include="number").columns
                            if c not in exclude]

        if Path(ENSEMBLE_ARTIFACT).exists():
            model = FraudEnsemble.load(ENSEMBLE_ARTIFACT)
        else:
            raise FileNotFoundError(
                f"Ensemble model not found at {ENSEMBLE_ARTIFACT}. "
                "Run scripts/train.py first."
            )

        proba = model.predict_proba(df[feature_cols].fillna(0))[:, 1]
        df["fraud_score"] = proba
        df["fraud_label"] = (proba >= 0.5).astype(int)

        out_path = anomaly_path.replace("anomaly_batch", "scored_batch")
        df.to_parquet(out_path, index=False)
        high_risk = (proba >= 0.7).sum()
        logger.info("Scoring done. High-risk transactions: {:,}.", high_risk)
        return out_path

    # -----------------------------------------------------------------------
    # Task 5: RAG explanation for flagged transactions
    # -----------------------------------------------------------------------

    @task(task_id="rag_explanation")
    def rag_explanation(scored_path: str) -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from models.rag.rag_explainer import RAGExplainer
        from models.ensemble.fraud_classifier import FraudEnsemble

        df = pd.read_parquet(scored_path)
        flagged = df[df["fraud_score"] >= 0.5].copy()

        if len(flagged) == 0:
            logger.info("No flagged transactions — skipping RAG.")
            out_path = scored_path.replace("scored_batch", "explained_batch")
            df["explanation"] = ""
            df.to_parquet(out_path, index=False)
            return out_path

        explainer = RAGExplainer()
        model = FraudEnsemble.load(ENSEMBLE_ARTIFACT) if Path(ENSEMBLE_ARTIFACT).exists() else None

        explanations = {}
        for _, row in flagged.head(50).iterrows():  # cap at 50 to stay within LLM budget
            txn = row.to_dict()
            shap_contribs = {}
            if model and model._shap_explainer:
                try:
                    feature_cols = json.loads(Path(FEATURE_COLS_PATH).read_text())
                    explained = model.explain(pd.DataFrame([row[feature_cols]]))
                    shap_contribs = explained[0].get("shap_values", {}) if explained else {}
                except Exception:
                    pass
            explanation = explainer.explain(
                transaction=txn,
                fraud_score=row.get("fraud_score", 0),
                anomaly_score=row.get("anomaly_score", 0),
                shap_contributions=shap_contribs,
            )
            explanations[row.get("transaction_id", str(_))] = explanation

        df["explanation"] = df.get("transaction_id", df.index.astype(str)).map(explanations).fillna("")
        out_path = scored_path.replace("scored_batch", "explained_batch")
        df.to_parquet(out_path, index=False)
        logger.info("RAG explanations generated for {:,} transactions.", len(explanations))
        return out_path

    # -----------------------------------------------------------------------
    # Task 6: Store results to S3
    # -----------------------------------------------------------------------

    @task(task_id="store_results")
    def store_results(explained_path: str, **context) -> None:
        s3 = boto3.client("s3")
        execution_date = context["execution_date"].strftime("%Y%m%d_%H%M")
        key = f"{S3_RESULTS_PREFIX}results_{execution_date}.parquet"
        s3.upload_file(explained_path, S3_BUCKET, key)
        logger.info("Results uploaded → s3://{}/{}", S3_BUCKET, key)

    # -----------------------------------------------------------------------
    # Task 7: Drift check
    # -----------------------------------------------------------------------

    @task(task_id="drift_check")
    def drift_check(explained_path: str) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from monitoring.drift_monitor import DriftMonitor

        df = pd.read_parquet(explained_path)
        monitor = DriftMonitor()
        report = monitor.run(df)
        if report.get("drift_detected"):
            logger.warning("DATA DRIFT DETECTED! Metrics: {}", report)
        else:
            logger.info("No significant drift. Score distribution stable.")

    # -----------------------------------------------------------------------
    # Wire tasks
    # -----------------------------------------------------------------------

    raw_path = ingest_transactions()
    feat_path = feature_engineering(raw_path)
    anomaly_path = anomaly_scoring(feat_path)
    scored_path = ensemble_scoring(anomaly_path)
    explained_path = rag_explanation(scored_path)
    store_results(explained_path)
    drift_check(explained_path)


# ---------------------------------------------------------------------------
# Synthetic data helper (demo/testing)
# ---------------------------------------------------------------------------

def _generate_synthetic_batch(n: int = 500) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    n_fraud = int(n * 0.02)
    amounts = np.concatenate([
        rng.lognormal(4, 1.2, n - n_fraud),
        rng.lognormal(6, 0.5, n_fraud),
    ])
    rng.shuffle(amounts)
    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:06d}" for i in range(n)],
        "user_id": rng.integers(1000, 9999, n),
        "amount": amounts.round(2),
        "merchant_category": rng.choice(
            ["retail", "grocery", "crypto", "gambling", "restaurant"], n
        ),
        "payment_method": rng.choice(["credit", "debit", "wire"], n),
        "device_type": rng.choice(["mobile", "desktop", "tablet"], n),
        "channel": rng.choice(["online", "pos", "mobile_app"], n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="30s"),
        "account_age_days": rng.integers(1, 3650, n),
        "credit_utilization": rng.uniform(0, 1, n).round(4),
        "prior_fraud_count": rng.choice([0, 0, 0, 0, 1, 2], n),
        "is_fraud": ([0] * (n - n_fraud) + [1] * n_fraud),
    })
    return df
