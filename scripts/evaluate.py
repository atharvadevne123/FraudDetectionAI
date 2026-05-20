"""
Offline evaluation script for the fraud detection ensemble.
Loads saved model artifacts and computes metrics against a held-out test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble.fraud_classifier import FraudEnsemble
from models.anomaly.anomaly_detector import AnomalyDetector
from pipeline.feature_engineering import TransactionFeatureEngineer
from utils.metrics import precision_at_k, average_precision, detection_rate, false_positive_rate


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved fraud detection models")
    p.add_argument("--test-data", required=True, help="Path to test CSV or Parquet")
    p.add_argument("--ensemble", default="models/fraud_ensemble.joblib")
    p.add_argument("--anomaly", default="models/anomaly_detector.joblib")
    p.add_argument("--feature-engineer", default="models/feature_engineer.joblib")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    p.add_argument("--top-k", type=int, default=100, help="k for Precision@k")
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    args = parse_args()

    logger.info("Loading test data from {}", args.test_data)
    df = load_data(args.test_data)

    if "is_fraud" not in df.columns:
        logger.error("Test data must contain an 'is_fraud' column.")
        sys.exit(1)

    y_true = df["is_fraud"].values

    logger.info("Loading artifacts…")
    fe: TransactionFeatureEngineer = TransactionFeatureEngineer.load(args.feature_engineer) \
        if hasattr(TransactionFeatureEngineer, "load") else TransactionFeatureEngineer()

    fe_path = Path(args.feature_engineer)
    if fe_path.exists():
        import joblib
        fe = joblib.load(fe_path)
    else:
        logger.warning("Feature engineer artifact not found; using unfitted instance.")

    ensemble = FraudEnsemble.load(args.ensemble)
    anomaly = AnomalyDetector.load(args.anomaly)

    logger.info("Engineering features…")
    df_feat = fe.transform(df)
    feature_cols = [c for c in df_feat.columns if c not in ("is_fraud", "user_id", "timestamp")]
    X = df_feat[feature_cols].fillna(0).values

    anomaly_scores = anomaly.score(X)
    X_full = np.column_stack([X, anomaly_scores])

    scores = ensemble.predict_proba(X_full)
    y_pred = (scores >= args.threshold).astype(int)

    ap = average_precision(y_true, scores)
    pak = precision_at_k(y_true, scores, k=args.top_k)
    dr = detection_rate(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)

    fraud_count = int(y_true.sum())
    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info("Test samples : {:,}", len(y_true))
    logger.info("Fraud samples: {:,} ({:.1%})", fraud_count, fraud_count / len(y_true))
    logger.info("Threshold    : {:.2f}", args.threshold)
    logger.info("Avg Precision: {:.4f}", ap)
    logger.info("P@{}          : {:.4f}", args.top_k, pak)
    logger.info("Detection Rate: {:.4f}", dr)
    logger.info("False Pos Rate: {:.4f}", fpr)


if __name__ == "__main__":
    main()
