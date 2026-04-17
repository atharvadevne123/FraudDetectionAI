"""
End-to-end model training script.
Usage:
    python scripts/train.py --data data/raw/transactions.parquet --output models/
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.feature_engineering import TransactionFeatureEngineer
from models.anomaly.anomaly_detector import AnomalyDetector
from models.ensemble.fraud_classifier import FraudEnsemble
from monitoring.drift_monitor import DriftMonitor


def parse_args():
    p = argparse.ArgumentParser(description="Train fraud detection models")
    p.add_argument("--data", type=str, default="data/raw/transactions.parquet",
                   help="Path to labelled transaction parquet file")
    p.add_argument("--output", type=str, default="models/",
                   help="Output directory for model artifacts")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--contamination", type=float, default=0.02,
                   help="Expected fraud rate for anomaly detector")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """Generate synthetic transaction dataset for initial training / demo."""
    rng = np.random.default_rng(2024)
    n_fraud = int(n * 0.02)
    amounts = np.concatenate([
        rng.lognormal(4, 1.2, n - n_fraud),
        rng.lognormal(6.5, 0.8, n_fraud),
    ])
    idx = rng.permutation(n)
    amounts = amounts[idx]
    labels = np.array([0] * (n - n_fraud) + [1] * n_fraud)[idx]

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(n)],
        "user_id": rng.integers(1_000, 50_000, n),
        "amount": amounts.round(2),
        "merchant_category": rng.choice(
            ["retail", "grocery", "crypto", "gambling", "restaurant", "travel"], n
        ),
        "payment_method": rng.choice(["credit", "debit", "wire", "crypto"], n),
        "device_type": rng.choice(["mobile", "desktop", "tablet"], n),
        "channel": rng.choice(["online", "pos", "mobile_app"], n),
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="30s"),
        "account_age_days": rng.integers(1, 3650, n),
        "credit_utilization": rng.uniform(0, 1, n).round(4),
        "prior_fraud_count": rng.choice([0, 0, 0, 0, 1, 2], n),
        "ip_address": [f"192.168.{rng.integers(0,255)}.{rng.integers(0,255)}" for _ in range(n)],
        "device_fingerprint": [f"DEV{rng.integers(1000,9999)}" for _ in range(n)],
        "country": rng.choice(["US", "US", "US", "GB", "CN", "NG"], n),
        "is_fraud": labels,
    })
    return df


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or generate data
    data_path = Path(args.data)
    if data_path.exists():
        logger.info("Loading data from {}", data_path)
        df = pd.read_parquet(data_path)
    else:
        logger.warning("Data file not found. Generating {:,}-row synthetic dataset.", 50_000)
        df = generate_synthetic_data(50_000)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        logger.info("Synthetic data saved → {}", data_path)

    logger.info("Dataset: {:,} rows, fraud rate: {:.2%}", len(df), df["is_fraud"].mean())

    # 2. Feature engineering
    fe = TransactionFeatureEngineer()
    df_feat = fe.fit_transform(df)
    fe_path = output_dir / "feature_engineer.joblib"
    joblib.dump(fe, fe_path)
    logger.success("Feature engineer saved → {}", fe_path)

    # 3. Prepare feature matrix
    exclude = {"is_fraud", "transaction_id", "user_id", "timestamp",
               "merchant_id", "ip_address", "device_fingerprint", "country",
               "merchant_category", "payment_method", "device_type", "channel"}
    feature_cols = [c for c in df_feat.select_dtypes(include="number").columns
                    if c not in exclude]
    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols))
    logger.info("{} feature columns selected.", len(feature_cols))

    X = df_feat[feature_cols].fillna(0)
    y = df_feat["is_fraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # 4. Train anomaly detector (on clean transactions only)
    clean_mask = y_train == 0
    anomaly_det = AnomalyDetector(contamination=args.contamination)
    anomaly_det.fit(X_train[clean_mask])
    anomaly_det.save(output_dir / "anomaly_detector.joblib")
    logger.success("Anomaly detector saved.")

    # Add anomaly scores as a feature
    X_train_aug = X_train.copy()
    X_val_aug = X_val.copy()
    X_train_aug["anomaly_score"] = anomaly_det.score(X_train)
    X_val_aug["anomaly_score"] = anomaly_det.score(X_val)
    augmented_cols = feature_cols + ["anomaly_score"]
    (output_dir / "feature_cols_augmented.json").write_text(json.dumps(augmented_cols))

    # 5. Train ensemble
    ensemble = FraudEnsemble()
    ensemble.fit(
        X_train_aug,
        y_train,
        eval_X=X_val_aug,
        eval_y=y_val,
        mlflow_run=not args.no_mlflow,
    )
    ensemble.save(output_dir / "fraud_ensemble.joblib")
    logger.success("Ensemble model saved.")

    # 6. Set drift reference
    val_results = X_val_aug.copy()
    val_proba = ensemble.predict_proba(X_val_aug)[:, 1]
    val_results["fraud_score"] = val_proba
    val_results["anomaly_score"] = X_val_aug["anomaly_score"]
    val_results["is_fraud"] = y_val.values
    monitor = DriftMonitor()
    monitor.set_reference(val_results)
    logger.success("Drift reference distribution saved.")

    logger.success("Training complete. Artifacts saved to {}", output_dir)


if __name__ == "__main__":
    main()
