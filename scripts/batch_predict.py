"""
Batch prediction script for offline scoring of transaction datasets.
Reads input from CSV/Parquet, writes scored output with risk tiers.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.constants import MAX_BATCH_SIZE, RISK_TIERS  # noqa: E402


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Batch-score transactions with the fraud model")
    p.add_argument("--input", required=True, help="Input CSV or Parquet file")
    p.add_argument("--output", required=True, help="Output CSV path for scored results")
    p.add_argument("--ensemble", default="models/fraud_ensemble.joblib")
    p.add_argument("--anomaly", default="models/anomaly_detector.joblib")
    p.add_argument("--feature-engineer", default="models/feature_engineer.joblib")
    p.add_argument(
        "--chunk-size", type=int, default=MAX_BATCH_SIZE, help="Process this many rows at a time"
    )
    return p.parse_args()


def assign_risk_tier(score: float) -> str:
    """Map a fraud probability score to a named risk tier."""
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score < hi:
            return tier
    return "CLEAN"


def load_artifacts(args: Namespace):
    import joblib

    fe = joblib.load(args.feature_engineer)
    ensemble = joblib.load(args.ensemble)
    anomaly = joblib.load(args.anomaly)
    return fe, ensemble, anomaly


def score_chunk(chunk: pd.DataFrame, fe, ensemble, anomaly) -> pd.DataFrame:
    df_feat = fe.transform(chunk)
    feature_cols = [c for c in df_feat.columns if c not in ("is_fraud", "user_id", "timestamp")]
    X = df_feat[feature_cols].fillna(0).values
    anomaly_scores = anomaly.score(X)
    X_full = np.column_stack([X, anomaly_scores])
    scores = ensemble.predict_proba(X_full)
    chunk = chunk.copy()
    chunk["fraud_score"] = scores
    chunk["anomaly_score"] = anomaly_scores
    chunk["risk_tier"] = [assign_risk_tier(s) for s in scores]
    return chunk


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input file not found: {}", args.input)
        sys.exit(1)

    logger.info("Loading artifacts…")
    try:
        fe, ensemble, anomaly = load_artifacts(args)
    except FileNotFoundError as e:
        logger.error("Artifact not found: {}", e)
        sys.exit(1)

    logger.info("Reading input: {}", args.input)
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    logger.info("Scoring {:,} transactions in chunks of {}", len(df), args.chunk_size)

    chunks = []
    for start in range(0, len(df), args.chunk_size):
        chunk = df.iloc[start : start + args.chunk_size]
        scored = score_chunk(chunk, fe, ensemble, anomaly)
        chunks.append(scored)
        logger.info("Scored rows {}-{}", start, min(start + args.chunk_size, len(df)))

    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(args.output, index=False)

    tier_counts = result["risk_tier"].value_counts()
    logger.info("Done. Results written to {}", args.output)
    logger.info("Risk tier distribution:\n{}", tier_counts.to_string())


if __name__ == "__main__":
    main()
