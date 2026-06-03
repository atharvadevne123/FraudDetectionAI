"""
Quickly generate a synthetic labelled transaction dataset for development/testing.
Usage: python scripts/generate_synthetic_data.py --rows 100000 --out data/raw/transactions.parquet
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import generate_synthetic_data


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic fraud transaction dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rows", type=int, default=100_000, help="Number of transactions to generate")
    p.add_argument(
        "--out",
        type=str,
        default="data/raw/transactions.parquet",
        help="Output file path (.parquet or .csv)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    """Generate a synthetic labelled transaction dataset and write it to disk."""
    args = parse_args()
    df = generate_synthetic_data(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)

    logger.info("Generated {:,} rows → {}", len(df), out)
    logger.info("Fraud rate: {:.2%}", df["is_fraud"].mean())
    logger.info("Columns: {}", list(df.columns))


if __name__ == "__main__":
    main()
