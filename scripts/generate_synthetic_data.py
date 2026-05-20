"""
Quickly generate a synthetic labelled transaction dataset for development/testing.
Usage: python scripts/generate_synthetic_data.py --rows 100000 --out data/raw/transactions.parquet
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import generate_synthetic_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--out", type=str, default="data/raw/transactions.parquet")
    args = p.parse_args()

    df = generate_synthetic_data(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Generated {:,} rows → {}", len(df), out)
    logger.info("Fraud rate: {:.2%}", df["is_fraud"].mean())


if __name__ == "__main__":
    main()
