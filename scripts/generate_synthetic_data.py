"""
Quickly generate a synthetic labelled transaction dataset for development/testing.
Usage: python scripts/generate_synthetic_data.py --rows 100000 --out data/raw/transactions.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import generate_synthetic_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--out", type=str, default="data/raw/transactions.parquet")
    args = p.parse_args()

    df = generate_synthetic_data(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Generated {len(df):,} rows → {out}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
