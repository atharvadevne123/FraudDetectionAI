"""
Data quality check script for transaction datasets.
Checks for missing values, out-of-range values, and schema compliance.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_COLS = ["user_id", "amount", "merchant_category", "payment_method"]
NUMERIC_BOUNDS = {
    "amount": (0.0, 1_000_000.0),
    "credit_utilization": (0.0, 1.0),
    "account_age_days": (0, 36_500),
    "prior_fraud_count": (0, 1_000),
}


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Run data quality checks on a transaction dataset")
    p.add_argument("--input", required=True, help="Input CSV or Parquet file")
    p.add_argument(
        "--missing-threshold",
        type=float,
        default=0.05,
        help="Max allowed missing ratio per column (default 5%%)",
    )
    p.add_argument("--fail-fast", action="store_true", help="Exit non-zero on first quality issue")
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame based on file extension."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return error strings for any required columns absent from the DataFrame."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]
    return []


def check_missing_values(df: pd.DataFrame, threshold: float) -> list[str]:
    """Return error strings for columns whose missing-value ratio exceeds threshold."""
    issues = []
    for col in df.columns:
        ratio = df[col].isnull().mean()
        if ratio > threshold:
            issues.append(
                f"Column '{col}' has {ratio:.1%} missing values (threshold: {threshold:.0%})"
            )
    return issues


def check_numeric_bounds(df: pd.DataFrame) -> list[str]:
    """Return error strings for numeric columns with values outside their allowed range."""
    issues = []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            issues.append(f"Column '{col}' has {out_of_range} values outside [{lo}, {hi}]")
    return issues


def check_duplicate_rows(df: pd.DataFrame) -> list[str]:
    """Return an error string if the DataFrame contains any fully-duplicate rows."""
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        return [f"Dataset contains {n_dup} duplicate rows"]
    return []


def main() -> None:
    args = parse_args()
    path = Path(args.input)

    if not path.exists():
        logger.error("Input file not found: {}", args.input)
        sys.exit(1)

    logger.info("Loading data: {}", args.input)
    df = load_data(args.input)
    logger.info("Loaded {:,} rows × {} columns", len(df), df.shape[1])

    all_issues: list[str] = []

    checks = [
        ("Required columns", check_required_columns(df)),
        ("Missing values", check_missing_values(df, args.missing_threshold)),
        ("Numeric bounds", check_numeric_bounds(df)),
        ("Duplicate rows", check_duplicate_rows(df)),
    ]

    for check_name, issues in checks:
        if issues:
            for issue in issues:
                logger.warning("[{}] {}", check_name, issue)
            all_issues.extend(issues)
        else:
            logger.info("[{}] OK", check_name)

        if args.fail_fast and issues:
            sys.exit(1)

    if all_issues:
        logger.error("Data quality check FAILED with {} issues.", len(all_issues))
        sys.exit(1)
    else:
        logger.success("All data quality checks PASSED for {:,} rows.", len(df))


if __name__ == "__main__":
    main()
