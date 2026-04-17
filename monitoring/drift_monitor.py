"""
Data drift and model performance monitoring.
Uses Evidently for statistical drift detection.
Exports Power BI-compatible JSON/CSV for dashboard consumption.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ClassificationQualityMetric,
)
from evidently.report import Report
from loguru import logger


MONITOR_DIR = Path(__file__).parent / "reports"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = MONITOR_DIR / "reference_stats.parquet"


class DriftMonitor:
    """
    Compares current batch statistics against a reference distribution.
    Detects feature drift, missing values, and score distribution shifts.
    Exports Power BI-compatible flat files.
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        score_shift_threshold: float = 0.05,
    ):
        self.drift_threshold = drift_threshold
        self.score_shift_threshold = score_shift_threshold
        self._reference: Optional[pd.DataFrame] = None
        self._load_reference()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store a clean baseline for future drift comparisons."""
        self._reference = df.copy()
        df.to_parquet(REFERENCE_PATH, index=False)
        logger.info("Reference distribution saved ({:,} samples).", len(df))

    def run(self, current: pd.DataFrame) -> dict:
        """
        Run full drift analysis on current batch vs reference.
        Returns dict with drift_detected flag and per-feature stats.
        """
        if self._reference is None:
            logger.warning("No reference distribution set. Storing current batch as reference.")
            self.set_reference(current)
            return {"drift_detected": False, "reason": "Reference just set."}

        numeric_cols = [c for c in current.select_dtypes(include="number").columns
                        if c in self._reference.columns]

        report_data = self._run_evidently(current, numeric_cols)
        score_shift = self._check_score_shift(current)
        missing_ratio = self._check_missing(current)

        drift_detected = (
            report_data.get("dataset_drift", False)
            or abs(score_shift) > self.score_shift_threshold
        )

        summary = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_size": len(current),
            "drift_detected": drift_detected,
            "dataset_drift": report_data.get("dataset_drift", False),
            "drifted_features": report_data.get("drifted_features", []),
            "score_mean_current": score_shift,
            "missing_ratio": missing_ratio,
            "evidently_metrics": report_data,
        }

        self._export_powerbi(current, summary)
        self._save_report(summary)

        if drift_detected:
            logger.warning("DRIFT DETECTED — {} features drifted.", len(summary["drifted_features"]))
        else:
            logger.info("No significant drift. Batch stats healthy.")

        return summary

    # ------------------------------------------------------------------
    # Evidently integration
    # ------------------------------------------------------------------

    def _run_evidently(self, current: pd.DataFrame, numeric_cols: list[str]) -> dict:
        ref = self._reference[numeric_cols].copy()
        cur = current[numeric_cols].copy()

        col_mapping = ColumnMapping()
        if "is_fraud" in self._reference.columns and "is_fraud" in current.columns:
            col_mapping.target = "is_fraud"
        if "fraud_score" in current.columns:
            col_mapping.prediction = "fraud_score"

        report = Report(metrics=[
            DatasetDriftMetric(drift_share_threshold=self.drift_threshold),
            DataDriftTable(num_stattest="ks", cat_stattest="chi2"),
            DatasetMissingValuesMetric(),
        ])

        try:
            report.run(reference_data=ref, current_data=cur, column_mapping=col_mapping)
            result = report.as_dict()
            metrics = result.get("metrics", [])

            dataset_metric = next(
                (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
            )
            drift_table = next(
                (m for m in metrics if m.get("metric") == "DataDriftTable"), {}
            )

            drifted = dataset_metric.get("result", {}).get("dataset_drift", False)
            drifted_cols = [
                col for col, stats in drift_table.get("result", {}).get("drift_by_columns", {}).items()
                if stats.get("drift_detected")
            ]
            return {"dataset_drift": drifted, "drifted_features": drifted_cols}
        except Exception as e:
            logger.warning("Evidently report failed: {}", e)
            return self._fallback_drift_check(current, numeric_cols)

    def _fallback_drift_check(self, current: pd.DataFrame, cols: list[str]) -> dict:
        """KS-based drift check when Evidently fails."""
        from scipy import stats
        drifted = []
        for col in cols:
            if col not in self._reference.columns:
                continue
            ref_vals = self._reference[col].dropna()
            cur_vals = current[col].dropna()
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue
            _, p_val = stats.ks_2samp(ref_vals, cur_vals)
            if p_val < 0.05:
                drifted.append(col)
        return {"dataset_drift": len(drifted) > 0, "drifted_features": drifted}

    # ------------------------------------------------------------------
    # Score shift monitoring
    # ------------------------------------------------------------------

    def _check_score_shift(self, current: pd.DataFrame) -> float:
        if "fraud_score" not in current.columns or "fraud_score" not in self._reference.columns:
            return 0.0
        ref_mean = self._reference["fraud_score"].mean()
        cur_mean = current["fraud_score"].mean()
        shift = cur_mean - ref_mean
        logger.debug("Score shift: {:.4f} (ref mean: {:.4f}, cur mean: {:.4f}).",
                     shift, ref_mean, cur_mean)
        return float(shift)

    def _check_missing(self, current: pd.DataFrame) -> float:
        total = current.size
        missing = current.isnull().sum().sum()
        return round(missing / total, 4) if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Power BI export
    # ------------------------------------------------------------------

    def _export_powerbi(self, current: pd.DataFrame, summary: dict) -> None:
        """
        Exports two flat files for Power BI consumption:
          1. powerbi_transactions.csv   — scored transaction-level data
          2. powerbi_drift_metrics.csv  — per-run drift KPIs
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Transaction-level export
        txn_cols = ["transaction_id", "user_id", "amount", "merchant_category",
                    "timestamp", "fraud_score", "anomaly_score", "fraud_label",
                    "is_anomaly", "hour_of_day", "is_weekend", "is_night"]
        available = [c for c in txn_cols if c in current.columns]
        txn_path = MONITOR_DIR / f"powerbi_transactions_{ts}.csv"
        current[available].to_csv(txn_path, index=False)

        # KPI-level export
        kpi_row = {
            "run_timestamp": summary["run_timestamp"],
            "batch_size": summary["batch_size"],
            "drift_detected": int(summary["drift_detected"]),
            "n_drifted_features": len(summary.get("drifted_features", [])),
            "fraud_rate": current.get("fraud_label", pd.Series(dtype=float)).mean()
                         if "fraud_label" in current.columns else None,
            "avg_fraud_score": current["fraud_score"].mean() if "fraud_score" in current.columns else None,
            "avg_anomaly_score": current["anomaly_score"].mean() if "anomaly_score" in current.columns else None,
            "missing_ratio": summary["missing_ratio"],
        }
        kpi_path = MONITOR_DIR / "powerbi_drift_kpis.csv"
        kpi_df = pd.DataFrame([kpi_row])
        if kpi_path.exists():
            existing = pd.read_csv(kpi_path)
            kpi_df = pd.concat([existing, kpi_df], ignore_index=True)
        kpi_df.to_csv(kpi_path, index=False)

        logger.info("Power BI exports written → {} and {}", txn_path.name, kpi_path.name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, summary: dict) -> None:
        ts = summary["run_timestamp"].replace(":", "-")[:19]
        path = MONITOR_DIR / f"drift_report_{ts}.json"
        path.write_text(json.dumps(summary, indent=2, default=str))

    def _load_reference(self) -> None:
        if REFERENCE_PATH.exists():
            self._reference = pd.read_parquet(REFERENCE_PATH)
            logger.info("Reference distribution loaded ({:,} rows).", len(self._reference))
