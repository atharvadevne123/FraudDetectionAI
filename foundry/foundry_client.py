"""
Palantir Foundry integration client for FraudDetectionAI.

Wraps the Foundry REST API with:
  - Dataset upload / download (Parquet, transactional writes)
  - Model registration in Foundry's model registry
  - Pipeline build triggers and status polling
  - Pre-upload data-quality validation
  - Batch multi-dataset uploads
  - Automatic retries with exponential back-off
  - Structured error handling and logging
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 60  # seconds
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


class FoundryError(Exception):
    """Raised when a Foundry API call fails."""


def _build_session() -> requests.Session:
    """Build a requests Session with automatic retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET", "POST", "PUT"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FoundryClient:
    """
    Palantir Foundry REST API client.

    Reads credentials from the environment if not supplied directly:
        FOUNDRY_HOST   — e.g. https://your-instance.palantirfoundry.com
        FOUNDRY_TOKEN  — bearer token (service-account or user token)
    """

    def __init__(self, host: Optional[str] = None, token: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT):
        self.host = (host or os.getenv("FOUNDRY_HOST", "")).rstrip("/")
        self._token = token or os.getenv("FOUNDRY_TOKEN", "")
        self.timeout = timeout
        self._session = _build_session()

        if not self.host:
            logger.warning("FOUNDRY_HOST not set — Foundry calls will fail unless mocked.")
        if not self._token:
            logger.warning("FOUNDRY_TOKEN not set — Foundry calls will fail unless mocked.")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def upload_dataset(
        self,
        df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
        transaction_type: str = "APPEND",
    ) -> dict[str, Any]:
        """
        Upload a pandas DataFrame as a Parquet file to a Foundry dataset.

        Opens a transaction, PUTs the Parquet payload, and commits. The
        transaction is aborted if the upload or commit fails.
        """
        logger.info(
            "Uploading {:,} rows to Foundry dataset {} (branch={}, type={}).",
            len(df), dataset_rid, branch, transaction_type,
        )
        txn_rid = self.create_transaction(dataset_rid, transaction_type, branch=branch)
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            self._put_file(dataset_rid, txn_rid, "data.parquet", buf.read())
            result = self.commit_transaction(dataset_rid, txn_rid)
            logger.success("Uploaded {:,} rows to {} (txn={}).", len(df), dataset_rid, txn_rid)
            return result
        except Exception as exc:
            self.abort_transaction(dataset_rid, txn_rid)
            raise FoundryError(f"Upload failed: {exc}") from exc

    def read_dataset(self, dataset_rid: str, branch: str = "master") -> pd.DataFrame:
        """
        Read a Foundry dataset branch as a pandas DataFrame.

        Fetches all Parquet files in the dataset and concatenates them.
        """
        logger.info("Reading Foundry dataset {} (branch={}).", dataset_rid, branch)
        files = self._list_files(dataset_rid, branch)
        parquet_files = [f for f in files if f.get("logicalPath", "").endswith(".parquet")]
        if not parquet_files:
            logger.warning("No Parquet files found in dataset {}.", dataset_rid)
            return pd.DataFrame()

        frames = []
        for file_info in parquet_files:
            content = self._get_file(dataset_rid, branch, file_info["logicalPath"])
            frames.append(pd.read_parquet(io.BytesIO(content)))
        df = pd.concat(frames, ignore_index=True)
        logger.success("Read {:,} rows from {} ({} file(s)).", len(df), dataset_rid, len(frames))
        return df

    def create_dataset(self, name: str, parent_folder_rid: str) -> dict[str, Any]:
        """Create a new Foundry dataset inside a compass folder."""
        logger.info("Creating Foundry dataset '{}' in {}.", name, parent_folder_rid)
        url = f"{self.host}/foundry-catalog/api/catalog/datasets"
        resp = self._post(url, json={"name": name, "parentFolderRid": parent_folder_rid})
        logger.success("Dataset created: {}", resp.get("rid", resp))
        return resp

    def list_branches(self, dataset_rid: str) -> list[dict[str, Any]]:
        """List all branches of a dataset."""
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/branches"
        return self._get(url).get("data", [])

    def get_schema(self, dataset_rid: str, branch: str = "master") -> dict[str, Any]:
        """Fetch the schema of a dataset branch."""
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/schema?branchId={branch}"
        return self._get(url)

    # ------------------------------------------------------------------
    # Model registry & predictions
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: dict[str, float],
        version: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Register a model version in the Foundry model registry."""
        payload = {
            "name": model_name,
            "version": version,
            "modelPath": model_path,
            "metrics": metrics,
            "description": description,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "service": "fraud-detection-ai",
        }
        logger.info("Registering model '{}' v{} in Foundry registry.", model_name, version)
        url = f"{self.host}/foundry-model-registry/api/models"
        resp = self._post(url, json=payload)
        logger.success("Model registered: {}", resp)
        return resp

    def log_predictions(
        self,
        predictions_df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
    ) -> dict[str, Any]:
        """Append a batch of predictions to a Foundry dataset with a timestamp."""
        predictions_df = predictions_df.copy()
        predictions_df["logged_at"] = datetime.now(timezone.utc).isoformat()
        return self.upload_dataset(
            predictions_df, dataset_rid, branch=branch, transaction_type="APPEND"
        )

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        null_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Validate a DataFrame before uploading to Foundry.

        Checks that all required columns are present and that no column
        exceeds the allowed null ratio. Returns a dict with a ``passed``
        bool and an ``issues`` list describing every violation found.
        """
        issues: list[str] = []

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        if len(df) == 0:
            issues.append("DataFrame is empty.")
        else:
            for col, ratio in df.isna().mean().items():
                if ratio > null_threshold:
                    issues.append(
                        f"Column '{col}' has {ratio:.1%} nulls "
                        f"(threshold {null_threshold:.1%})."
                    )

        result = {"passed": not issues, "issues": issues, "row_count": len(df)}
        if issues:
            logger.warning("Data quality check failed: {}", issues)
        else:
            logger.info("Data quality check passed ({:,} rows).", len(df))
        return result

    # ------------------------------------------------------------------
    # Pipelines / builds
    # ------------------------------------------------------------------

    def publish_to_pipeline(
        self, pipeline_rid: str, trigger_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Trigger a Foundry pipeline build with the given parameters."""
        logger.info("Triggering Foundry pipeline {}.", pipeline_rid)
        url = f"{self.host}/foundry-builds/api/builds"
        resp = self._post(url, json={"pipelineRid": pipeline_rid, "parameters": trigger_params})
        logger.success("Pipeline build triggered: {}", resp.get("buildRid", resp))
        return resp

    def get_build_status(self, build_rid: str) -> dict[str, Any]:
        """Fetch the status of a Foundry build."""
        url = f"{self.host}/foundry-builds/api/builds/{build_rid}"
        return self._get(url)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_upload(
        self,
        dfs_dict: dict[str, pd.DataFrame],
        branch: str = "master",
    ) -> dict[str, Any]:
        """
        Upload multiple DataFrames, keyed by dataset RID.

        Continues on per-dataset failure and reports every outcome, so one
        bad dataset does not abort the whole batch.
        """
        results: dict[str, Any] = {}
        for dataset_rid, df in dfs_dict.items():
            try:
                results[dataset_rid] = {
                    "status": "ok",
                    "result": self.upload_dataset(df, dataset_rid, branch=branch),
                }
            except FoundryError as exc:
                logger.error("Batch upload failed for {}: {}", dataset_rid, exc)
                results[dataset_rid] = {"status": "error", "error": str(exc)}
        failed = sum(1 for r in results.values() if r["status"] == "error")
        logger.info(
            "Batch upload complete: {}/{} datasets succeeded.",
            len(results) - failed, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def create_transaction(
        self,
        dataset_rid: str,
        transaction_type: str = "APPEND",
        branch: str = "master",
    ) -> str:
        """Open a transaction on a dataset and return its RID."""
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/transactions"
        resp = self._post(url, json={"branchId": branch, "transactionType": transaction_type})
        txn_rid = resp["rid"]
        logger.info("Opened {} transaction {} on {}.", transaction_type, txn_rid, dataset_rid)
        return txn_rid

    def commit_transaction(self, dataset_rid: str, transaction_rid: str) -> dict[str, Any]:
        """Commit an open transaction."""
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{transaction_rid}/commit"
        result = self._post(url, json={})
        logger.info("Committed transaction {} on {}.", transaction_rid, dataset_rid)
        return result

    def abort_transaction(self, dataset_rid: str, transaction_rid: str) -> None:
        """Abort an open transaction. Logs but does not raise on failure."""
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{transaction_rid}/abort"
        try:
            self._post(url, json={})
            logger.info("Aborted transaction {} on {}.", transaction_rid, dataset_rid)
        except FoundryError as exc:
            logger.error("Failed to abort transaction {}: {}", transaction_rid, exc)

    # ------------------------------------------------------------------
    # Low-level REST helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _put_file(self, dataset_rid: str, txn_rid: str, logical_path: str, data: bytes) -> None:
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{txn_rid}/files:upload"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/octet-stream",
            "X-Foundry-Logical-Path": logical_path,
        }
        resp = self._session.put(url, data=data, headers=headers, timeout=self.timeout)
        self._raise_for_status(resp, "PUT file")

    def _list_files(self, dataset_rid: str, branch: str) -> list[dict]:
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/files?branchId={branch}"
        return self._get(url).get("data", [])

    def _get_file(self, dataset_rid: str, branch: str, logical_path: str) -> bytes:
        url = (
            f"{self.host}/api/v1/datasets/{dataset_rid}/files/"
            f"{logical_path}/content?branchId={branch}"
        )
        resp = self._session.get(
            url, headers={"Authorization": f"Bearer {self._token}"}, timeout=self.timeout
        )
        self._raise_for_status(resp, f"GET file {logical_path}")
        return resp.content

    def _get(self, url: str) -> dict:
        resp = self._session.get(url, headers=self._headers(), timeout=self.timeout)
        self._raise_for_status(resp, f"GET {url}")
        return resp.json()

    def _post(self, url: str, **kwargs) -> dict:
        resp = self._session.post(url, headers=self._headers(), timeout=self.timeout, **kwargs)
        self._raise_for_status(resp, f"POST {url}")
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status_code}

    @staticmethod
    def _raise_for_status(resp: requests.Response, context: str = "") -> None:
        if not resp.ok:
            raise FoundryError(
                f"Foundry API error [{context}]: "
                f"HTTP {resp.status_code} — {resp.text[:500]}"
            )
