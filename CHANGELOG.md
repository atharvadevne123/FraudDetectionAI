# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `/version` endpoint returning API version and Python runtime metadata
- `/readiness` endpoint for Kubernetes-style readiness probes
- `_determine_risk_tier()` helper extracted from `_score_transaction()`
- Full type annotations across all modules
- `Makefile` with `install`, `test`, `lint`, `run`, `docker-build` targets
- `.pre-commit-config.yaml` with ruff and trailing-whitespace hooks
- `CONTRIBUTING.md` with developer setup guide
- `tests/test_drift_monitor.py` — unit tests for DriftMonitor
- `tests/test_fraud_classifier.py` — unit tests for FraudEnsemble
- `tests/test_rag_explainer.py` — unit tests for RAGExplainer with mocked deps
- Parametrized edge-case tests in `tests/test_api.py`

### Changed
- CI workflow: fixed action versions (v6 → v4/v5), added coverage artifact upload
- Improved SMOTE handling with correct minority-class ratio in tests

### Fixed
- CI action version pinning for `actions/checkout`, `actions/setup-python`, `actions/upload-artifact`

## [1.0.0] — Initial Release

### Added
- Real-time fraud scoring API with Flask + flask-restx
- XGBoost + LightGBM + Random Forest ensemble with SMOTE balancing
- Isolation Forest + LOF + One-Class SVM anomaly detection
- RAG-powered explanations via FAISS + Sentence Transformers + Claude
- Apache Airflow DAG for automated 15-minute retraining pipeline
- KS-test and Evidently drift monitoring with Power BI export
- MLflow experiment tracking with AUC-ROC and average precision metrics
- Docker multi-service deployment with PostgreSQL and Redis
- Prometheus metrics endpoint
- API key authentication and rate limiting
