# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.1.0] — 2026-06-03

### Added
- 60-commit quality improvement pass: ruff formatting, type annotations, docstrings, test expansion
- Google-style docstrings on `FraudEnsemble.fit()`, `RAGExplainer.save_index/load_index`,
  `AnomalyDetector.save/load`, `DriftMonitor._fallback_drift_check`,
  `TransactionFeatureEngineer._behavioral_features`, all `utils/metrics.py` and `utils/validation.py` functions,
  and all script `main()`/`load_data()`/`score_chunk()` functions
- Return type annotations on all Flask-RestX route handlers and `get_logger()`
- Parametrized test classes across 14 test files (150+ new assertions)
- Coverage configuration in `pyproject.toml` (`fail_under = 70`)
- Makefile targets: `format`, `format-check`, `security`, `generate-data`
- `.env.example` expanded with all required variables, threshold overrides, and inline comments
- CONTRIBUTING.md expanded with testing guide, code style section, and commit examples

### Changed
- Ruff format applied to 10 source files (api/app.py, config/constants.py, models/*, monitoring/*, scripts/*)
- E402 lint suppressions added to `scripts/batch_predict.py` and `scripts/evaluate.py` for intentional `sys.path` imports

### Added
- `/version` endpoint returning API version and Python runtime metadata
- `/readiness` endpoint for Kubernetes-style readiness probes
- `_determine_risk_tier()` helper extracted from `_score_transaction()`
- Full type annotations across all modules (Union, Optional, Any, dict[str, float])
- `Makefile` with `install`, `test`, `lint`, `run`, `docker-build`, `evaluate`, `check-data` targets
- `.pre-commit-config.yaml` with ruff and trailing-whitespace hooks
- `CONTRIBUTING.md` with developer setup guide
- `.editorconfig` for consistent cross-editor formatting
- `SECURITY.md` expanded with API auth, rate limiting, and secrets guidance
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `utils/` package: `validation.py`, `metrics.py`, `logging.py`
- `config/` package: `constants.py` with risk tiers, defaults, and ensemble weights
- `scripts/evaluate.py` — offline model evaluation with precision@k, AP, DR, FPR
- `scripts/batch_predict.py` — chunked offline scoring with risk tier assignment
- `scripts/check_data_quality.py` — schema, bounds, missing-value, and duplicate checks
- `AnomalyDetector.predict_proba()` returning 2-column probability matrix
- `models/__init__.py`, `models/anomaly/__init__.py`, `models/ensemble/__init__.py`, `models/rag/__init__.py` with public exports
- `api/wsgi.py` improved with `LOG_LEVEL`, `LOG_FILE`, and `PORT` environment variable support
- Tests: `test_anomaly_detector_extended`, `test_anomaly_predict_proba`, `test_validation`,
  `test_metrics`, `test_config`, `test_batch_predict`, `test_data_quality`,
  `test_logging_utils`, `test_api_health`, `test_api_endpoints`, `test_feature_engineering_extended`
- CI badge added to README; Testing and Contributing sections added
- docstrings added to all private methods across models, monitoring, and pipeline modules

### Changed
- CI workflow: fixed action versions (v6 → v4/v5), added coverage artifact upload
- `pyproject.toml` expanded with PyPI classifiers, ruff config, mypy config, pytest paths
- `_velocity_features` refactored to preserve original DataFrame index (alignment-safe)
- `generate_synthetic_data.py` improved with `parse_args()`, CSV output, `--seed` flag
- Airflow DAG `drift_check` task wrapped in try/except to prevent pipeline failure on monitoring errors

### Fixed
- CI action version pinning for `actions/checkout`, `actions/setup-python`, `actions/upload-artifact`
- Unused variable `req_json`/`res_json` in `scripts/take_screenshots.py`
- Import sort order across 14 modules (ruff I001)

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
