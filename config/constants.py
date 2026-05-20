"""System-wide constants for the fraud detection pipeline."""

from __future__ import annotations

API_VERSION = "1.0.0"

# Risk scoring thresholds
RISK_TIERS: dict[str, tuple[float, float]] = {
    "CRITICAL": (0.90, 1.01),
    "HIGH":     (0.70, 0.90),
    "MEDIUM":   (0.50, 0.70),
    "LOW":      (0.30, 0.50),
    "CLEAN":    (0.00, 0.30),
}

# Anomaly detector defaults
DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_ESTIMATORS = 100

# SMOTE oversampling target minority/majority ratio
SMOTE_TARGET_RATIO = 0.10

# Number of engineered features (32 + 1 anomaly score)
FEATURE_COUNT = 33

# Ensemble soft-vote weights: [XGBoost, LightGBM, RandomForest]
ENSEMBLE_WEIGHTS: list[float] = [0.4, 0.4, 0.2]

# Velocity rolling-window sizes in days
VELOCITY_WINDOWS: list[int] = [1, 7, 30]

# Rate limits
RATE_LIMIT_DAY = "200 per day"
RATE_LIMIT_HOUR = "50 per hour"

# Batch prediction ceiling
MAX_BATCH_SIZE = 500
