"""Configuration constants for the fraud detection system."""

from config.constants import (
    API_VERSION,
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    ENSEMBLE_WEIGHTS,
    FEATURE_COUNT,
    RISK_TIERS,
    SMOTE_TARGET_RATIO,
)

__all__ = [
    "RISK_TIERS",
    "DEFAULT_CONTAMINATION",
    "DEFAULT_N_ESTIMATORS",
    "SMOTE_TARGET_RATIO",
    "FEATURE_COUNT",
    "ENSEMBLE_WEIGHTS",
    "API_VERSION",
]
