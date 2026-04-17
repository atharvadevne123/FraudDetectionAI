"""
Supervised fraud classification ensemble.
XGBoost + LightGBM + Random Forest with soft-voting and calibrated probabilities.
SHAP values computed per-prediction for local explainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from loguru import logger
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class FraudEnsemble:
    """
    Three-model soft-voting ensemble with SMOTE oversampling,
    probability calibration, and SHAP-based explanations.
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        rf_params: Optional[dict] = None,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.feature_names = []

        xgb_defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=50,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        lgb_defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        rf_defaults = dict(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.xgb_model = xgb.XGBClassifier(**(xgb_params or xgb_defaults))
        self.lgb_model = lgb.LGBMClassifier(**(lgb_params or lgb_defaults))
        self.rf_model = RandomForestClassifier(**(rf_params or rf_defaults))

        self.ensemble: Optional[CalibratedClassifierCV] = None
        self.scaler = StandardScaler()
        self._shap_explainer: Optional[shap.TreeExplainer] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_X: Optional[pd.DataFrame] = None,
        eval_y: Optional[pd.Series] = None,
        mlflow_run: bool = True,
    ) -> "FraudEnsemble":
        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.values)
        y_arr = y.values

        logger.info("Applying SMOTE to balance {:,} samples (fraud rate: {:.2%}).",
                    len(y_arr), y_arr.mean())
        smote = SMOTE(sampling_strategy=0.1, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_arr, y_arr)
        logger.info("Post-SMOTE: {:,} samples.", len(y_res))

        voter = VotingClassifier(
            estimators=[
                ("xgb", self.xgb_model),
                ("lgb", self.lgb_model),
                ("rf", self.rf_model),
            ],
            voting="soft",
            weights=[0.4, 0.4, 0.2],
        )
        self.ensemble = CalibratedClassifierCV(
            voter,
            method=self.calibration_method,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
        )

        if mlflow_run:
            with mlflow.start_run(run_name="fraud_ensemble"):
                self.ensemble.fit(X_res, y_res)
                self._log_metrics(X_res, y_res, eval_X, eval_y)
        else:
            self.ensemble.fit(X_res, y_res)

        # Build SHAP explainer on the first XGBoost base estimator
        try:
            base_xgb = self.ensemble.calibrated_classifiers_[0].estimator.named_estimators_["xgb"]
            self._shap_explainer = shap.TreeExplainer(base_xgb)
            logger.success("SHAP TreeExplainer initialised.")
        except Exception as e:
            logger.warning("Could not build SHAP explainer: {}", e)

        self._fitted = True
        logger.success("FraudEnsemble fitted.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.ensemble.predict_proba(self.scaler.transform(X.values))

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def explain(self, X: pd.DataFrame, max_display: int = 10) -> dict:
        """Return SHAP values and top-k feature contributions for each row."""
        if self._shap_explainer is None:
            return {"error": "SHAP explainer not available."}
        X_scaled = self.scaler.transform(X.values)
        shap_values = self._shap_explainer.shap_values(X_scaled)
        # shap_values shape: (n_samples, n_features) for binary
        sv = shap_values if not isinstance(shap_values, list) else shap_values[1]
        results = []
        for i in range(len(X)):
            top_idx = np.argsort(np.abs(sv[i]))[::-1][:max_display]
            results.append({
                "shap_values": {self.feature_names[j]: float(sv[i][j]) for j in top_idx},
                "base_value": float(self._shap_explainer.expected_value
                                    if not isinstance(self._shap_explainer.expected_value, list)
                                    else self._shap_explainer.expected_value[1]),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "fraud_ensemble.joblib"
        joblib.dump(self, path)
        logger.info("FraudEnsemble saved → {}", path)
        return path

    @classmethod
    def load(cls, path=None) -> "FraudEnsemble":
        path = Path(path) if path else ARTIFACT_DIR / "fraud_ensemble.joblib"
        obj = joblib.load(path)
        logger.info("FraudEnsemble loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, X_train, y_train, eval_X, eval_y):
        train_proba = self.ensemble.predict_proba(X_train)[:, 1]
        mlflow.log_metric("train_auc_roc", roc_auc_score(y_train, train_proba))
        mlflow.log_metric("train_avg_precision", average_precision_score(y_train, train_proba))
        if eval_X is not None and eval_y is not None:
            eval_X_scaled = self.scaler.transform(eval_X.values)
            eval_proba = self.ensemble.predict_proba(eval_X_scaled)[:, 1]
            mlflow.log_metric("val_auc_roc", roc_auc_score(eval_y, eval_proba))
            mlflow.log_metric("val_avg_precision", average_precision_score(eval_y, eval_proba))
            logger.info("Val AUC-ROC: {:.4f}", roc_auc_score(eval_y, eval_proba))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Fit FraudEnsemble before calling predict.")
