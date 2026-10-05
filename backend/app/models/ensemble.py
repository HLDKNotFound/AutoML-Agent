"""
Ensemble Blending module.
Combines multiple diverse pipelines (e.g. CatBoost, LightGBM, RandomForest, ExtraTrees, Ridge)
via soft probability voting (classification) or weighted prediction averaging (regression).
"""

from typing import Any, List, Tuple
import numpy as np
import pandas as pd


class EnsembleBlendPipeline:
    """
    Serializable ensemble estimator combining pre-fitted scikit-learn pipelines.
    Supports atomic predict and predict_proba with zero refitting.
    """
    def __init__(self, estimators: List[Tuple[str, Any]], weights: List[float], problem_type: str):
        self.estimators = estimators  # [(model_name, fitted_pipeline), ...]
        weights_arr = np.array(weights, dtype=float)
        self.weights = weights_arr / np.sum(weights_arr)
        self.problem_type = problem_type
        
        # Discover classes_ from first valid classification estimator
        self.classes_ = None
        for _, est in estimators:
            if hasattr(est, "classes_"):
                self.classes_ = est.classes_
                break
        if self.classes_ is None and problem_type in ("binary_classification", "multiclass_classification"):
            self.classes_ = np.array([False, True])

    def predict(self, X: pd.DataFrame):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            proba = self.predict_proba(X)
            if self.problem_type == "binary_classification":
                pos_idx = 1 if proba.shape[1] > 1 else 0
                is_bool = isinstance(self.classes_[0], (bool, np.bool_))
                if is_bool:
                    return proba[:, pos_idx] >= 0.5
                return (proba[:, pos_idx] >= 0.5).astype(type(self.classes_[0]))
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Weighted average of regression predictions
            preds = np.zeros(len(X), dtype=float)
            for (_, est), w in zip(self.estimators, self.weights):
                preds += w * np.array(est.predict(X), dtype=float)
            return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        total_prob = None
        for (_, est), w in zip(self.estimators, self.weights):
            if hasattr(est, "predict_proba"):
                try:
                    p = est.predict_proba(X)
                except Exception:
                    raw = np.array(est.predict(X), dtype=float)
                    p = np.column_stack([1.0 - raw, raw])
            else:
                raw = np.array(est.predict(X), dtype=float)
                p = np.column_stack([1.0 - raw, raw])

            if total_prob is None:
                total_prob = w * p
            else:
                total_prob += w * p

        return total_prob
