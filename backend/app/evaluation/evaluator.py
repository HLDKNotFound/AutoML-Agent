"""
Model Evaluation Service.
Calculates standardized metrics, timing benchmarks, model memory footprints,
and error matrices for both classification and regression tasks.
"""

import time
import sys
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, roc_curve, precision_recall_curve
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes all classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
    }

    # ROC-AUC and Log Loss if probabilities are available
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary ROC-AUC
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob
                auc = float(roc_auc_score(y_true, prob_pos))
                metrics["roc_auc"] = round(auc, 4)
                ll = float(log_loss(y_true, y_prob))
                metrics["log_loss"] = round(ll, 4)
            elif n_classes > 2:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                metrics["roc_auc"] = round(auc, 4)
                ll = float(log_loss(y_true, y_prob))
                metrics["log_loss"] = round(ll, 4)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes all regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = 0.0

    return {
        "rmse": round(rmse, 4),
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4)
    }


def evaluate_pipeline(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str,
    train_time_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates a fitted pipeline on validation or test data.
    Measures inference latency and pickled model footprint.
    """
    # Measure inference latency
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    t_infer = (time.perf_counter() - t_start) * 1000.0  # ms
    
    # Model size in KB
    try:
        model_size_kb = len(pickle.dumps(pipeline)) / 1024.0
    except Exception:
        model_size_kb = 0.0

    # Probabilities if classification
    y_prob = None
    is_classification = problem_type in ("binary_classification", "multiclass_classification")
    if is_classification and hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_val)
        except Exception:
            pass

    if is_classification:
        metrics = compute_classification_metrics(y_val.to_numpy(), y_pred, y_prob)
    else:
        metrics = compute_regression_metrics(y_val.to_numpy(), y_pred)

    return {
        "metrics": metrics,
        "train_time_sec": round(train_time_sec, 3),
        "inference_time_ms": round(t_infer, 2),
        "model_size_kb": round(model_size_kb, 2),
        "y_pred": y_pred,
        "y_prob": y_prob
    }
