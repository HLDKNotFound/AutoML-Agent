"""
Canonical Metric Registry for AutoML-Agent.
Centralizes metric definitions, optimization directions, supported problem types,
score transformations, and fair comparison logic across evaluation, tuning, and reporting.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
import numpy as np


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_name: str
    direction: str  # "maximize" or "minimize"
    problem_types: List[str]
    description: str
    higher_is_better: bool

    def score(self, raw_value: float) -> float:
        """
        Converts raw metric value into a canonical score where higher is ALWAYS better.
        For maximization metrics, returns raw_value.
        For minimization metrics (RMSE, MAE, etc.), returns -raw_value.
        """
        val = float(raw_value)
        return val if self.higher_is_better else -val

    def is_better(self, val_a: float, val_b: float) -> bool:
        """Returns True if val_a is strictly better than val_b according to metric direction."""
        if self.higher_is_better:
            return float(val_a) > float(val_b)
        return float(val_a) < float(val_b)

    def format_value(self, val: float) -> str:
        """Formats the metric value cleanly for reports."""
        return f"{float(val):.4f}"


METRIC_REGISTRY: Dict[str, MetricDefinition] = {
    # Classification Metrics
    "f1_macro": MetricDefinition(
        name="f1_macro",
        display_name="Macro F1-Score",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Harmonic mean of precision and recall unweighted across all classes.",
        higher_is_better=True,
    ),
    "f1_weighted": MetricDefinition(
        name="f1_weighted",
        display_name="Weighted F1-Score",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="F1-score weighted by support for each class.",
        higher_is_better=True,
    ),
    "accuracy": MetricDefinition(
        name="accuracy",
        display_name="Accuracy",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Ratio of correct predictions over total predictions.",
        higher_is_better=True,
    ),
    "precision_macro": MetricDefinition(
        name="precision_macro",
        display_name="Macro Precision",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Unweighted average precision across classes.",
        higher_is_better=True,
    ),
    "recall_macro": MetricDefinition(
        name="recall_macro",
        display_name="Macro Recall",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Unweighted average recall across classes.",
        higher_is_better=True,
    ),
    "precision_weighted": MetricDefinition(
        name="precision_weighted",
        display_name="Weighted Precision",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Support-weighted average precision.",
        higher_is_better=True,
    ),
    "recall_weighted": MetricDefinition(
        name="recall_weighted",
        display_name="Weighted Recall",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Support-weighted average recall.",
        higher_is_better=True,
    ),
    "roc_auc": MetricDefinition(
        name="roc_auc",
        display_name="ROC-AUC",
        direction="maximize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Area under receiver operating characteristic curve.",
        higher_is_better=True,
    ),
    "log_loss": MetricDefinition(
        name="log_loss",
        display_name="Log Loss (Cross-Entropy)",
        direction="minimize",
        problem_types=["binary_classification", "multiclass_classification"],
        description="Logarithmic loss of probabilistic classification.",
        higher_is_better=False,
    ),

    # Regression Metrics
    "rmse": MetricDefinition(
        name="rmse",
        display_name="Root Mean Squared Error (RMSE)",
        direction="minimize",
        problem_types=["regression"],
        description="Square root of average squared prediction errors.",
        higher_is_better=False,
    ),
    "mse": MetricDefinition(
        name="mse",
        display_name="Mean Squared Error (MSE)",
        direction="minimize",
        problem_types=["regression"],
        description="Average squared difference between predicted and actual values.",
        higher_is_better=False,
    ),
    "mae": MetricDefinition(
        name="mae",
        display_name="Mean Absolute Error (MAE)",
        direction="minimize",
        problem_types=["regression"],
        description="Average absolute difference between predicted and actual values.",
        higher_is_better=False,
    ),
    "r2": MetricDefinition(
        name="r2",
        display_name="R² Score (Coefficient of Determination)",
        direction="maximize",
        problem_types=["regression"],
        description="Proportion of variance explained by the model.",
        higher_is_better=True,
    ),
    "mape": MetricDefinition(
        name="mape",
        display_name="Mean Absolute Percentage Error (MAPE)",
        direction="minimize",
        problem_types=["regression"],
        description="Average percentage difference between predicted and actual values.",
        higher_is_better=False,
    ),
}


def get_metric_definition(metric_name: str) -> MetricDefinition:
    """Retrieves metric definition from registry with case-insensitive fallback."""
    key = (metric_name or "").strip().lower()
    if key in METRIC_REGISTRY:
        return METRIC_REGISTRY[key]
    
    # Sensible fallback heuristics for unknown metric names
    higher = not any(err_kw in key for err_kw in ("loss", "error", "mse", "mae", "rmse", "mape", "deviance"))
    direction = "maximize" if higher else "minimize"
    return MetricDefinition(
        name=key,
        display_name=metric_name,
        direction=direction,
        problem_types=["binary_classification", "multiclass_classification", "regression"],
        description=f"User-specified metric '{metric_name}'",
        higher_is_better=higher,
    )


def is_metric_better(metric_name: str, candidate_val: float, baseline_val: float) -> bool:
    """Returns True if candidate_val is strictly superior to baseline_val for metric_name."""
    definition = get_metric_definition(metric_name)
    return definition.is_better(candidate_val, baseline_val)


def get_optimization_score(metric_name: str, raw_metric_value: float) -> float:
    """Converts raw metric to a maximization score for optimizers like Optuna."""
    definition = get_metric_definition(metric_name)
    return definition.score(raw_metric_value)


def validate_metric_compatibility(metric_name: str, problem_type: str) -> bool:
    """Validates that the chosen metric is compatible with the given problem type."""
    definition = get_metric_definition(metric_name)
    return problem_type in definition.problem_types
