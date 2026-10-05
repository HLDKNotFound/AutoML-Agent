"""
Pydantic Schemas for ML Automation Agent.
Strict structured interfaces for LLM Agent responses and API communication.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataProblem(BaseModel):
    severity: str = Field(description="'critical', 'warning', or 'info'")
    category: str = Field(description="Category such as 'leakage', 'missing_values', 'outliers', 'class_imbalance', 'high_cardinality', 'collinearity', 'constant_feature'")
    feature: Optional[str] = Field(default=None, description="Feature column name associated with problem, or None if dataset-wide")
    description: str = Field(description="Detailed explanation of the detected problem")
    suggestion: str = Field(description="Recommended action to mitigate or resolve the problem")


class DataQualityReport(BaseModel):
    valid: bool = Field(description="Whether the dataset is valid for machine learning training")
    problems: List[DataProblem] = Field(default_factory=list, description="List of detected data quality problems")
    warnings: List[str] = Field(default_factory=list, description="General cautionary warnings")
    recommended_features: List[str] = Field(default_factory=list, description="Features recommended to keep for training")
    removed_features: List[str] = Field(default_factory=list, description="Features recommended to drop (due to leakage, ID nature, high missingness, etc.)")
    reasoning: str = Field(description="Comprehensive rationale from the LLM Data Scientist")
    iteration: int = Field(default=1, description="Validation loop iteration index (1 to 3)")


class FeatureSelectionResult(BaseModel):
    selected_features: List[str] = Field(description="Final subset of selected input features")
    dropped_features: List[str] = Field(default_factory=list, description="Features excluded from modeling")
    reasons: Dict[str, str] = Field(default_factory=dict, description="Explanation for dropping each dropped feature")
    importance_scores: Dict[str, float] = Field(default_factory=dict, description="Heuristic or mutual info importance scores if computed")


class PreprocessingStrategy(BaseModel):
    numeric_imputer: str = Field(default="median", description="'median', 'mean', 'most_frequent', or 'knn'")
    numeric_scaler: Optional[str] = Field(default="robust", description="'robust', 'standard', 'minmax', 'quantile', or null")
    categorical_imputer: str = Field(default="most_frequent", description="'most_frequent' or 'constant'")
    categorical_encoder: str = Field(default="one_hot", description="'one_hot', 'ordinal', or 'target'")
    handle_unknown: str = Field(default="ignore", description="Strategy for unknown categories at inference")
    feature_engineering: Optional[List[str]] = Field(default_factory=list, description="Optional transformations like polynomial or log1p")


class ModelCandidate(BaseModel):
    model_name: str = Field(description="Identifier e.g. 'RandomForest', 'LogisticRegression', 'LightGBM', 'SVC', 'GradientBoosting', 'MLP'")
    model_family: str = Field(description="'linear', 'tree_ensemble', 'gradient_boosting', 'support_vector', 'neighbors', or 'neural_network'")
    reason: str = Field(description="Detailed reason for selecting this candidate based on dataset characteristics")
    preprocessing_strategy: PreprocessingStrategy = Field(description="Tailored preprocessing configuration specific to this model")


class MLStrategy(BaseModel):
    problem_type: str = Field(description="'binary_classification', 'multiclass_classification', or 'regression'")
    target_column: str = Field(description="Target column name")
    primary_metric: str = Field(description="'f1_macro', 'roc_auc', 'accuracy', 'rmse', 'r2', 'mae'")
    candidate_models: List[ModelCandidate] = Field(description="Exactly 5 candidate models from at least 3 distinct model families")
    reasoning: str = Field(description="Strategic architectural justification for problem type, metric, and model selection")


class HyperparameterSearchPlan(BaseModel):
    model_name: str
    coarse_space: Dict[str, Any] = Field(description="Broad parameter distributions for stage 1 coarse search")
    fine_space: Dict[str, Any] = Field(description="Refined parameter search bounds for stage 2 Bayesian optimization")
    strategy_notes: str = Field(description="Guidance on key hyperparameters and tuning rationale")


class EvaluationResult(BaseModel):
    model_name: str
    model_family: str
    metrics: Dict[str, float] = Field(description="Validation set metrics e.g. accuracy, f1, precision, recall, roc_auc, or rmse, mae, r2")
    train_time_sec: float
    inference_time_ms: float
    model_size_kb: float
    best_hyperparameters: Dict[str, Any]
    preprocessing_summary: Dict[str, Any]
    status: str = Field(default="completed", description="'completed' or 'failed'")
    error: Optional[str] = None


class FinalTestReport(BaseModel):
    best_model_name: str
    best_model_family: str
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    metric_deltas: Dict[str, float] = Field(description="Difference between test and validation metrics to check overfitting")
    generalization_verdict: str = Field(description="Evaluation of whether the model generalizes well without overfitting")
    plots: Dict[str, str] = Field(default_factory=dict, description="Base64 PNG plots (confusion matrix, ROC, residuals, etc.)")
    executive_summary: str = Field(description="High-level senior ML engineer conclusions and deployment recommendation")


class WorkflowEvent(BaseModel):
    step_id: int
    step_name: str
    status: str = Field(description="'pending', 'running', 'completed', 'failed'")
    message: str
    payload: Optional[Dict[str, Any]] = None
    timestamp: str


class PredictionRow(BaseModel):
    id: Any
    predict: Any
    probability: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    total_rows: int
    preview: List[Dict[str, Any]]
    download_url: str
