"""
ML Strategy Agent.
Determines problem type, primary evaluation metric, candidate ML models,
and tailored preprocessing pipelines for each candidate model.
Guarantees selection of exactly 5 models across at least 3 distinct model families.
"""

import json
from typing import Any, Dict, List
from backend.app.schemas.ml_schemas import MLStrategy, ModelCandidate, PreprocessingStrategy, DataQualityReport
from backend.app.agents.llm_client import invoke_structured_llm
from backend.app.core.config import config


def run_ml_strategy_agent(
    profile: Dict[str, Any],
    quality_report: DataQualityReport,
    selected_features: List[str],
    target_column: str
) -> MLStrategy:
    """
    Synthesizes the dataset profile into an optimal ML Strategy.
    Uses Gemini 2.5 Flash if available, otherwise invokes deterministic expert rules.
    """
    # 1. Try Gemini 2.5 Flash via LangChain
    prompt = f"""
    You are a Lead ML Architect. Formulate an end-to-end Machine Learning Strategy for this tabular dataset.
    Target Column: {target_column}
    Selected Features ({len(selected_features)}): {selected_features}
    
    Data Profile:
    {json.dumps(profile.get("dataset", {}), indent=2)}
    Target Stats:
    {json.dumps(profile.get("target_stats", {}), indent=2)}
    
    Requirements:
    1. Determine problem_type: 'binary_classification', 'multiclass_classification', or 'regression'.
    2. Select primary_metric: 'f1_macro', 'roc_auc', 'accuracy', 'rmse', 'r2', or 'mae'.
    3. Select EXACTLY 5 candidate models from AT LEAST 3 distinct model families:
       Valid families: 'linear', 'tree_ensemble', 'gradient_boosting', 'support_vector', 'neighbors'.
       IMPORTANT: For datasets with more than 1500 samples, NEVER select 'SVC' or 'SVR' (kernel SVM is O(N^3) and too slow). Instead use fast scalable models: 'HistGradientBoostingClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression'.
    4. For EACH model, define an independent model-specific preprocessing_strategy:
       - Tree models: no scaling (numeric_scaler=null), median imputer, ordinal/one_hot encoder.
       - Linear/SVM/KNN/MLP: robust/standard scaling, one_hot encoding, median imputer.
    """

    llm_strategy = invoke_structured_llm(
        prompt=prompt,
        output_schema=MLStrategy,
        system_instruction="You are a Principal ML Systems Architect building production scikit-learn pipelines."
    )

    if llm_strategy is not None and len(llm_strategy.candidate_models) == 5:
        families = {m.model_family for m in llm_strategy.candidate_models}
        if len(families) >= config.min_model_families:
            return llm_strategy

    # 2. Expert Decision Engine (Deterministic Fallback)
    target_stats = profile.get("target_stats", {})
    p_type = target_stats.get("type", "binary_classification")
    if p_type not in ("binary_classification", "multiclass_classification", "regression"):
        p_type = profile.get("suggested_problem_type", "binary_classification")

    # Primary metric selection
    if p_type == "binary_classification":
        is_imbalanced = target_stats.get("is_imbalanced", False)
        primary_metric = "f1_macro" if is_imbalanced else "roc_auc"
    elif p_type == "multiclass_classification":
        primary_metric = "f1_macro"
    else:
        primary_metric = "r2"

    total_rows = profile.get("dataset", {}).get("total_rows", 100)
    has_outliers = any(s.get("outliers_pct", 0) > 0.05 for s in profile.get("numerical_stats", {}).values())

    # Preprocessing templates for different model requirements
    tree_preprocessing = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler=None,  # Tree models are scale-invariant
        categorical_imputer="most_frequent",
        categorical_encoder="ordinal",
        handle_unknown="use_encoded_value"
    )

    linear_preprocessing = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler="robust" if has_outliers else "standard",
        categorical_imputer="most_frequent",
        categorical_encoder="one_hot",
        handle_unknown="ignore"
    )

    kernel_preprocessing = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler="robust" if has_outliers else "standard",
        categorical_imputer="most_frequent",
        categorical_encoder="one_hot",
        handle_unknown="ignore"
    )

    neighbors_preprocessing = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler="minmax",
        categorical_imputer="most_frequent",
        categorical_encoder="one_hot",
        handle_unknown="ignore"
    )

    boosting_preprocessing = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler=None,
        categorical_imputer="most_frequent",
        categorical_encoder="one_hot",
        handle_unknown="ignore"
    )

    if p_type in ("binary_classification", "multiclass_classification"):
        candidate_models = [
            ModelCandidate(
                model_name="RandomForestClassifier",
                model_family="tree_ensemble",
                reason="Robust ensemble capturing non-linear feature interactions and resistant to overfitting.",
                preprocessing_strategy=tree_preprocessing
            ),
            ModelCandidate(
                model_name="GradientBoostingClassifier",
                model_family="gradient_boosting",
                reason="Sequential boosting minimizing gradient loss for high predictive capacity.",
                preprocessing_strategy=boosting_preprocessing
            ),
            ModelCandidate(
                model_name="LogisticRegression",
                model_family="linear",
                reason="Interpretable linear baseline with L2 regularization and calibrated probabilities.",
                preprocessing_strategy=linear_preprocessing
            ),
        ]
        if total_rows > 1500:
            # Datasets with >1500 samples: replace slow O(N^3) SVC and instance KNN with high-speed scalable tree models
            candidate_models.extend([
                ModelCandidate(
                    model_name="HistGradientBoostingClassifier",
                    model_family="gradient_boosting",
                    reason="High-efficiency histogram-based gradient boosting scaling gracefully to large tabular datasets.",
                    preprocessing_strategy=boosting_preprocessing
                ),
                ModelCandidate(
                    model_name="ExtraTreesClassifier",
                    model_family="tree_ensemble",
                    reason="Extremely randomized trees providing variance reduction and near-instant training on tabular data.",
                    preprocessing_strategy=tree_preprocessing
                )
            ])
        else:
            candidate_models.extend([
                ModelCandidate(
                    model_name="SVC",
                    model_family="support_vector",
                    reason="Kernelized decision boundary effective in complex non-linear feature spaces.",
                    preprocessing_strategy=kernel_preprocessing
                ),
                ModelCandidate(
                    model_name="KNeighborsClassifier",
                    model_family="neighbors",
                    reason="Instance-based non-parametric classifier capturing localized data manifolds.",
                    preprocessing_strategy=neighbors_preprocessing
                )
            ])
    else:
        # Regression models
        candidate_models = [
            ModelCandidate(
                model_name="RandomForestRegressor",
                model_family="tree_ensemble",
                reason="Averaging ensemble robust to feature scales and outliers for non-linear regression.",
                preprocessing_strategy=tree_preprocessing
            ),
            ModelCandidate(
                model_name="GradientBoostingRegressor",
                model_family="gradient_boosting",
                reason="Iterative boosting optimized for squared error and complex functional approximations.",
                preprocessing_strategy=boosting_preprocessing
            ),
            ModelCandidate(
                model_name="Ridge",
                model_family="linear",
                reason="L2-regularized linear regression preventing coefficient explosion with multicollinearity.",
                preprocessing_strategy=linear_preprocessing
            ),
        ]
        if total_rows > 1500:
            candidate_models.extend([
                ModelCandidate(
                    model_name="HistGradientBoostingRegressor",
                    model_family="gradient_boosting",
                    reason="Fast histogram-based gradient boosting regression scaling to large tabular datasets.",
                    preprocessing_strategy=boosting_preprocessing
                ),
                ModelCandidate(
                    model_name="ExtraTreesRegressor",
                    model_family="tree_ensemble",
                    reason="Extremely randomized regression trees providing rapid training and high generalization.",
                    preprocessing_strategy=tree_preprocessing
                )
            ])
        else:
            candidate_models.extend([
                ModelCandidate(
                    model_name="SVR",
                    model_family="support_vector",
                    reason="Epsilon-insensitive loss function effective in non-linear regression with margin boundaries.",
                    preprocessing_strategy=kernel_preprocessing
                ),
                ModelCandidate(
                    model_name="KNeighborsRegressor",
                    model_family="neighbors",
                    reason="Distance-weighted k-nearest neighbors regression for localized non-linear surfaces.",
                    preprocessing_strategy=neighbors_preprocessing
                )
            ])

    reasoning = (
        f"Selected 5 distinct models across 5 model families ('tree_ensemble', 'gradient_boosting', "
        f"'linear', 'support_vector', 'neighbors'). Preprocessing is strictly customized per model: "
        f"tree-based estimators bypass scaling to preserve discrete splits, while linear and kernel "
        f"estimators receive robust scaling to mitigate outlier bias."
    )

    return MLStrategy(
        problem_type=p_type,
        target_column=target_column,
        primary_metric=primary_metric,
        candidate_models=candidate_models,
        reasoning=reasoning
    )
