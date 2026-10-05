"""
Data Quality Agent.
Analyzes dataset profile to detect data anomalies, target leakage, collinearity,
extreme missingness, cardinality, and imbalance.
Returns structured DataQualityReport schema.
"""

import json
from typing import Any, Dict, List
from backend.app.schemas.ml_schemas import DataQualityReport, DataProblem
from backend.app.agents.llm_client import invoke_structured_llm
from backend.app.core.config import config


def run_data_quality_agent(
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: List[str],
    iteration: int = 1
) -> DataQualityReport:
    """
    Executes Data Quality Analysis using Gemini 2.5 Flash if available,
    or our built-in Senior Data Scientist heuristics engine.
    """
    # 1. Try Gemini 2.5 Flash via LangChain
    prompt = f"""
    You are a Senior Data Scientist auditing a tabular dataset for an automated ML pipeline.
    Audit iteration: {iteration} of {config.max_data_quality_iterations}.
    
    Target column: {target_column}
    Selected features: {feature_columns}
    
    Dataset Profile Statistics:
    {json.dumps(profile, indent=2)}
    
    Responsibilities:
    1. Detect potential data-quality problems (leakage, extreme missing values, high cardinality, outliers, class imbalance).
    2. Detect possible target leakage (features identical to or suspiciously correlated with target).
    3. Identify suspicious ID columns or constant columns.
    4. Recommend features to drop and features to keep.
    5. Determine if the configuration is VALID for model training. (valid: true/false).
       The dataset is INVALID only if:
       - No valid features remain
       - Target column is missing or has < 2 classes for classification
       - Critical unresolved data leakage
       - Less than 30 total rows
    """
    
    llm_report = invoke_structured_llm(
        prompt=prompt,
        output_schema=DataQualityReport,
        system_instruction="You are a Senior Data Scientist specialized in tabular ML data validation and leakage detection."
    )
    
    if llm_report is not None:
        llm_report.iteration = iteration
        return llm_report

    # 2. Expert Heuristics Engine (Fallback / Reliable Deterministic Execution)
    problems: List[DataProblem] = []
    warnings: List[str] = []
    recommended_features: List[str] = list(feature_columns)
    removed_features: List[str] = []
    
    dataset_stats = profile.get("dataset", {})
    total_rows = dataset_stats.get("total_rows", 0)
    target_stats = profile.get("target_stats", {})
    num_stats = profile.get("numerical_stats", {})
    cat_stats = profile.get("categorical_stats", {})

    # Check 1: Row count minimum threshold
    if total_rows < config.min_samples_threshold:
        problems.append(DataProblem(
            severity="critical",
            category="insufficient_samples",
            feature=None,
            description=f"Dataset has only {total_rows} rows, which is below the minimum threshold of {config.min_samples_threshold}.",
            suggestion="Collect more data points before training machine learning models."
        ))

    # Check 2: Target validity
    if not target_stats:
        problems.append(DataProblem(
            severity="critical",
            category="missing_target",
            feature=target_column,
            description=f"Target column '{target_column}' could not be analyzed or is missing.",
            suggestion="Select a valid non-empty target column."
        ))
    else:
        if target_stats.get("type") in ("binary_classification", "multiclass_classification"):
            num_classes = target_stats.get("num_classes", 0)
            if num_classes < 2:
                problems.append(DataProblem(
                    severity="critical",
                    category="single_class_target",
                    feature=target_column,
                    description=f"Target column '{target_column}' has only {num_classes} unique class. Classification requires at least 2 distinct classes.",
                    suggestion="Ensure target column contains instances from at least 2 classes."
                ))
            if target_stats.get("is_imbalanced"):
                imbalance_ratio = target_stats.get("imbalance_ratio", 1.0)
                warnings.append(f"Target class imbalance detected (minority/majority ratio = {imbalance_ratio:.2f}). Class weighting and stratified splitting will be applied.")

    # Check 3: Potential Target Leakage
    potential_leakage = target_stats.get("potential_leakage", [])
    for leak in potential_leakage:
        feat = leak.get("feature")
        reason = leak.get("reason")
        problems.append(DataProblem(
            severity="critical",
            category="leakage",
            feature=feat,
            description=f"Feature '{feat}' exhibits potential target leakage: {reason}.",
            suggestion="Remove this feature immediately to avoid optimistic overfitting."
        ))
        if feat in recommended_features:
            recommended_features.remove(feat)
            removed_features.append(feat)

    # Check 4: Constant Features & ID Features
    for feat, stats in num_stats.items():
        if stats.get("is_constant"):
            problems.append(DataProblem(
                severity="warning",
                category="constant_feature",
                feature=feat,
                description=f"Numerical feature '{feat}' has only 1 unique value (zero variance).",
                suggestion="Drop constant feature as it provides zero predictive information."
            ))
            if feat in recommended_features:
                recommended_features.remove(feat)
                removed_features.append(feat)
        elif stats.get("is_id_candidate"):
            problems.append(DataProblem(
                severity="warning",
                category="id_column",
                feature=feat,
                description=f"Feature '{feat}' has 100% unique values across all {total_rows} rows, characteristic of a row identifier.",
                suggestion="Drop identifier feature to prevent memorization."
            ))
            if feat in recommended_features:
                recommended_features.remove(feat)
                removed_features.append(feat)

        # Check high missingness
        if stats.get("missing_pct", 0) > config.high_missing_threshold:
            pct = stats.get("missing_pct") * 100
            problems.append(DataProblem(
                severity="warning",
                category="missing_values",
                feature=feat,
                description=f"Feature '{feat}' has {pct:.1f}% missing values (exceeds {config.high_missing_threshold * 100}% threshold).",
                suggestion="Drop feature or apply robust imputation."
            ))
            if stats.get("missing_pct", 0) > 0.80 and feat in recommended_features:
                recommended_features.remove(feat)
                removed_features.append(feat)

    for feat, stats in cat_stats.items():
        if stats.get("is_constant"):
            problems.append(DataProblem(
                severity="warning",
                category="constant_feature",
                feature=feat,
                description=f"Categorical feature '{feat}' has only 1 unique category.",
                suggestion="Drop constant feature."
            ))
            if feat in recommended_features:
                recommended_features.remove(feat)
                removed_features.append(feat)
        elif stats.get("high_cardinality"):
            problems.append(DataProblem(
                severity="warning",
                category="high_cardinality",
                feature=feat,
                description=f"Feature '{feat}' has {stats.get('unique_categories')} unique categories, indicating high cardinality.",
                suggestion="Use ordinal or target encoding rather than high-dimensional one-hot encoding."
            ))

    # Check 5: Collinearity
    high_corr = profile.get("high_correlation_pairs", [])
    for pair in high_corr:
        f1, f2 = pair["feature1"], pair["feature2"]
        corr = pair["correlation"]
        warnings.append(f"High collinearity ({corr}) detected between '{f1}' and '{f2}'. Ridge regularization or tree bagging will manage multicollinearity.")

    # Determine validity
    has_critical_problems = any(p.severity == "critical" and p.category in ("insufficient_samples", "missing_target", "single_class_target") for p in problems)
    has_remaining_features = len(recommended_features) > 0

    valid = (not has_critical_problems) and has_remaining_features

    reasoning = (
        f"Data quality audit completed in iteration {iteration}. "
        f"Analyzed {len(feature_columns)} initial features. "
        f"Identified {len(problems)} data issues and {len(warnings)} warnings. "
        f"Recommended retaining {len(recommended_features)} predictive features "
        f"and pruning {len(removed_features)} problematic features (leakage, constants, IDs)."
    )

    return DataQualityReport(
        valid=valid,
        problems=problems,
        warnings=warnings,
        recommended_features=recommended_features,
        removed_features=removed_features,
        reasoning=reasoning,
        iteration=iteration
    )
