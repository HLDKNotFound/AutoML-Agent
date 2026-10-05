"""
Data Quality Agent.
Analyzes dataset profile to detect data anomalies, target leakage, collinearity,
extreme missingness, cardinality, and imbalance.
Returns structured DataQualityReport schema.
"""

import json
import logging
from typing import Any, Dict, List
from backend.app.schemas.ml_schemas import DataQualityReport, DataProblem
from backend.app.agents.llm_client import invoke_structured_llm
from backend.app.core.config import config

logger = logging.getLogger(__name__)


def _sanitize_llm_report(
    report: DataQualityReport,
    target_column: str,
    feature_columns: List[str],
    total_rows: int,
    iteration: int
) -> DataQualityReport:
    """
    Sanitizes LLM outputs against ground-truth schema and boundaries.
    Prevents hallucinations (non-existent columns, keeping target column, etc.).
    """
    feature_set = set(feature_columns)
    
    # Filter recommended features to only valid initial features, excluding target
    sanitized_recommended = [
        f for f in report.recommended_features
        if f in feature_set and f != target_column
    ]
    # Remove duplicates preserving order
    sanitized_recommended = list(dict.fromkeys(sanitized_recommended))
    
    # Compute removed features consistently
    sanitized_removed = [
        f for f in feature_columns
        if f not in sanitized_recommended
    ]
    
    # Enforce dataset validity invariants
    valid = report.valid
    if len(sanitized_recommended) == 0:
        valid = False
    if total_rows > 0 and total_rows < config.min_samples_threshold:
        valid = False

    return DataQualityReport(
        valid=valid,
        problems=report.problems,
        warnings=report.warnings,
        recommended_features=sanitized_recommended,
        removed_features=sanitized_removed,
        reasoning=report.reasoning,
        iteration=iteration
    )


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
    dataset_stats = profile.get("dataset", {})
    total_rows = dataset_stats.get("total_rows", 0)

    # 1. Try LLM (Gemini) via LangChain with safe serialization & error handling
    try:
        # Use default=str to safely serialize any NumPy scalar types or special floats
        profile_json = json.dumps(profile, indent=2, default=str)
        prompt = f"""
    You are a Senior Data Scientist auditing a tabular dataset for an automated ML pipeline.
    Audit iteration: {iteration} of {config.max_data_quality_iterations}.
    
    Target column: {target_column}
    Selected features: {feature_columns}
    
    Dataset Profile Statistics:
    {profile_json}
    
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
       - Less than {config.min_samples_threshold} total rows
    """
        llm_report = invoke_structured_llm(
            prompt=prompt,
            output_schema=DataQualityReport,
            system_instruction="You are a Senior Data Scientist specialized in tabular ML data validation and leakage detection."
        )
        if llm_report is not None:
            return _sanitize_llm_report(
                report=llm_report,
                target_column=target_column,
                feature_columns=feature_columns,
                total_rows=total_rows,
                iteration=iteration
            )
    except Exception as e:
        logger.warning(f"Data quality LLM invocation failed: {e}. Falling back to deterministic heuristics.")

    # 2. Expert Heuristics Engine (Fallback / Reliable Deterministic Execution)
    problems: List[DataProblem] = []
    warnings: List[str] = []
    recommended_features: List[str] = list(feature_columns)
    removed_features: List[str] = []

    def drop_feature(feat: str) -> None:
        """Helper to safely remove a feature and record it without duplicates."""
        if feat in recommended_features:
            recommended_features.remove(feat)
        if feat not in removed_features:
            removed_features.append(feat)

    target_stats = profile.get("target_stats", {})
    num_stats = profile.get("numerical_stats", {})
    cat_stats = profile.get("categorical_stats", {})

    # Check 1: Row count minimum threshold & duplicate rows
    if total_rows < config.min_samples_threshold:
        problems.append(DataProblem(
            severity="critical",
            category="insufficient_samples",
            feature=None,
            description=f"Dataset has only {total_rows} rows, which is below the minimum threshold of {config.min_samples_threshold}.",
            suggestion="Collect more data points before training machine learning models."
        ))

    duplicate_rows = dataset_stats.get("duplicate_rows", 0)
    if duplicate_rows > 0:
        dup_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows ({dup_pct:.1f}%). Deduplication is advised to avoid train/test contamination.")

    # Check 2: Target validity and missing target values
    if not target_stats:
        problems.append(DataProblem(
            severity="critical",
            category="missing_target",
            feature=target_column,
            description=f"Target column '{target_column}' could not be analyzed or is missing.",
            suggestion="Select a valid non-empty target column."
        ))
    else:
        target_missing = target_stats.get("missing_count", 0)
        if target_missing > 0:
            problems.append(DataProblem(
                severity="critical",
                category="missing_target",
                feature=target_column,
                description=f"Target column '{target_column}' contains {target_missing} missing values.",
                suggestion="Drop rows with missing target values prior to model training."
            ))

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
        if feat:
            drop_feature(feat)

    # Check 4: Numerical Features (Constant, ID candidates, High missingness, Outliers)
    for feat, stats in num_stats.items():
        if stats.get("is_constant"):
            problems.append(DataProblem(
                severity="warning",
                category="constant_feature",
                feature=feat,
                description=f"Numerical feature '{feat}' has only 1 unique value (zero variance).",
                suggestion="Drop constant feature as it provides zero predictive information."
            ))
            drop_feature(feat)
        elif stats.get("is_id_candidate"):
            problems.append(DataProblem(
                severity="warning",
                category="id_column",
                feature=feat,
                description=f"Feature '{feat}' has 100% unique values across all {total_rows} rows, characteristic of a row identifier.",
                suggestion="Drop identifier feature to prevent memorization."
            ))
            drop_feature(feat)

        # Check high missingness
        missing_pct = stats.get("missing_pct", 0)
        if missing_pct > config.high_missing_threshold:
            pct = missing_pct * 100
            problems.append(DataProblem(
                severity="warning",
                category="missing_values",
                feature=feat,
                description=f"Numerical feature '{feat}' has {pct:.1f}% missing values (exceeds {config.high_missing_threshold * 100:.0f}% threshold).",
                suggestion="Drop feature or apply robust imputation."
            ))
            if missing_pct > 0.80:
                drop_feature(feat)

        # Check outliers
        outliers_pct = stats.get("outliers_pct", 0)
        if outliers_pct > 0.05 and abs(stats.get("skewness", 0)) > 1.5:
            problems.append(DataProblem(
                severity="info",
                category="outliers",
                feature=feat,
                description=f"Feature '{feat}' exhibits {outliers_pct * 100:.1f}% outliers with high skewness ({stats.get('skewness'):.2f}).",
                suggestion="Use robust scaling (RobustScaler) or quantile transformations."
            ))

    # Check 5: Categorical Features (Constant, ID candidates, High missingness, High cardinality)
    for feat, stats in cat_stats.items():
        unique_cats = stats.get("unique_categories", 0)
        # ID column candidate check for categorical string features
        is_cat_id = bool(unique_cats == total_rows and total_rows > 50)
        
        if stats.get("is_constant"):
            problems.append(DataProblem(
                severity="warning",
                category="constant_feature",
                feature=feat,
                description=f"Categorical feature '{feat}' has only 1 unique category.",
                suggestion="Drop constant feature."
            ))
            drop_feature(feat)
        elif is_cat_id:
            problems.append(DataProblem(
                severity="warning",
                category="id_column",
                feature=feat,
                description=f"Categorical feature '{feat}' has 100% unique string values across {total_rows} rows, characteristic of an identifier.",
                suggestion="Drop identifier feature to prevent memorization."
            ))
            drop_feature(feat)
        elif stats.get("high_cardinality"):
            problems.append(DataProblem(
                severity="warning",
                category="high_cardinality",
                feature=feat,
                description=f"Feature '{feat}' has {unique_cats} unique categories, indicating high cardinality.",
                suggestion="Use ordinal or target encoding rather than high-dimensional one-hot encoding."
            ))

        # Check high missingness for categorical
        cat_missing_pct = stats.get("missing_pct", 0)
        if cat_missing_pct > config.high_missing_threshold:
            pct = cat_missing_pct * 100
            problems.append(DataProblem(
                severity="warning",
                category="missing_values",
                feature=feat,
                description=f"Categorical feature '{feat}' has {pct:.1f}% missing values (exceeds {config.high_missing_threshold * 100:.0f}% threshold).",
                suggestion="Drop feature or impute with frequent category."
            ))
            if cat_missing_pct > 0.80:
                drop_feature(feat)

    # Check 6: Collinearity
    high_corr = profile.get("high_correlation_pairs", [])
    for pair in high_corr:
        f1, f2 = pair["feature1"], pair["feature2"]
        corr = pair["correlation"]
        problems.append(DataProblem(
            severity="warning",
            category="collinearity",
            feature=f"{f1}, {f2}",
            description=f"High linear correlation ({corr:.2f}) detected between '{f1}' and '{f2}'.",
            suggestion="Use regularized models (Ridge/Lasso) or tree ensembles with feature subsampling."
        ))
        warnings.append(f"High collinearity ({corr:.2f}) detected between '{f1}' and '{f2}'. Multicollinearity will be handled by regularized models or tree bagging.")

    # Determine validity
    has_critical_problems = any(
        p.severity == "critical" and p.category in ("insufficient_samples", "missing_target", "single_class_target")
        for p in problems
    )
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
