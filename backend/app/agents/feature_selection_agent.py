"""
Feature Selection Agent.
Performs feature filtering based on quality audit, correlation analysis, and predictive potential.
"""

from typing import Any, Dict, List
from backend.app.schemas.ml_schemas import FeatureSelectionResult, DataQualityReport


def run_feature_selection_agent(
    quality_report: DataQualityReport,
    initial_features: List[str]
) -> FeatureSelectionResult:
    """
    Finalizes selected features by excluding confirmed problem columns and pruning redundancies.
    """
    recommended = list(quality_report.recommended_features)
    removed = list(quality_report.removed_features)
    reasons = {}

    for prob in quality_report.problems:
        if prob.feature and prob.feature in removed:
            reasons[prob.feature] = f"{prob.category}: {prob.description}"

    # Ensure at least one feature is retained
    if not recommended and initial_features:
        # Fallback to keep at least the first non-removed feature if all were dropped
        for feat in initial_features:
            if feat not in reasons or "leakage" not in reasons[feat]:
                recommended.append(feat)
                if feat in removed:
                    removed.remove(feat)
                break

    return FeatureSelectionResult(
        selected_features=recommended,
        dropped_features=removed,
        reasons=reasons,
        importance_scores={}
    )
