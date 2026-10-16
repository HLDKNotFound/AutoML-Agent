"""
Report Agent.
Synthesizes validation vs test evaluation, inspects overfitting deltas with metric-direction awareness,
and generates structured executive ML summaries.
"""

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.ml_schemas import FinalTestReport
from backend.app.agents.llm_client import invoke_structured_llm
from backend.app.evaluation.metric_registry import get_metric_definition

logger = logging.getLogger(__name__)


class ExecutiveNarrative(BaseModel):
    summary: str = Field(description="Comprehensive executive summary explaining model viability, metrics, and deployment caveats")


def run_report_agent(
    best_model_name: str,
    best_model_family: str,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    plots: Dict[str, str],
    primary_metric: str,
    test_sample_size: Optional[int] = None,
    api_key: Optional[str] = None
) -> FinalTestReport:
    """
    Produces final evaluation report comparing validation vs test performance with
    correct metric direction semantics and without fabricated production-readiness claims.
    """
    metric_def = get_metric_definition(primary_metric)
    higher_is_better = metric_def.higher_is_better

    deltas = {}
    for metric_k, test_val in test_metrics.items():
        if metric_k in validation_metrics:
            deltas[metric_k] = round(test_val - validation_metrics[metric_k], 4)

    val_primary = float(validation_metrics.get(primary_metric, 0.0))
    test_primary = float(test_metrics.get(primary_metric, 0.0))
    raw_delta = test_primary - val_primary

    # Direction-aware relative delta calculation
    denom = abs(val_primary) if abs(val_primary) > 1e-6 else 1.0
    relative_change_pct = (raw_delta / denom) * 100.0

    # Direction-aware performance assessment
    # For maximization: positive raw_delta means test is better.
    # For minimization (errors): negative raw_delta means test has lower error (is better).
    test_is_better = raw_delta > 0 if higher_is_better else raw_delta < 0
    test_is_worse = raw_delta < 0 if higher_is_better else raw_delta > 0
    rel_abs_pct = abs(relative_change_pct)

    metric_name_upper = primary_metric.upper()

    if rel_abs_pct <= 7.5:
        generalization_verdict = (
            f"Consistent generalization across partitions. Primary metric {metric_name_upper} changed by "
            f"{raw_delta:+.4f} ({relative_change_pct:+.1f}%) from validation ({val_primary:.4f}) "
            f"to test ({test_primary:.4f}). This indicates stable performance without severe partition-specific overfitting."
        )
    elif test_is_worse:
        generalization_verdict = (
            f"Observed test partition degradation. Primary metric {metric_name_upper} degraded by "
            f"{abs(raw_delta):.4f} ({rel_abs_pct:.1f}% relative change) from validation ({val_primary:.4f}) "
            f"to test ({test_primary:.4f}). Regularization tuning and residual diagnostics are advised before deployment."
        )
    else:
        generalization_verdict = (
            f"Positive test partition divergence. Test {metric_name_upper} ({test_primary:.4f}) is superior to validation "
            f"({val_primary:.4f}) by {abs(raw_delta):.4f} ({rel_abs_pct:.1f}% relative change). "
            f"Note: Superior test performance alone does not prove production-readiness; variance should be monitored across batches."
        )

    sample_context = f" (N={test_sample_size})" if test_sample_size else ""
    deterministic_summary = (
        f"Automated ML pipeline selected champion model '{best_model_name}' ({best_model_family}). "
        f"On the held-out test partition{sample_context}, it achieved {metric_name_upper} of {test_primary:.4f} "
        f"compared to validation {metric_name_upper} of {val_primary:.4f} ({metric_def.direction} target). "
        f"{generalization_verdict} Evaluation is based on a fixed partition split. "
        f"Production deployment requires monitoring input distribution drift and periodic recalibration."
    )

    executive_summary = deterministic_summary

    # Optional LLM-assisted synthesis when API key/credentials are available
    try:
        prompt = f"""
        You are a Principal Machine Learning Engineer compiling an executive evaluation report.
        Best Model: {best_model_name} (Family: {best_model_family})
        Primary Metric: {primary_metric} (Direction: {metric_def.direction}, Higher is better: {higher_is_better})
        Validation Metric: {val_primary:.4f}
        Test Metric: {test_primary:.4f}
        Metric Deltas: {deltas}
        Generalization Assessment: {generalization_verdict}
        Test Sample Size: {test_sample_size or 'Not specified'}

        Provide a concise executive summary (3-4 sentences) that:
        1. Accurately reports validation vs test metrics using the correct metric direction.
        2. Acknowledges partition variance and avoids unconditional claims of production readiness.
        3. Identifies concrete monitoring requirements (drift detection, recalibration).
        """
        llm_res = invoke_structured_llm(
            prompt=prompt,
            output_schema=ExecutiveNarrative,
            system_instruction="You are a Principal ML Systems Auditor who prioritizes statistical rigor and honest reporting."
        )
        if llm_res and llm_res.summary and len(llm_res.summary.strip()) > 30:
            executive_summary = llm_res.summary.strip()
    except Exception as e:
        logger.debug("LLM narrative generation skipped: %s", e)

    return FinalTestReport(
        best_model_name=best_model_name,
        best_model_family=best_model_family,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        metric_deltas=deltas,
        generalization_verdict=generalization_verdict,
        plots=plots,
        executive_summary=executive_summary
    )
