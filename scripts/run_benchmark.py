"""
Automated AutoML-Agent Benchmark Harness.
Runs fair, reproducible benchmarks comparing AutoML champion pipelines against
heuristic Dummy baselines across classification and regression datasets.
Measures wall-clock training time, inference latency, memory footprint, and test generalization.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Set project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import config, DATA_DIR, ARTIFACTS_DIR
from backend.app.graph.workflow import create_ml_workflow
from backend.app.evaluation.metric_registry import get_metric_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkHarness")


BENCHMARK_SUITE = [
    {
        "name": "Customer Churn (Binary Classification)",
        "file": "churn.csv",
        "target": "churn",
        "task": "binary_classification",
        "primary_metric": "f1_macro"
    },
    {
        "name": "California Housing (Regression)",
        "file": "housing.csv",
        "target": "MedHouseVal",
        "task": "regression",
        "primary_metric": "rmse"
    },
    {
        "name": "Wine Cultivars (Multiclass Classification)",
        "file": "wine.csv",
        "target": "target",
        "task": "multiclass_classification",
        "primary_metric": "f1_macro"
    }
]


def run_benchmark_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    dataset_file = DATA_DIR / scenario["file"]
    if not dataset_file.exists():
        logger.error("Dataset %s missing from data directory.", scenario["file"])
        return {"status": "skipped", "error": f"File {scenario['file']} not found"}

    df = pd.read_csv(dataset_file)
    target_col = scenario["target"]
    features = [c for c in df.columns if c != target_col]
    session_id = f"bench_{scenario['file'].replace('.csv', '')}_{int(time.time())}"

    logger.info("=" * 70)
    logger.info("BENCHMARK SCENARIO: %s (%d rows, %d features)", scenario["name"], len(df), len(features))
    logger.info("=" * 70)

    events_log = []
    def event_collector(e):
        events_log.append(e)

    workflow = create_ml_workflow(event_emitter=event_collector)

    initial_state = {
        "session_id": session_id,
        "dataset_path": str(dataset_file),
        "dataframe_summary": {},
        "initial_features": features,
        "selected_features": features,
        "target_column": target_col,
        "data_quality_report": {},
        "feature_selection": {},
        "data_quality_iteration": 0,
        "problem_type": scenario["task"],
        "primary_metric": scenario["primary_metric"],
        "preprocessing_configs": {},
        "candidate_models": [],
        "ml_strategy_reasoning": "",
        "train_indices": [],
        "val_indices": [],
        "test_indices": [],
        "optimization_results": {},
        "model_results": [],
        "fitted_pipelines": {},
        "best_model": {},
        "test_results": {},
        "plots": {},
        "artifacts": {},
        "errors": [],
        "agent_decisions": [],
        "gemini_api_key": None,
        "status": "running"
    }

    t0 = time.perf_counter()
    final_state = workflow.invoke(initial_state)
    elapsed_time = time.perf_counter() - t0

    if final_state.get("status") == "failed":
        logger.error("Scenario %s failed: %s", scenario["name"], final_state.get("errors"))
        return {
            "scenario": scenario["name"],
            "status": "failed",
            "errors": final_state.get("errors")
        }

    # Extract Baseline vs Champion metrics
    p_metric = scenario["primary_metric"]
    metric_def = get_metric_definition(p_metric)
    all_results = final_state.get("model_results", [])

    baseline_entry = next((r for r in all_results if r.get("model_family") == "baseline"), None)
    champion_entry = final_state.get("best_model", {})
    test_metrics = final_state.get("test_results", {})

    baseline_val = baseline_entry["metrics"].get(p_metric, 0.0) if baseline_entry else 0.0
    champ_val = champion_entry.get("metrics", {}).get(p_metric, 0.0)
    champ_test = test_metrics.get(p_metric, 0.0)

    result_summary = {
        "scenario": scenario["name"],
        "file": scenario["file"],
        "rows": len(df),
        "features": len(features),
        "primary_metric": p_metric,
        "metric_direction": metric_def.direction,
        "baseline_val_metric": baseline_val,
        "champion_model": champion_entry.get("model_name"),
        "champion_family": champion_entry.get("model_family"),
        "champion_val_metric": champ_val,
        "champion_test_metric": champ_test,
        "inference_latency_ms": champion_entry.get("inference_time_ms", 0.0),
        "model_size_kb": champion_entry.get("model_size_kb", 0.0),
        "total_workflow_time_sec": round(elapsed_time, 2),
        "evaluated_models_count": len(all_results),
        "status": "success"
    }

    logger.info("Champion Selected: %s (%s)", result_summary["champion_model"], result_summary["champion_family"])
    logger.info("Validation %s: Champion=%.4f vs Baseline=%.4f (Direction: %s)",
                p_metric.upper(), champ_val, baseline_val, metric_def.direction)
    logger.info("Test %s: %.4f | Latency: %.2f ms | Footprint: %.1f KB",
                p_metric.upper(), champ_test, result_summary["inference_latency_ms"], result_summary["model_size_kb"])

    return result_summary


def main():
    logger.info("Starting AutoML-Agent Comprehensive Benchmark Suite...")
    benchmark_records = []

    for scn in BENCHMARK_SUITE:
        rec = run_benchmark_scenario(scn)
        benchmark_records.append(rec)

    # Save benchmark report artifact
    out_dir = ARTIFACTS_DIR / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"benchmark_run_{int(time.time())}.json"
    
    with open(report_file, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info("BENCHMARK EXECUTIVE SUMMARY TABLE")
    logger.info("=" * 90)
    header = f"{'Scenario':<32} | {'Metric':<10} | {'Baseline':<8} | {'Champion':<24} | {'Val Score':<10} | {'Test Score':<10} | {'Time (s)':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in benchmark_records:
        if r.get("status") == "success":
            logger.info(
                f"{r['scenario'][:32]:<32} | {r['primary_metric']:<10} | {r['baseline_val_metric']:<8.4f} | "
                f"{r['champion_model'][:24]:<24} | {r['champion_val_metric']:<10.4f} | {r['champion_test_metric']:<10.4f} | "
                f"{r['total_workflow_time_sec']:<8.1f}"
            )

    logger.info("=" * 90)
    logger.info("Benchmark report written to %s", report_file)


if __name__ == "__main__":
    main()
