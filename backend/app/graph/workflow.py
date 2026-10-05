"""
LangGraph Orchestration Workflow for the ML Automation Agent.
Implements state transitions, the iterative validation loop (max 3 cycles),
dataset partitioning, model-specific optimization, and final test evaluation.
"""

import time
import os
import json
from typing import Any, Callable, Dict, List, Optional, TypedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph, START, END

from backend.app.core.config import config, ARTIFACTS_DIR
from backend.app.schemas.ml_schemas import (
    DataQualityReport, FeatureSelectionResult, MLStrategy,
    EvaluationResult, FinalTestReport, PreprocessingStrategy
)
from backend.app.services.profiler import compute_dataset_profile
from backend.app.visualization.plots import (
    plot_target_distribution, plot_missing_values, plot_correlation_matrix,
    plot_model_comparison, plot_confusion_matrix, plot_roc_curve, plot_regression_residuals
)
from backend.app.agents.data_quality_agent import run_data_quality_agent
from backend.app.agents.feature_selection_agent import run_feature_selection_agent
from backend.app.agents.ml_strategy_agent import run_ml_strategy_agent
from backend.app.agents.hyperparameter_agent import get_search_plan_for_model
from backend.app.agents.report_agent import run_report_agent
from backend.app.optimization.optimizer import optimize_candidate_model
from backend.app.evaluation.evaluator import evaluate_pipeline
from backend.app.services.artifact_manager import save_model_artifact


class MLWorkflowState(TypedDict):
    session_id: str
    dataset_path: str
    dataframe_summary: Dict[str, Any]
    initial_features: List[str]
    selected_features: List[str]
    target_column: str

    data_quality_report: Dict[str, Any]
    feature_selection: Dict[str, Any]
    data_quality_iteration: int

    problem_type: str
    primary_metric: str
    preprocessing_configs: Dict[str, Any]
    candidate_models: List[Dict[str, Any]]
    ml_strategy_reasoning: str

    # Partitions saved in session state
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]

    optimization_results: Dict[str, Any]
    model_results: List[Dict[str, Any]]

    best_model: Dict[str, Any]
    test_results: Dict[str, Any]

    fitted_pipelines: Dict[str, Any]
    plots: Dict[str, str]
    artifacts: Dict[str, Any]
    errors: List[str]
    agent_decisions: List[Dict[str, Any]]
    status: str


def create_ml_workflow(event_emitter: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Constructs and compiles the complete LangGraph workflow.
    """
    
    def emit(step_id: int, step_name: str, status: str, message: str, payload: Any = None):
        if event_emitter:
            event_emitter({
                "step_id": step_id,
                "step_name": step_name,
                "status": status,
                "message": message,
                "payload": payload,
                "timestamp": time.strftime("%H:%M:%S")
            })

    # Node 1: load_dataset
    def load_dataset(state: MLWorkflowState) -> Dict[str, Any]:
        emit(1, "Upload Dataset", "running", "Loading dataset from CSV into memory...")
        df = pd.read_csv(state["dataset_path"])
        
        # Verify target exists
        target = state["target_column"]
        if target not in df.columns:
            err = f"Target column '{target}' not found in dataset."
            emit(1, "Upload Dataset", "failed", err)
            return {"errors": [err], "status": "failed"}

        # Validate selected features
        features = [f for f in state.get("initial_features", []) if f in df.columns and f != target]
        if not features:
            features = [c for c in df.columns if c != target]

        emit(1, "Upload Dataset", "completed", f"Dataset successfully loaded: {len(df)} rows, {len(df.columns)} columns.")
        return {
            "initial_features": features,
            "selected_features": features,
            "status": "loaded"
        }

    # Node 2: profile_dataset
    def profile_dataset(state: MLWorkflowState) -> Dict[str, Any]:
        emit(2, "Dataset Analysis", "running", "Computing comprehensive statistical profile...")
        df = pd.read_csv(state["dataset_path"])
        profile = compute_dataset_profile(
            df=df,
            target_column=state["target_column"],
            feature_columns=state["selected_features"]
        )

        # Generate exploratory visualizations
        plots = dict(state.get("plots", {}))
        target_plot = plot_target_distribution(df, state["target_column"], profile["suggested_problem_type"])
        if target_plot:
            plots["target_distribution"] = target_plot

        missing_plot = plot_missing_values(df, state["selected_features"])
        if missing_plot:
            plots["missing_values"] = missing_plot

        num_feats = profile.get("feature_types", {}).get("numeric", [])
        if len(num_feats) >= 2:
            corr_plot = plot_correlation_matrix(df, num_feats)
            if corr_plot:
                plots["correlation_matrix"] = corr_plot

        emit(2, "Dataset Analysis", "completed", "Dataset profiling and visualizations generated.", {"profile": profile})
        return {
            "dataframe_summary": profile,
            "plots": plots,
            "data_quality_iteration": state.get("data_quality_iteration", 0) + 1
        }

    # Node 3: data_quality_agent
    def data_quality_agent_node(state: MLWorkflowState) -> Dict[str, Any]:
        iteration = state.get("data_quality_iteration", 1)
        emit(3, "Data Quality", "running", f"LLM Data Analyst analyzing data quality (Iteration {iteration}/{config.max_data_quality_iterations})...")
        
        report: DataQualityReport = run_data_quality_agent(
            profile=state["dataframe_summary"],
            target_column=state["target_column"],
            feature_columns=state["selected_features"],
            iteration=iteration
        )

        decisions = list(state.get("agent_decisions", []))
        decisions.append({
            "agent": "DataQualityAgent",
            "decision": "Dataset Valid" if report.valid else "Dataset Needs Remediation",
            "reason": report.reasoning,
            "input_statistics": f"{len(state['selected_features'])} features, {len(report.problems)} problems identified",
            "output_configuration": f"Valid={report.valid}, Recommended features count={len(report.recommended_features)}"
        })

        emit(3, "Data Quality", "completed" if report.valid else "running", 
             f"Data quality audit completed. Valid={report.valid}, {len(report.problems)} problems identified.",
             {"quality_report": report.model_dump()})

        return {
            "data_quality_report": report.model_dump(),
            "agent_decisions": decisions
        }

    # Node 4: validate_data (Conditional Routing)
    def validate_data(state: MLWorkflowState) -> str:
        report_data = state.get("data_quality_report", {})
        is_valid = report_data.get("valid", False)
        iteration = state.get("data_quality_iteration", 1)

        if is_valid:
            return "split_dataset"
        elif iteration < config.max_data_quality_iterations:
            # Re-run data quality with updated feature recommendations
            return "reanalyze"
        else:
            return "validation_failed"

    # Remediation Node if invalid
    def handle_remediation(state: MLWorkflowState) -> Dict[str, Any]:
        iteration = state.get("data_quality_iteration", 1)
        report_data = state.get("data_quality_report", {})
        recommended = report_data.get("recommended_features", state["selected_features"])
        emit(3, "Data Quality", "running", f"Applying remediation from iteration {iteration}. Pruning problematic features...")
        return {
            "selected_features": recommended,
            "data_quality_iteration": iteration + 1
        }

    # Failed Node if 3 iterations exceeded
    def validation_failed_node(state: MLWorkflowState) -> Dict[str, Any]:
        report_data = state.get("data_quality_report", {})
        err = (
            f"Dataset failed automated validation after {config.max_data_quality_iterations} iterations. "
            f"Reasons: {[p['description'] for p in report_data.get('problems', [])]}. "
            f"Please inspect feature types, missingness, or target class labels."
        )
        emit(3, "Data Quality", "failed", err)
        return {
            "errors": [err],
            "status": "failed"
        }

    # Node 5: split_dataset
    def split_dataset(state: MLWorkflowState) -> Dict[str, Any]:
        emit(4, "Dataset Splitting", "running", "Splitting dataset strictly into 70% Train, 15% Validation, 15% Test...")
        df = pd.read_csv(state["dataset_path"])
        target_col = state["target_column"]
        y = df[target_col]
        indices = np.arange(len(df))

        stratify = None
        # Stratify if classification with sufficient counts
        if state["dataframe_summary"].get("target_stats", {}).get("type") in ("binary_classification", "multiclass_classification"):
            class_counts = y.value_counts()
            if (class_counts >= 2).all():
                stratify = y

        # 70% Train, 30% Temp (Val + Test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=config.random_state,
            stratify=stratify
        )

        stratify_temp = None
        if stratify is not None:
            stratify_temp = y.iloc[temp_idx]

        # 15% Validation, 15% Test
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=config.random_state,
            stratify=stratify_temp
        )

        emit(4, "Dataset Splitting", "completed", 
             f"Dataset partitioned: {len(train_idx)} Train (70%), {len(val_idx)} Validation (15%), {len(test_idx)} Test (15%).")

        return {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist()
        }

    # Node 6: feature_selection_agent
    def feature_selection_agent_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(5, "Feature Selection", "running", "Agent pruning confirmed redundant and leaking features...")
        quality_rep = DataQualityReport.model_validate(state["data_quality_report"])
        
        fs_res = run_feature_selection_agent(
            quality_report=quality_rep,
            initial_features=state["initial_features"]
        )

        decisions = list(state.get("agent_decisions", []))
        decisions.append({
            "agent": "FeatureSelectionAgent",
            "decision": f"Retained {len(fs_res.selected_features)} features",
            "reason": f"Dropped {len(fs_res.dropped_features)} features due to leakage, constant variance, or identifier nature.",
            "input_statistics": f"{len(state['initial_features'])} initial features",
            "output_configuration": f"Selected features: {fs_res.selected_features}"
        })

        emit(5, "Feature Selection", "completed", 
             f"Feature selection complete: {len(fs_res.selected_features)} features selected, {len(fs_res.dropped_features)} dropped.",
             {"feature_selection": fs_res.model_dump()})

        return {
            "selected_features": fs_res.selected_features,
            "feature_selection": fs_res.model_dump(),
            "agent_decisions": decisions
        }

    # Node 7: ml_strategy_agent
    def ml_strategy_agent_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(6, "ML Strategy", "running", "Determining problem type, evaluation metric, and 5 candidate models...")
        quality_rep = DataQualityReport.model_validate(state["data_quality_report"])
        
        strategy: MLStrategy = run_ml_strategy_agent(
            profile=state["dataframe_summary"],
            quality_report=quality_rep,
            selected_features=state["selected_features"],
            target_column=state["target_column"]
        )

        decisions = list(state.get("agent_decisions", []))
        model_names = [m.model_name for m in strategy.candidate_models]
        families = list({m.model_family for m in strategy.candidate_models})
        
        decisions.append({
            "agent": "MLStrategyAgent",
            "decision": f"Problem: {strategy.problem_type}, Metric: {strategy.primary_metric}",
            "reason": strategy.reasoning,
            "input_statistics": f"{len(state['selected_features'])} features, {strategy.problem_type}",
            "output_configuration": f"Models: {model_names} across families: {families}"
        })

        emit(6, "ML Strategy", "completed", 
             f"Strategy defined: {strategy.problem_type} with primary metric {strategy.primary_metric}. 5 models chosen across {len(families)} families.",
             {"strategy": strategy.model_dump()})

        return {
            "problem_type": strategy.problem_type,
            "primary_metric": strategy.primary_metric,
            "candidate_models": [m.model_dump() for m in strategy.candidate_models],
            "ml_strategy_reasoning": strategy.reasoning,
            "agent_decisions": decisions
        }

    # Node 8: preprocessing_agent
    def preprocessing_agent_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(7, "Preprocessing", "running", "Building model-specific preprocessing pipelines...")
        df = pd.read_csv(state["dataset_path"])
        sel_features = state["selected_features"]
        
        # Categorize feature types
        numeric = [c for c in sel_features if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in sel_features if c not in numeric]

        configs = {}
        for m in state["candidate_models"]:
            configs[m["model_name"]] = {
                "numeric_features": numeric,
                "categorical_features": categorical,
                "strategy": m["preprocessing_strategy"]
            }

        emit(7, "Preprocessing", "completed", 
             f"Model-specific preprocessing constructed: {len(numeric)} numeric, {len(categorical)} categorical features.")

        return {"preprocessing_configs": configs}

    # Node 9: model_selection_agent
    def model_selection_agent_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(8, "Model Selection", "completed", 
             f"Confirmed 5 candidate models across {len({m['model_family'] for m in state['candidate_models']})} distinct families.")
        return {}

    # Node 10: coarse_and_fine_optimization
    def optimize_models_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(9, "Hyperparameter Optimization", "running", "Beginning coarse broad search and fine Bayesian optimization for 5 models...")
        df = pd.read_csv(state["dataset_path"])
        train_df = df.iloc[state["train_indices"]]
        val_df = df.iloc[state["val_indices"]]

        X_train = train_df[state["selected_features"]]
        y_train = train_df[state["target_column"]]
        X_val = val_df[state["selected_features"]]
        y_val = val_df[state["target_column"]]

        num_feats = state["preprocessing_configs"][state["candidate_models"][0]["model_name"]]["numeric_features"]
        cat_feats = state["preprocessing_configs"][state["candidate_models"][0]["model_name"]]["categorical_features"]

        model_results = []
        fitted_pipelines = {}

        for idx, model_cand in enumerate(state["candidate_models"]):
            m_name = model_cand["model_name"]
            m_fam = model_cand["model_family"]
            prep_strat = PreprocessingStrategy.model_validate(model_cand["preprocessing_strategy"])
            search_plan = get_search_plan_for_model(m_name, state["problem_type"])

            emit(9, "Hyperparameter Optimization", "running", 
                 f"[{idx+1}/5] Optimizing {m_name} ({m_fam}). Coarse broad scan + Fine Bayesian tuning...")

            res = optimize_candidate_model(
                model_name=m_name,
                model_family=m_fam,
                problem_type=state["problem_type"],
                search_plan=search_plan,
                preprocessing_strategy=prep_strat,
                numeric_features=num_feats,
                categorical_features=cat_feats,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                primary_metric=state["primary_metric"]
            )

            fitted_pipelines[m_name] = res["best_pipeline"]

            # Save clean serializable evaluation result
            eval_entry = {
                "model_name": m_name,
                "model_family": m_fam,
                "metrics": res["validation_metrics"],
                "train_time_sec": res["train_time_sec"],
                "inference_time_ms": res["inference_time_ms"],
                "model_size_kb": res["model_size_kb"],
                "best_hyperparameters": res["best_params"],
                "preprocessing_summary": model_cand["preprocessing_strategy"],
                "status": "completed"
            }
            model_results.append(eval_entry)

        emit(9, "Hyperparameter Optimization", "completed", 
             f"Hyperparameter optimization completed for all 5 candidate models.")

        return {
            "model_results": model_results,
            "fitted_pipelines": fitted_pipelines
        }

    # Node 11: validation_evaluation
    def validation_evaluation_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(10, "Validation Evaluation", "running", "Benchmarking validation performance across all candidate models...")
        plots = dict(state.get("plots", {}))
        comp_plot = plot_model_comparison(state["model_results"], state["primary_metric"])
        if comp_plot:
            plots["model_comparison"] = comp_plot

        emit(10, "Validation Evaluation", "completed", "Validation comparison table and charts ready.", {"plots": plots})
        return {"plots": plots}

    # Node 12: select_best_model
    def select_best_model_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(10, "Select Best Model", "running", "Selecting highest-performing pipeline strictly on validation metrics...")
        p_metric = state["primary_metric"]
        p_type = state["problem_type"]
        results = state["model_results"]

        # Sort strictly on validation metric
        is_error_metric = p_metric in ("rmse", "mse", "mae", "mape", "log_loss")
        best = min(results, key=lambda r: r["metrics"].get(p_metric, float("inf"))) if is_error_metric \
            else max(results, key=lambda r: r["metrics"].get(p_metric, -float("inf")))

        decisions = list(state.get("agent_decisions", []))
        decisions.append({
            "agent": "ModelSelectionAgent",
            "decision": f"Selected Champion: {best['model_name']}",
            "reason": f"Achieved superior validation {p_metric.upper()} of {best['metrics'].get(p_metric, 0.0):.4f} with inference latency of {best['inference_time_ms']:.2f} ms.",
            "input_statistics": f"5 evaluated models: {[r['model_name'] for r in results]}",
            "output_configuration": f"Best parameters: {best['best_hyperparameters']}"
        })

        emit(10, "Select Best Model", "completed", 
             f"Champion model selected: {best['model_name']} with validation {p_metric.upper()}={best['metrics'].get(p_metric):.4f}.")

        return {
            "best_model": best,
            "agent_decisions": decisions
        }

    # Node 13: final_test_evaluation
    def final_test_evaluation_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(11, "Final Test Evaluation", "running", "Evaluating chosen pipeline on untouched 15% Test set...")
        best_name = state["best_model"]["model_name"]
        pipeline = state["fitted_pipelines"][best_name]

        df = pd.read_csv(state["dataset_path"])
        test_df = df.iloc[state["test_indices"]]
        X_test = test_df[state["selected_features"]]
        y_test = test_df[state["target_column"]]

        test_eval = evaluate_pipeline(
            pipeline=pipeline,
            X_val=X_test,
            y_val=y_test,
            problem_type=state["problem_type"]
        )

        plots = dict(state.get("plots", {}))
        is_classification = state["problem_type"] in ("binary_classification", "multiclass_classification")

        if is_classification:
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            y_true_np = y_test.to_numpy()
            y_pred_np = test_eval["y_pred"]
            classes = [str(c) for c in np.unique(y_true_np)]
            cm = confusion_matrix(y_true_np, y_pred_np)
            plots["test_confusion_matrix"] = plot_confusion_matrix(cm, classes)

            # ROC curve for binary classification
            if len(classes) == 2 and test_eval["y_prob"] is not None:
                from sklearn.metrics import roc_curve
                prob_pos = test_eval["y_prob"][:, 1] if test_eval["y_prob"].ndim == 2 else test_eval["y_prob"]
                fpr, tpr, _ = roc_curve(y_true_np, prob_pos)
                plots["test_roc_curve"] = plot_roc_curve(fpr, tpr, test_eval["metrics"].get("roc_auc", 0.0))
        else:
            # Regression residuals
            plots["test_residuals"] = plot_regression_residuals(y_test.to_numpy(), test_eval["y_pred"])

        emit(11, "Final Test Evaluation", "completed", 
             f"Final test evaluation completed on untouched partition: {state['primary_metric'].upper()}={test_eval['metrics'].get(state['primary_metric']):.4f}.")

        return {
            "test_results": test_eval["metrics"],
            "plots": plots
        }

    # Node 14: generate_report_and_save_artifact
    def generate_report_node(state: MLWorkflowState) -> Dict[str, Any]:
        emit(12, "Model Artifact", "running", "Packaging full pipeline, metadata, schema, and generating executive report...")
        best_name = state["best_model"]["model_name"]
        pipeline = state["fitted_pipelines"][best_name]

        report = run_report_agent(
            best_model_name=best_name,
            best_model_family=state["best_model"]["model_family"],
            validation_metrics=state["best_model"]["metrics"],
            test_metrics=state["test_results"],
            plots=state["plots"],
            primary_metric=state["primary_metric"]
        )

        # Save model artifact bundle
        training_report_payload = {
            "session_id": state["session_id"],
            "data_quality_report": state["data_quality_report"],
            "feature_selection": state["feature_selection"],
            "all_model_results": state["model_results"],
            "best_model": state["best_model"],
            "test_metrics": state["test_results"],
            "agent_decisions": state["agent_decisions"],
            "executive_summary": report.executive_summary
        }

        artifact_dir = save_model_artifact(
            session_id=state["session_id"],
            pipeline=pipeline,
            selected_features=state["selected_features"],
            target_column=state["target_column"],
            problem_type=state["problem_type"],
            best_params=state["best_model"]["best_hyperparameters"],
            validation_metrics=state["best_model"]["metrics"],
            test_metrics=state["test_results"],
            training_report=training_report_payload
        )

        emit(12, "Model Artifact", "completed", 
             f"Pipeline artifact saved successfully to {artifact_dir}. Ready for download and batch inference.",
             {"report": report.model_dump()})

        return {
            "artifacts": {
                "artifact_dir": str(artifact_dir),
                "zip_path": str(ARTIFACTS_DIR / f"{state['session_id']}_model_artifact.zip")
            },
            "status": "completed"
        }

    # Construct StateGraph
    builder = StateGraph(MLWorkflowState)

    builder.add_node("load_dataset", load_dataset)
    builder.add_node("profile_dataset", profile_dataset)
    builder.add_node("data_quality_agent", data_quality_agent_node)
    builder.add_node("handle_remediation", handle_remediation)
    builder.add_node("validation_failed", validation_failed_node)
    builder.add_node("split_dataset", split_dataset)
    builder.add_node("feature_selection_agent", feature_selection_agent_node)
    builder.add_node("ml_strategy_agent", ml_strategy_agent_node)
    builder.add_node("preprocessing_agent", preprocessing_agent_node)
    builder.add_node("model_selection_agent", model_selection_agent_node)
    builder.add_node("optimize_models", optimize_models_node)
    builder.add_node("validation_evaluation", validation_evaluation_node)
    builder.add_node("select_best_model", select_best_model_node)
    builder.add_node("final_test_evaluation", final_test_evaluation_node)
    builder.add_node("generate_report", generate_report_node)

    # Define edges
    builder.add_edge(START, "load_dataset")
    builder.add_edge("load_dataset", "profile_dataset")
    builder.add_edge("profile_dataset", "data_quality_agent")

    # Conditional Routing for Data Quality Loop (max 3 iterations)
    builder.add_conditional_edges(
        "data_quality_agent",
        validate_data,
        {
            "split_dataset": "split_dataset",
            "reanalyze": "handle_remediation",
            "validation_failed": "validation_failed"
        }
    )
    builder.add_edge("handle_remediation", "data_quality_agent")
    builder.add_edge("validation_failed", END)

    # Forward Pipeline Execution
    builder.add_edge("split_dataset", "feature_selection_agent")
    builder.add_edge("feature_selection_agent", "ml_strategy_agent")
    builder.add_edge("ml_strategy_agent", "preprocessing_agent")
    builder.add_edge("preprocessing_agent", "model_selection_agent")
    builder.add_edge("model_selection_agent", "optimize_models")
    builder.add_edge("optimize_models", "validation_evaluation")
    builder.add_edge("validation_evaluation", "select_best_model")
    builder.add_edge("select_best_model", "final_test_evaluation")
    builder.add_edge("final_test_evaluation", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()
