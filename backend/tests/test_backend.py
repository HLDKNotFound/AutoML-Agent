"""
Comprehensive Test Suite for ML Automation Agent.
Tests profiling, agents, preprocessing, model factory, coarse-fine optimization,
LangGraph execution, serialization, inference, and end-to-end integration.
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.app.core.config import config, DATA_DIR, ARTIFACTS_DIR
from backend.app.services.profiler import compute_dataset_profile
from backend.app.schemas.ml_schemas import (
    DataQualityReport, PreprocessingStrategy, ModelCandidate, MLStrategy
)
from backend.app.agents.data_quality_agent import run_data_quality_agent
from backend.app.agents.feature_selection_agent import run_feature_selection_agent
from backend.app.agents.ml_strategy_agent import run_ml_strategy_agent
from backend.app.agents.hyperparameter_agent import get_search_plan_for_model
from backend.app.preprocessing.pipeline_builder import build_full_model_pipeline
from backend.app.models.model_factory import create_base_estimator
from backend.app.evaluation.evaluator import evaluate_pipeline
from backend.app.optimization.optimizer import run_coarse_search, run_fine_search
from backend.app.services.artifact_manager import save_model_artifact, load_model_artifact
from backend.app.services.inference import run_batch_inference
from backend.app.graph.workflow import create_ml_workflow


@pytest.fixture
def synthetic_classification_df(tmp_path):
    """Creates a synthetic classification dataset for testing."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 150000, size=n),
        "credit_score": np.random.normal(650, 50, size=n),
        "category": np.random.choice(["bronze", "silver", "gold"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    })
    # Inject missing values
    df.loc[1:5, "income"] = np.nan
    file_path = tmp_path / "test_classification.csv"
    df.to_csv(file_path, index=False)
    return df, str(file_path)


def test_data_profiler(synthetic_classification_df):
    df, _ = synthetic_classification_df
    profile = compute_dataset_profile(df, target_column="target")

    assert profile["dataset"]["total_rows"] == 120
    assert profile["dataset"]["total_columns"] == 6
    assert "income" in profile["numerical_stats"]
    assert profile["numerical_stats"]["income"]["missing_count"] > 0
    assert profile["target_stats"]["type"] == "binary_classification"
    assert profile["target_stats"]["num_classes"] == 2


def test_data_quality_agent(synthetic_classification_df):
    df, _ = synthetic_classification_df
    profile = compute_dataset_profile(df, target_column="target")
    
    report = run_data_quality_agent(
        profile=profile,
        target_column="target",
        feature_columns=["id_col", "age", "income", "credit_score", "category"],
        iteration=1
    )

    assert isinstance(report, DataQualityReport)
    assert report.valid is True
    # id_col should be flagged as an ID candidate or removed
    assert "id_col" in report.removed_features or any(p.category == "id_column" for p in report.problems)


def test_feature_selection_agent(synthetic_classification_df):
    df, _ = synthetic_classification_df
    profile = compute_dataset_profile(df, target_column="target")
    quality_report = run_data_quality_agent(
        profile=profile,
        target_column="target",
        feature_columns=["id_col", "age", "income", "credit_score", "category"]
    )
    
    fs_result = run_feature_selection_agent(
        quality_report=quality_report,
        initial_features=["id_col", "age", "income", "credit_score", "category"]
    )

    assert len(fs_result.selected_features) > 0
    assert "target" not in fs_result.selected_features


def test_ml_strategy_agent(synthetic_classification_df):
    df, _ = synthetic_classification_df
    profile = compute_dataset_profile(df, target_column="target")
    quality_report = run_data_quality_agent(profile, "target", ["age", "income", "category"])
    
    strategy = run_ml_strategy_agent(
        profile=profile,
        quality_report=quality_report,
        selected_features=["age", "income", "category"],
        target_column="target"
    )

    assert strategy.problem_type == "binary_classification"
    assert len(strategy.candidate_models) == 5
    families = {m.model_family for m in strategy.candidate_models}
    assert len(families) >= 3, "Must include at least 3 distinct model families"


def test_model_pipeline_and_optimization(synthetic_classification_df):
    df, _ = synthetic_classification_df
    X = df[["age", "income", "category"]]
    y = df["target"]
    
    X_train, X_val = X.iloc[:80], X.iloc[80:]
    y_train, y_val = y.iloc[:80], y.iloc[80:]

    strategy = PreprocessingStrategy(
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="most_frequent",
        categorical_encoder="one_hot"
    )

    search_plan = get_search_plan_for_model("RandomForestClassifier", "binary_classification")
    
    # Test Coarse Search
    coarse = run_coarse_search(
        model_name="RandomForestClassifier",
        problem_type="binary_classification",
        search_plan=search_plan,
        preprocessing_strategy=strategy,
        numeric_features=["age", "income"],
        categorical_features=["category"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        primary_metric="f1_macro",
        n_trials=3
    )

    assert coarse["best_pipeline"] is not None
    assert len(coarse["trials_history"]) == 3

    # Test Fine Search
    fine = run_fine_search(
        model_name="RandomForestClassifier",
        problem_type="binary_classification",
        coarse_best_params=coarse["best_params"],
        search_plan=search_plan,
        preprocessing_strategy=strategy,
        numeric_features=["age", "income"],
        categorical_features=["category"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        primary_metric="f1_macro",
        n_trials=3
    )

    assert fine["best_pipeline"] is not None


def test_artifact_serialization_and_inference(synthetic_classification_df, tmp_path):
    df, _ = synthetic_classification_df
    session_id = "test_sess_001"
    
    # Train simple pipeline
    strategy = PreprocessingStrategy(numeric_imputer="median", numeric_scaler="robust", categorical_encoder="one_hot")
    estimator = create_base_estimator("LogisticRegression", "binary_classification")
    pipeline = build_full_model_pipeline(
        estimator,
        numeric_features=["age", "income"],
        categorical_features=["category"],
        strategy=strategy
    )
    pipeline.fit(df[["age", "income", "category"]], df["target"])

    # Save artifact
    artifact_dir = save_model_artifact(
        session_id=session_id,
        pipeline=pipeline,
        selected_features=["age", "income", "category"],
        target_column="target",
        problem_type="binary_classification",
        best_params={"C": 1.0},
        validation_metrics={"accuracy": 0.85},
        test_metrics={"accuracy": 0.82},
        training_report={"status": "ok"}
    )

    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "feature_schema.json").exists()

    # Test batch inference with new unlabeled DataFrame
    unlabeled_df = pd.DataFrame({
        "id": [101, 102, 103],
        "age": [25, 45, 60],
        "income": [35000, 85000, 120000],
        "category": ["silver", "gold", "bronze"]
    })

    pred_df, summary = run_batch_inference(session_id, unlabeled_df)
    assert len(pred_df) == 3
    assert "id" in pred_df.columns
    assert "predict" in pred_df.columns
    assert list(pred_df["id"]) == [101, 102, 103]

    # Clean up test artifact
    shutil.rmtree(artifact_dir, ignore_errors=True)
    zip_path = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    if zip_path.exists():
        zip_path.unlink()


def test_full_langgraph_workflow_end_to_end(synthetic_classification_df):
    """Integration test running the entire LangGraph workflow."""
    _, file_path = synthetic_classification_df
    session_id = "test_e2e_session"

    events = []
    def event_collector(e):
        events.append(e)

    workflow = create_ml_workflow(event_emitter=event_collector)

    initial_state = {
        "session_id": session_id,
        "dataset_path": file_path,
        "dataframe_summary": {},
        "initial_features": ["age", "income", "credit_score", "category"],
        "selected_features": ["age", "income", "credit_score", "category"],
        "target_column": "target",
        "data_quality_report": {},
        "feature_selection": {},
        "data_quality_iteration": 0,
        "problem_type": "",
        "primary_metric": "",
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
        "status": "running"
    }

    final_state = workflow.invoke(initial_state)

    assert final_state["status"] == "completed"
    assert len(final_state["model_results"]) == 5
    assert "best_model" in final_state
    assert final_state["best_model"]["model_name"] != ""
    assert "test_results" in final_state
    assert len(events) > 10

    # Clean up
    art_dir = ARTIFACTS_DIR / session_id
    if art_dir.exists():
        shutil.rmtree(art_dir, ignore_errors=True)
    zip_file = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    if zip_file.exists():
        zip_file.unlink()


def test_data_quality_agent_edge_cases():
    """Verifies edge case handling in data quality agent heuristics and sanitization."""
    profile = {
        "dataset": {
            "total_rows": 100,
            "total_columns": 5,
            "duplicate_rows": 10,
            "missing_cells": 15
        },
        "target_stats": {
            "type": "binary_classification",
            "num_classes": 2,
            "missing_count": 5
        },
        "numerical_stats": {
            "outlier_col": {
                "outliers_pct": 0.12,
                "skewness": 2.5,
                "missing_pct": 0.0,
                "is_constant": False,
                "is_id_candidate": False
            }
        },
        "categorical_stats": {
            "cat_id": {
                "unique_categories": 100,
                "missing_pct": 0.0,
                "is_constant": False,
                "high_cardinality": True
            },
            "high_missing_cat": {
                "unique_categories": 3,
                "missing_pct": 0.90,
                "is_constant": False,
                "high_cardinality": False
            }
        },
        "high_correlation_pairs": [
            {"feature1": "f1", "feature2": "f2", "correlation": 0.92}
        ]
    }

    report = run_data_quality_agent(
        profile=profile,
        target_column="target",
        feature_columns=["cat_id", "high_missing_cat", "outlier_col", "f1", "f2"],
        iteration=1
    )

    assert isinstance(report, DataQualityReport)
    # Target missing should be flagged
    assert any(p.category == "missing_target" for p in report.problems)
    # Cat ID should be removed
    assert "cat_id" in report.removed_features
    # Extreme missing categorical (> 80%) should be dropped
    assert "high_missing_cat" in report.removed_features
    # Duplicate rows warning
    assert any("duplicate rows" in w.lower() for w in report.warnings)
    # Collinearity problem
    assert any(p.category == "collinearity" for p in report.problems)
    # Outliers problem
    assert any(p.category == "outliers" for p in report.problems)


def test_dataframe_cache_and_pipeline_registry(synthetic_classification_df):
    from backend.app.graph.workflow import get_cached_dataframe, clear_dataframe_cache, _SESSION_PIPELINES, get_session_pipelines
    _, file_path = synthetic_classification_df
    clear_dataframe_cache()

    df1 = get_cached_dataframe(file_path)
    df2 = get_cached_dataframe(file_path)
    assert df1 is df2, "Cached DataFrame should return the exact same instance in memory"

    clear_dataframe_cache(file_path)
    df3 = get_cached_dataframe(file_path)
    assert len(df3) == 120

    _SESSION_PIPELINES["sess_test_123"] = {"mock_model": object()}
    assert "mock_model" in get_session_pipelines("sess_test_123")
    _SESSION_PIPELINES.pop("sess_test_123", None)


def test_endpoints_improvements():
    import asyncio
    import time
    from backend.app.api.endpoints import configure_llm, LLMConfigRequest, cleanup_expired_sessions, SESSIONS
    from fastapi import HTTPException

    # Test OAuth token rejects with 400
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(configure_llm(LLMConfigRequest(gemini_api_key="AQ.Ab8RN6Igaw_EIZkPHL9k16OGydetx4r6mg3tyuKDLNVeZQxnaQV")))
    assert exc_info.value.status_code == 400

    # Test empty API key clears key successfully
    res = asyncio.run(configure_llm(LLMConfigRequest(gemini_api_key="   ")))
    assert res["status"] == "success"

    # Test session cleanup
    SESSIONS["old_sess"] = {"created_at": time.time() - 100000}
    SESSIONS["fresh_sess"] = {"created_at": time.time()}
    cleaned = cleanup_expired_sessions(max_age_seconds=3600)
    assert cleaned >= 1
    assert "old_sess" not in SESSIONS
    assert "fresh_sess" in SESSIONS
    SESSIONS.pop("fresh_sess", None)


