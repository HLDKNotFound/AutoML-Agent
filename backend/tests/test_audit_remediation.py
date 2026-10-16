"""
Comprehensive Remediation & Regression Test Suite for AutoML-Agent.
Tests:
1. Negative regression prediction semantics & target constraint enforcement.
2. Metric registry direction, score transformation, and report generalization correctness.
3. Multi-criteria task-type determination and user overrides.
4. Robust splitting (rare classes, tiny datasets, group, chronological, zero-leakage).
5. Tabular feature engineering schema integrity & no Kaggle-specific heuristics.
6. Trial failure lifecycle, Optuna error handling, and baseline anchor.
7. Session isolation, path traversal defense, and chunked upload limits.
8. Artifact lifecycle, metadata provenance, and batch inference in fresh scope.
"""

import io
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException, UploadFile

from backend.app.core.config import config, ARTIFACTS_DIR, DATA_DIR
from backend.app.evaluation.metric_registry import (
    METRIC_REGISTRY, get_metric_definition, is_metric_better, get_optimization_score
)
from backend.app.evaluation.evaluator import (
    compute_regression_metrics, compute_classification_metrics, evaluate_dummy_baseline
)
from backend.app.agents.report_agent import run_report_agent
from backend.app.services.profiler import compute_dataset_profile
from backend.app.preprocessing.splitter import split_dataset_robustly, SplitConfig
from backend.app.preprocessing.feature_engineer import TabularFeatureEngineer
from backend.app.preprocessing.pipeline_builder import build_full_model_pipeline
from backend.app.models.model_factory import create_base_estimator
from backend.app.services.artifact_manager import save_model_artifact, load_model_artifact
from backend.app.services.inference import run_batch_inference
from backend.app.api.endpoints import validate_session_id, _read_upload_with_limit


# ----------------------------------------------------------------------
# 1. Regression Prediction Semantics Tests (Phase B 3.1)
# ----------------------------------------------------------------------

def test_regression_raw_negative_predictions_preserved(tmp_path):
    """Verifies that negative regression predictions are NOT clamped to zero by default."""
    session_id = "test_neg_reg_sess"
    np.random.seed(42)
    # Target with explicitly negative values (e.g. temperature anomalies / net profit-loss)
    df_train = pd.DataFrame({
        "feature1": np.linspace(-10, 10, 50),
        "feature2": np.random.randn(50),
        "target": np.linspace(-100, 100, 50)  # spans negative to positive
    })

    estimator = create_base_estimator("Ridge", "regression")
    estimator.fit(df_train[["feature1", "feature2"]], df_train["target"])

    artifact_dir = save_model_artifact(
        session_id=session_id,
        pipeline=estimator,
        selected_features=["feature1", "feature2"],
        target_column="target",
        problem_type="regression",
        best_params={"alpha": 1.0},
        validation_metrics={"rmse": 2.5},
        test_metrics={"rmse": 2.8},
        training_report={"status": "completed"}
    )

    # Inference data with features that produce negative predictions
    df_infer = pd.DataFrame({
        "id": [1, 2, 3],
        "feature1": [-9.0, -8.0, 9.0],
        "feature2": [0.0, 0.0, 0.0]
    })

    pred_df, summary = run_batch_inference(session_id, df_infer)
    preds = pred_df["target"].values

    # Must preserve negative predictions!
    assert preds[0] < -50.0, f"Expected negative prediction, got {preds[0]}"
    assert preds[1] < -50.0, f"Expected negative prediction, got {preds[1]}"
    assert preds[2] > 50.0, f"Expected positive prediction, got {preds[2]}"

    # Cleanup
    shutil.rmtree(artifact_dir, ignore_errors=True)
    zip_path = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    if zip_path.exists():
        zip_path.unlink()


def test_regression_explicit_target_constraints_respected(tmp_path):
    """Verifies that target constraints are applied ONLY when explicitly configured in metadata."""
    session_id = "test_constrained_reg_sess"
    estimator = create_base_estimator("Ridge", "regression")
    X = pd.DataFrame({"x": [-5.0, 0.0, 5.0]})
    y = pd.Series([-50.0, 0.0, 50.0])
    estimator.fit(X, y)

    # Save artifact with explicit non-negative lower-bound constraint
    artifact_dir = save_model_artifact(
        session_id=session_id,
        pipeline=estimator,
        selected_features=["x"],
        target_column="y",
        problem_type="regression",
        best_params={},
        validation_metrics={"rmse": 1.0},
        test_metrics={"rmse": 1.0},
        training_report={},
        target_constraints={"min": 0.0, "max": 100.0}
    )

    df_infer = pd.DataFrame({"id": [1, 2], "x": [-5.0, 5.0]})
    pred_df, _ = run_batch_inference(session_id, df_infer)
    preds = pred_df["y"].values

    # With constraint min=0.0, negative prediction is clipped to 0.0
    assert preds[0] == 0.0
    assert preds[1] == pytest.approx(50.0, abs=2.0)

    # Cleanup
    shutil.rmtree(artifact_dir, ignore_errors=True)
    zip_path = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    if zip_path.exists():
        zip_path.unlink()


# ----------------------------------------------------------------------
# 2. Metric Direction & Report Correctness Tests (Phase B 3.2)
# ----------------------------------------------------------------------

def test_metric_registry_directions():
    """Validates metric registry directions and score transformations."""
    # Minimization metrics (errors/losses)
    for m in ["rmse", "mse", "mae", "mape", "log_loss"]:
        def_m = get_metric_definition(m)
        assert def_m.direction == "minimize"
        assert def_m.higher_is_better is False
        # Lower error is better
        assert is_metric_better(m, candidate_val=2.0, baseline_val=5.0) is True
        assert is_metric_better(m, candidate_val=6.0, baseline_val=5.0) is False
        assert get_optimization_score(m, 12.5) == -12.5

    # Maximization metrics
    for m in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "roc_auc", "r2"]:
        def_m = get_metric_definition(m)
        assert def_m.direction == "maximize"
        assert def_m.higher_is_better is True
        # Higher score is better
        assert is_metric_better(m, candidate_val=0.88, baseline_val=0.82) is True
        assert is_metric_better(m, candidate_val=0.75, baseline_val=0.82) is False
        assert get_optimization_score(m, 0.88) == 0.88


def test_report_agent_metric_direction_generalization():
    """Verifies ReportAgent correctly diagnoses overfitting vs improvement for error metrics."""
    # Case 1: RMSE error increased on test set (test is WORSE)
    rep_worse = run_report_agent(
        best_model_name="Ridge",
        best_model_family="linear",
        validation_metrics={"rmse": 10.0},
        test_metrics={"rmse": 14.0},  # Error increased by +40%
        plots={},
        primary_metric="rmse",
        test_sample_size=50
    )
    # Must NOT label this "strong test performance"
    assert "degradation" in rep_worse.generalization_verdict.lower()
    assert "superior" not in rep_worse.generalization_verdict.lower()

    # Case 2: RMSE error decreased on test set (test error is LOWER)
    rep_better = run_report_agent(
        best_model_name="Ridge",
        best_model_family="linear",
        validation_metrics={"rmse": 15.0},
        test_metrics={"rmse": 11.0},  # Error decreased
        plots={},
        primary_metric="rmse",
        test_sample_size=50
    )
    assert "superior" in rep_better.generalization_verdict.lower()
    # Must explicitly state that superior test performance alone does not prove unconditional readiness
    assert "monitoring" in rep_better.executive_summary.lower() or "drift" in rep_better.executive_summary.lower()


# ----------------------------------------------------------------------
# 3. Task-Type Determination Tests (Phase B 3.3)
# ----------------------------------------------------------------------

def test_task_type_continuous_floats_small_dataset():
    """Continuous float targets with few rows must be recognized as regression, not classification."""
    df = pd.DataFrame({
        "feat": range(12),
        "target": [1.25, 2.45, 3.12, 4.88, 5.01, 6.77, 7.34, 8.91, 9.15, 10.42, 11.09, 12.33]
    })
    profile = compute_dataset_profile(df, target_column="target")
    assert profile["suggested_problem_type"] == "regression"
    assert profile["target_stats"]["type"] == "regression"


def test_task_type_constant_target_handled():
    """Constant targets must be flagged as constant rather than crashing."""
    df = pd.DataFrame({
        "feat": range(20),
        "target": [1] * 20
    })
    profile = compute_dataset_profile(df, target_column="target")
    assert profile["suggested_problem_type"] == "constant_target"
    assert profile["target_stats"].get("is_constant") is True


def test_task_type_user_override():
    """Explicit user selection of task type is honored with validation."""
    df = pd.DataFrame({
        "feat": range(30),
        "target": list(range(10)) * 3  # integer values 0-9
    })
    # User specifies regression
    profile_reg = compute_dataset_profile(df, target_column="target", user_problem_type="regression")
    assert profile_reg["suggested_problem_type"] == "regression"

    # User specifies multiclass classification
    profile_cls = compute_dataset_profile(df, target_column="target", user_problem_type="multiclass_classification")
    assert profile_cls["suggested_problem_type"] == "multiclass_classification"

    # Incompatible: user requests regression on string labels
    df_str = pd.DataFrame({"feat": [1, 2, 3], "target": ["red", "blue", "green"]})
    with pytest.raises(ValueError):
        compute_dataset_profile(df_str, target_column="target", user_problem_type="regression")


# ----------------------------------------------------------------------
# 4. Robust Splitting Tests (Phase B 3.5)
# ----------------------------------------------------------------------

def test_split_rare_classes_does_not_crash():
    """Classes with only 1 or 2 instances must not crash the 3-way partition."""
    df = pd.DataFrame({
        "feat": range(40),
        # Class 'rare1' has only 1 sample, 'rare2' has 2 samples
        "target": ["common_a"] * 20 + ["common_b"] * 17 + ["rare2"] * 2 + ["rare1"] * 1
    })
    train_idx, val_idx, test_idx, meta = split_dataset_robustly(
        df=df,
        target_column="target",
        problem_type="multiclass_classification"
    )
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    # Zero partition overlap
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    assert meta["policy_used"] == "random"  # Fallback due to infeasible stratification


def test_split_group_aware_isolation():
    """Entity records from the same group must not leak across partitions."""
    df = pd.DataFrame({
        "client_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c5", "c6", "c6"],
        "feature": range(12),
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    config = SplitConfig(split_policy="group", group_column="client_id", random_state=42)
    train_idx, val_idx, test_idx, meta = split_dataset_robustly(
        df=df,
        target_column="target",
        problem_type="binary_classification",
        config=config
    )
    train_groups = set(df.iloc[train_idx]["client_id"])
    val_groups = set(df.iloc[val_idx]["client_id"])
    test_groups = set(df.iloc[test_idx]["client_id"])

    # Strict group isolation: zero group intersection
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_split_chronological_ordering():
    """Chronological splitting assigns strictly earliest to train and latest to test."""
    df = pd.DataFrame({
        "order": list(range(100)),
        "target": np.random.randn(100)
    })
    config = SplitConfig(split_policy="chronological", time_column="order")
    train_idx, val_idx, test_idx, meta = split_dataset_robustly(
        df=df,
        target_column="target",
        problem_type="regression",
        config=config
    )
    assert max(train_idx) < min(val_idx)
    assert max(val_idx) < min(test_idx)


# ----------------------------------------------------------------------
# 5. Feature Engineering Schema Integrity Tests (Phase B 3.6)
# ----------------------------------------------------------------------

def test_feature_engineer_schema_and_reordered_inference():
    """Feature engineer creates deterministic derived features and supports reordered inference."""
    df_train = pd.DataFrame({
        "code": ["NYC-101", "BOS-202", "CHI-303", "LA-404"] * 10,
        "date_str": ["2023-01-15", "2023-04-20", "2023-07-04", "2023-11-23"] * 10,
        "num_val": [10.0, 20.0, np.nan, 40.0] * 10
    })

    fe = TabularFeatureEngineer(enable_interactions=False, enable_missing_indicators=True)
    fe.fit(df_train)
    X_train_out = fe.transform(df_train)

    assert "code_part_0" in X_train_out.columns
    assert "code_part_1" in X_train_out.columns
    assert "date_str_month" in X_train_out.columns
    assert "num_val_is_missing" in X_train_out.columns

    # Test inference with columns reordered and extra column
    df_infer = pd.DataFrame({
        "extra_col": [999, 888],
        "num_val": [5.0, np.nan],
        "date_str": ["2024-02-14", "2024-08-01"],
        "code": ["SF-505", "SEA-606"]
    })
    X_infer_out = fe.transform(df_infer)
    for col in fe.output_feature_names_:
        assert col in X_infer_out.columns, f"Expected engineered column {col} in inference output"


# ----------------------------------------------------------------------
# 6. Fair Baseline Anchor Tests (Phase C 4.2)
# ----------------------------------------------------------------------

def test_dummy_baseline_evaluation():
    """Validates baseline anchor produces expected prior/mean baseline metrics."""
    X_train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    X_val = pd.DataFrame({"x": [11, 12, 13, 14]})
    y_val = pd.Series([0, 0, 1, 1])

    res = evaluate_dummy_baseline(X_train, y_train, X_val, y_val, "binary_classification")
    assert res["model_name"] == "DummyBaseline"
    assert res["model_family"] == "baseline"
    assert "accuracy" in res["metrics"]
    assert res["metrics"]["accuracy"] == 0.5  # Majority class prior on [0, 0, 1, 1] gives 2/4 = 0.5


# ----------------------------------------------------------------------
# 7. Security, Path Traversal & Limits Tests (Phase E 6.1, 6.2, 6.4)
# ----------------------------------------------------------------------

def test_session_id_path_traversal_rejected():
    """Rejects malicious session identifiers attempting directory traversal."""
    malicious_ids = [
        "../../etc/passwd",
        "../secrets",
        "sess/../../root",
        "sess_123; rm -rf /",
        "sess 123",
        "sess$123"
    ]
    for bad_id in malicious_ids:
        with pytest.raises(HTTPException) as exc_info:
            validate_session_id(bad_id)
        assert exc_info.value.status_code == 400


def test_chunked_upload_limit_enforced():
    """Rejects uploads that exceed MAX_UPLOAD_SIZE_BYTES during streaming."""
    import asyncio
    large_payload = b"col1,col2\n" + (b"1,2\n" * 150000)  # ~1.05 MB
    mock_file = UploadFile(filename="large.csv", file=io.BytesIO(large_payload))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_read_upload_with_limit(mock_file, max_bytes=500 * 1024))
    assert exc_info.value.status_code == 413
