"""
Artifact Management Service.
Handles serialization and packaging of full end-to-end pipelines (preprocessing + model),
metadata, schema definitions, and downloadable zip packages.
"""

import re
import sys
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import sklearn
from backend.app.core.config import ARTIFACTS_DIR, config


def _validate_safe_session_id(session_id: str) -> str:
    """Validates session_id against path traversal and strict alphanumeric format."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Invalid session_id: must be a non-empty string.")
    
    clean_id = session_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]+$", clean_id):
        raise ValueError(f"Invalid characters in session_id '{session_id}'. Only alphanumeric, underscores, and dashes allowed.")
    
    target_dir = (ARTIFACTS_DIR / clean_id).resolve()
    base_dir = ARTIFACTS_DIR.resolve()
    if not str(target_dir).startswith(str(base_dir)):
        raise PermissionError(f"Path traversal detected for session_id '{session_id}'.")
    return clean_id


def save_model_artifact(
    session_id: str,
    pipeline: Any,
    selected_features: List[str],
    target_column: str,
    problem_type: str,
    best_params: Dict[str, Any],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    training_report: Dict[str, Any],
    target_constraints: Optional[Dict[str, Any]] = None,
    split_config: Optional[Dict[str, Any]] = None,
    primary_metric: Optional[str] = None
) -> Path:
    """
    Saves complete model artifact bundle into backend/artifacts/{session_id}/
    with full reproducibility metadata and provenance.
    """
    clean_id = _validate_safe_session_id(session_id)
    artifact_dir = (ARTIFACTS_DIR / clean_id).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save complete pipeline (ColumnTransformer + Estimator)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Determine pipeline steps cleanly
    steps = []
    if hasattr(pipeline, "steps"):
        steps = [name for name, _ in pipeline.steps]
    elif hasattr(pipeline, "regressor") and hasattr(pipeline.regressor, "steps"):
        steps = [f"TransformedTarget({name})" for name, _ in pipeline.regressor.steps]
    elif hasattr(pipeline, "estimators"):
        steps = [f"EnsembleBlend({name})" for name, _ in pipeline.estimators]

    # 2. Save metadata.json with provenance
    metadata = {
        "artifact_version": "1.0.0",
        "session_id": clean_id,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "random_state": config.random_state,
        "problem_type": problem_type,
        "primary_metric": primary_metric or "",
        "target_column": target_column,
        "selected_features": selected_features,
        "best_hyperparameters": best_params,
        "target_constraints": target_constraints,
        "split_config": split_config or {
            "train_ratio": config.train_ratio,
            "validation_ratio": config.validation_ratio,
            "test_ratio": config.test_ratio,
            "random_state": config.random_state
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "pipeline_steps": steps,
        "model_class": type(pipeline).__name__
    }
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Save feature_schema.json
    feature_schema = {
        "expected_features": selected_features,
        "feature_count": len(selected_features),
        "target_column": target_column,
        "problem_type": problem_type
    }
    with open(artifact_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    # 4. Save training_report.json
    with open(artifact_dir / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2, default=str)

    # 5. Create compressed zip archive for user download
    zip_path = (ARTIFACTS_DIR / f"{clean_id}_model_artifact.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in artifact_dir.glob("*"):
            zipf.write(file, arcname=file.name)

    return artifact_dir


def load_model_artifact(session_id: str) -> Dict[str, Any]:
    """
    Loads saved model artifact and schema for inference with validation.
    """
    clean_id = _validate_safe_session_id(session_id)
    artifact_dir = (ARTIFACTS_DIR / clean_id).resolve()
    if not artifact_dir.exists() or not (artifact_dir / "model.joblib").exists():
        raise FileNotFoundError(f"Model artifact for session '{clean_id}' not found or incomplete.")

    pipeline = joblib.load(artifact_dir / "model.joblib")
    
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing for session '{clean_id}'.")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    schema_path = artifact_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema missing for session '{clean_id}'.")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "schema": schema,
        "artifact_dir": artifact_dir
    }
