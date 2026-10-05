"""
Artifact Management Service.
Handles serialization and packaging of full end-to-end pipelines (preprocessing + model),
metadata, schema definitions, and downloadable zip packages.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List
import joblib
import sklearn
from backend.app.core.config import ARTIFACTS_DIR, config


def save_model_artifact(
    session_id: str,
    pipeline: Any,
    selected_features: List[str],
    target_column: str,
    problem_type: str,
    best_params: Dict[str, Any],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    training_report: Dict[str, Any]
) -> Path:
    """
    Saves complete model artifact bundle into backend/artifacts/{session_id}/
    """
    artifact_dir = ARTIFACTS_DIR / session_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save complete pipeline (ColumnTransformer + Estimator)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # 2. Save metadata.json
    metadata = {
        "session_id": session_id,
        "scikit_learn_version": sklearn.__version__,
        "random_state": config.random_state,
        "problem_type": problem_type,
        "target_column": target_column,
        "selected_features": selected_features,
        "best_hyperparameters": best_params,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "pipeline_steps": [name for name, _ in pipeline.steps] if hasattr(pipeline, "steps") else []
    }
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Save feature_schema.json
    feature_schema = {
        "expected_features": selected_features,
        "feature_count": len(selected_features),
        "target_column": target_column
    }
    with open(artifact_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    # 4. Save training_report.json
    with open(artifact_dir / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2, default=str)

    # 5. Create compressed zip archive for user download
    zip_path = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in artifact_dir.glob("*"):
            zipf.write(file, arcname=file.name)

    return artifact_dir


def load_model_artifact(session_id: str) -> Dict[str, Any]:
    """
    Loads saved model artifact and schema for inference.
    """
    artifact_dir = ARTIFACTS_DIR / session_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Model artifact for session {session_id} not found.")

    pipeline = joblib.load(artifact_dir / "model.joblib")
    
    with open(artifact_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with open(artifact_dir / "feature_schema.json", "r") as f:
        schema = json.load(f)

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "schema": schema,
        "artifact_dir": artifact_dir
    }
