"""
Inference Service.
Executes batch predictions on uploaded unlabeled datasets.
Guarantees schema validation and zero pipeline refitting during inference.
Outputs standardized CSV formatted as `id,predict`.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from backend.app.services.artifact_manager import load_model_artifact


def run_batch_inference(
    session_id: str,
    df_input: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes inference using the fitted model artifact.
    Preserves existing 'id' column or generates deterministic 1-based IDs.
    Returns (prediction_df, metadata_summary).
    """
    artifact = load_model_artifact(session_id)
    pipeline = artifact["pipeline"]
    schema = artifact["schema"]
    metadata = artifact["metadata"]
    expected_features = schema["expected_features"]

    # 1. Check feature schema
    missing_features = [f for f in expected_features if f not in df_input.columns]
    if missing_features:
        raise ValueError(f"Uploaded CSV is missing required features: {missing_features}")

    # 2. Extract or generate ID column
    id_col = None
    for cand in ["id", "ID", "Id", "id_col"]:
        if cand in df_input.columns:
            id_col = df_input[cand].values
            break
            
    if id_col is None:
        id_col = list(range(1, len(df_input) + 1))

    # 3. Extract feature matrix in precise order
    X_infer = df_input[expected_features]

    # 4. Predict using the full frozen pipeline (zero refitting)
    predictions = pipeline.predict(X_infer)

    # 5. Build output DataFrame
    output_df = pd.DataFrame({
        "id": id_col,
        "predict": predictions
    })

    # Optional: include class probabilities if classification
    if metadata.get("problem_type") in ("binary_classification", "multiclass_classification"):
        if hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(X_infer)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    output_df["probability_class_1"] = probs[:, 1].round(4)
            except Exception:
                pass

    summary = {
        "session_id": session_id,
        "total_predictions": len(output_df),
        "target_column": metadata.get("target_column"),
        "problem_type": metadata.get("problem_type"),
        "features_used": expected_features
    }

    return output_df, summary
