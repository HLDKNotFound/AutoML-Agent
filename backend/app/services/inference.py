"""
Inference Service.
Executes batch predictions on uploaded unlabeled datasets.
Guarantees schema validation and zero pipeline refitting during inference.
Outputs standardized CSV formatted as `id,predict`.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
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
    id_col_name = "id"
    for cand in ["PassengerId", "Id", "id", "ID", "id_col", "key", "index"]:
        if cand in df_input.columns:
            id_col = df_input[cand].values
            id_col_name = cand
            break
            
    if id_col is None:
        for c in df_input.columns:
            if c.lower().endswith("id"):
                id_col = df_input[c].values
                id_col_name = c
                break

    if id_col is None:
        id_col = list(range(1, len(df_input) + 1))
        id_col_name = "id"

    # 3. Extract feature matrix in precise order
    X_infer = df_input[expected_features]

    # 4. Predict using the full frozen pipeline (zero refitting)
    predictions = pipeline.predict(X_infer)

    # 5. Format predictions cleanly
    target_col = metadata.get("target_column") or "predict"
    p_type = metadata.get("problem_type")

    # If regression, ensure valid non-negative values for positive metrics like prices
    if p_type == "regression":
        predictions = np.clip(predictions, a_min=0.0, a_max=None)
    elif p_type == "binary_classification":
        # Check if target values were boolean
        if isinstance(predictions[0], (bool, np.bool_)):
            predictions = [bool(p) for p in predictions]

    # 6. Build output DataFrame preserving original ID name and target column
    output_df = pd.DataFrame({
        id_col_name: id_col,
        target_col: predictions
    })

    # Ensure backward compatibility aliases for clients expecting 'id' and 'predict'
    if "id" not in output_df.columns:
        output_df["id"] = id_col
    if "predict" not in output_df.columns:
        output_df["predict"] = predictions

    summary = {
        "session_id": session_id,
        "total_predictions": len(output_df),
        "target_column": metadata.get("target_column"),
        "problem_type": metadata.get("problem_type"),
        "features_used": expected_features
    }

    return output_df, summary
