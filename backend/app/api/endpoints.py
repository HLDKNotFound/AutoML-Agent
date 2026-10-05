"""
FastAPI Route Endpoints.
Handles file uploads, session state management, pipeline execution,
real-time SSE event streaming, artifact downloads, and batch predictions.
"""

import io
import os
import uuid
import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.app.core.config import config, DATA_DIR, ARTIFACTS_DIR
from backend.app.services.profiler import compute_dataset_profile
from backend.app.graph.workflow import create_ml_workflow
from backend.app.services.inference import run_batch_inference

router = APIRouter(prefix="/api")

# Max upload limit (50 MB)
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

# In-memory session store, event queues, and event history buffer
SESSIONS: Dict[str, Dict[str, Any]] = {}
EVENT_QUEUES: Dict[str, List[asyncio.Queue]] = {}
EVENT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


def cleanup_expired_sessions(max_age_seconds: int = 86400) -> int:
    """Removes sessions older than max_age_seconds to prevent memory leaks."""
    now = time.time()
    expired = [
        sid for sid, s in SESSIONS.items()
        if now - s.get("created_at", now) > max_age_seconds
    ]
    for sid in expired:
        SESSIONS.pop(sid, None)
        EVENT_QUEUES.pop(sid, None)
        EVENT_HISTORY.pop(sid, None)
    return len(expired)


class StartPipelineRequest(BaseModel):
    session_id: str
    target_column: str
    selected_features: Optional[List[str]] = None
    gemini_api_key: Optional[str] = None


class SampleLoadRequest(BaseModel):
    sample_name: str


class LLMConfigRequest(BaseModel):
    gemini_api_key: str


def _summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates preview metadata for uploaded or loaded CSVs."""
    total_rows, total_cols = df.shape
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in columns}
    missing = {c: int(df[c].isna().sum()) for c in columns}
    unique = {c: int(df[c].nunique()) for c in columns}
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    preview = df.head(10).fillna("").to_dict(orient="records")

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing,
        "unique_counts": unique,
        "memory_usage_kb": mem_kb,
        "preview": preview
    }


@router.get("/sample-datasets")
async def get_sample_datasets():
    """Lists pre-packaged sample datasets for instantaneous testing."""
    samples = [
        {
            "name": "churn.csv",
            "title": "Customer Churn Prediction",
            "type": "Binary Classification",
            "rows": 300,
            "target": "churn",
            "description": "Tabular telecom churn dataset featuring mixed numeric, categorical, and missing values."
        },
        {
            "name": "housing.csv",
            "title": "California Housing Prices",
            "type": "Regression",
            "rows": 250,
            "target": "MedHouseVal",
            "description": "Continuous real-estate valuation predicting median home value across regional features."
        },
        {
            "name": "wine.csv",
            "title": "Wine Quality Cultivar",
            "type": "Multiclass Classification",
            "rows": 178,
            "target": "target",
            "description": "Multiclass chemical analysis data classifying wines into 3 distinct cultivars."
        }
    ]
    return {"samples": samples}


@router.post("/load-sample")
async def load_sample(req: SampleLoadRequest):
    """Loads a pre-packaged sample CSV into a new pipeline session."""
    cleanup_expired_sessions()
    sample_file = DATA_DIR / req.sample_name
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail=f"Sample dataset '{req.sample_name}' not found.")

    session_id = f"sample_{req.sample_name.replace('.csv', '')}_{uuid.uuid4().hex[:8]}"
    dest_path = DATA_DIR / f"{session_id}.csv"
    
    df = pd.read_csv(sample_file)
    df.to_csv(dest_path, index=False)

    summary = _summarize_dataframe(df)
    now = time.time()
    SESSIONS[session_id] = {
        "session_id": session_id,
        "dataset_path": str(dest_path),
        "filename": req.sample_name,
        "summary": summary,
        "status": "ready",
        "created_at": now,
        "updated_at": now
    }

    return {"session_id": session_id, "summary": summary}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Receives user CSV upload, validates format, and prepares session."""
    cleanup_expired_sessions()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    file_path = DATA_DIR / f"{session_id}.csv"

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 2 columns and at least 1 row.")

    df.to_csv(file_path, index=False)
    summary = _summarize_dataframe(df)

    now = time.time()
    SESSIONS[session_id] = {
        "session_id": session_id,
        "dataset_path": str(file_path),
        "filename": file.filename,
        "summary": summary,
        "status": "ready",
        "created_at": now,
        "updated_at": now
    }

    return {"session_id": session_id, "summary": summary}


@router.post("/configure-llm")
async def configure_llm(req: LLMConfigRequest):
    """Allows setting, updating, or clearing Gemini API key dynamically."""
    api_key = req.gemini_api_key.strip()
    if not api_key:
        config.gemini_api_key = ""
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        return {"status": "success", "message": "Gemini API Key cleared. System switched to Offline Heuristics Mode."}

    if api_key.startswith("AQ.") or api_key.startswith("ya29."):
        raise HTTPException(
            status_code=400,
            detail="The key provided appears to be an OAuth access token (starts with AQ/ya29). Google AI Studio Gemini API keys start with 'AIzaSy'."
        )

    config.gemini_api_key = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    return {"status": "success", "message": "Gemini 2.5 Flash API Key updated successfully."}


def _run_workflow_background(session_id: str, state_input: Dict[str, Any], loop: Optional[asyncio.AbstractEventLoop] = None):
    """Background worker executing the compiled LangGraph workflow."""
    if session_id not in EVENT_HISTORY:
        EVENT_HISTORY[session_id] = []

    def event_emitter(event: Dict[str, Any]):
        EVENT_HISTORY[session_id].append(event)
        # Push event to all connected SSE clients thread-safely
        for q in list(EVENT_QUEUES.get(session_id, [])):
            try:
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except Exception:
                pass

    try:
        workflow = create_ml_workflow(event_emitter=event_emitter)
        final_state = workflow.invoke(state_input)
        if session_id in SESSIONS:
            SESSIONS[session_id]["workflow_state"] = final_state
            SESSIONS[session_id]["status"] = final_state.get("status", "completed")
            SESSIONS[session_id]["updated_at"] = time.time()
        
        event_emitter({
            "step_id": 13,
            "step_name": "Pipeline Complete",
            "status": "completed",
            "message": "Full automated machine learning workflow completed successfully!",
            "payload": {
                "best_model": final_state.get("best_model"),
                "test_results": final_state.get("test_results")
            },
            "timestamp": "Done"
        })
    except Exception as e:
        err_msg = f"Workflow runtime error: {str(e)}"
        if session_id in SESSIONS:
            SESSIONS[session_id]["status"] = "failed"
            SESSIONS[session_id]["error"] = err_msg
            SESSIONS[session_id]["updated_at"] = time.time()
        event_emitter({
            "step_id": -1,
            "step_name": "Execution Error",
            "status": "failed",
            "message": err_msg,
            "timestamp": "Error"
        })


@router.post("/start-pipeline")
async def start_pipeline(req: StartPipelineRequest, background_tasks: BackgroundTasks):
    """Initiates the end-to-end LangGraph ML workflow."""
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if req.gemini_api_key and req.gemini_api_key.strip():
        k = req.gemini_api_key.strip()
        if not (k.startswith("AQ.") or k.startswith("ya29.")):
            config.gemini_api_key = k
            os.environ["GEMINI_API_KEY"] = k

    df = pd.read_csv(session["dataset_path"])
    
    # Validation checks
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' does not exist in dataset.")

    features = req.selected_features or [c for c in df.columns if c != req.target_column]
    if req.target_column in features:
        features.remove(req.target_column)

    if not features:
        raise HTTPException(status_code=400, detail="At least 1 feature column must be selected.")

    if len(df) < config.min_samples_threshold:
        raise HTTPException(status_code=400, detail=f"Dataset must have at least {config.min_samples_threshold} samples.")

    # Initialize event queue list for SSE streaming
    EVENT_QUEUES[req.session_id] = []

    state_input = {
        "session_id": req.session_id,
        "dataset_path": session["dataset_path"],
        "dataframe_summary": {},
        "initial_features": features,
        "selected_features": features,
        "target_column": req.target_column,
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

    session["status"] = "running"
    session["target_column"] = req.target_column
    session["selected_features"] = features
    session["updated_at"] = time.time()

    loop = asyncio.get_running_loop()
    background_tasks.add_task(_run_workflow_background, req.session_id, state_input, loop)

    return {
        "status": "started",
        "session_id": req.session_id,
        "target_column": req.target_column,
        "feature_count": len(features)
    }


@router.get("/stream/{session_id}")
async def stream_workflow_events(session_id: str):
    """SSE endpoint streaming real-time pipeline events."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    queue: asyncio.Queue = asyncio.Queue()
    if session_id not in EVENT_QUEUES:
        EVENT_QUEUES[session_id] = []
    EVENT_QUEUES[session_id].append(queue)

    async def event_generator():
        try:
            # Replay any previous events immediately to client so early/late stages are never missed
            for past_ev in list(EVENT_HISTORY.get(session_id, [])):
                yield f"data: {json.dumps(past_ev)}\n\n"

            while True:
                # Wait for next event with a timeout for heartbeat keep-alive
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    is_terminal = (
                        event.get("step_id") in (13, -1) or
                        event.get("status") in ("failed", "error") or
                        event.get("step_name") in ("Pipeline Complete", "Execution Error")
                    )
                    if is_terminal:
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat ping to keep connection open
                    yield f": heartbeat\n\n"
        finally:
            if session_id in EVENT_QUEUES and queue in EVENT_QUEUES[session_id]:
                EVENT_QUEUES[session_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Retrieves full snapshot of current session state, metrics, charts, and agent logs."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    workflow_state = session.get("workflow_state", {})
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "status": session.get("status", "pending"),
        "summary": session.get("summary", {}),
        "target_column": session.get("target_column"),
        "selected_features": session.get("selected_features", []),
        "profile": workflow_state.get("dataframe_summary", {}),
        "data_quality_report": workflow_state.get("data_quality_report", {}),
        "feature_selection": workflow_state.get("feature_selection", {}),
        "candidate_models": workflow_state.get("candidate_models", []),
        "problem_type": workflow_state.get("problem_type", ""),
        "primary_metric": workflow_state.get("primary_metric", ""),
        "model_results": workflow_state.get("model_results", []),
        "best_model": workflow_state.get("best_model", {}),
        "test_results": workflow_state.get("test_results", {}),
        "plots": workflow_state.get("plots", {}),
        "agent_decisions": workflow_state.get("agent_decisions", []),
        "errors": workflow_state.get("errors", []),
        "error": session.get("error") or (workflow_state.get("errors", [None])[0] if workflow_state.get("errors") else None)
    }


@router.get("/models/{session_id}/download")
async def download_model_artifact(session_id: str):
    """Downloads zipped package with model.joblib, metadata.json, feature_schema.json."""
    zip_path = ARTIFACTS_DIR / f"{session_id}_model_artifact.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact archive not found or training incomplete.")

    return FileResponse(
        path=str(zip_path),
        filename=f"ml_model_{session_id}.zip",
        media_type="application/zip"
    )


@router.post("/predict/{session_id}")
async def upload_and_predict(session_id: str, file: UploadFile = File(...)):
    """Receives unlabeled CSV and runs batch inference using fitted frozen pipeline."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
        )

    try:
        df_input = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    try:
        pred_df, summary = run_batch_inference(session_id, df_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

    out_file = DATA_DIR / f"{session_id}_predictions.csv"
    pred_df.to_csv(out_file, index=False)

    preview = pred_df.head(15).to_dict(orient="records")
    return {
        "status": "success",
        "total_rows": len(pred_df),
        "preview": preview,
        "summary": summary,
        "download_url": f"/api/predict/{session_id}/download"
    }


@router.get("/predict/{session_id}/download")
async def download_predictions(session_id: str):
    """Downloads generated prediction CSV formatted as id,predict."""
    pred_file = DATA_DIR / f"{session_id}_predictions.csv"
    if not pred_file.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found.")

    return FileResponse(
        path=str(pred_file),
        filename=f"predictions_{session_id}.csv",
        media_type="text/csv"
    )
