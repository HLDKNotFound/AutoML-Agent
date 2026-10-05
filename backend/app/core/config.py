"""
Configuration module for the ML Automation Agent.
Centralizes all hyperparameters, thresholds, model limits, and paths.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
TEMPLATES_DIR = BACKEND_DIR / "templates"

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def _get_api_key() -> str:
    key = (
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GOOGLE_API_KEY") or
        ""
    ).strip()
    # Reject OAuth access tokens that cause 401 ACCESS_TOKEN_TYPE_UNSUPPORTED
    if key.startswith("AQ.") or key.startswith("ya29."):
        return ""
    return key


class SystemConfig(BaseModel):
    # LLM Settings
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    gemini_api_key: str = Field(default_factory=_get_api_key)
    llm_temperature: float = Field(default=0.2)
    
    # Reproducibility
    random_state: int = Field(default=42)
    
    # Data Splitting
    train_ratio: float = Field(default=0.70)
    validation_ratio: float = Field(default=0.15)
    test_ratio: float = Field(default=0.15)
    
    # Data Quality Loop
    max_data_quality_iterations: int = Field(default=3)
    
    # Model Selection
    n_candidate_models: int = Field(default=5)
    min_model_families: int = Field(default=3)
    
    # Hyperparameter Optimization
    coarse_search_trials: int = Field(default=6)
    fine_search_trials: int = Field(default=8)
    max_training_time_sec: int = Field(default=300)
    
    # Data Quality Thresholds
    high_missing_threshold: float = Field(default=0.50)
    high_cardinality_threshold: int = Field(default=50)
    collinearity_threshold: float = Field(default=0.90)
    min_samples_threshold: int = Field(default=30)
    outlier_iqr_multiplier: float = Field(default=2.5)

    # Server Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)


# Global Config Singleton
config = SystemConfig()
