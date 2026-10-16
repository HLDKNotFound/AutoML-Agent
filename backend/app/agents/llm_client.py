"""
LLM Client integration module.
Initializes ChatGoogleGenerativeAI (Gemini 2.5 Flash) via LangChain.
Provides structured output invocation with robust fallback for offline/test environments.
Supports per-request/per-session BYOK (Bring Your Own Key) isolation without global state leakage.
"""

import os
import logging
from typing import Any, Type, Optional
from pydantic import BaseModel
from backend.app.core.config import config

logger = logging.getLogger(__name__)


def get_llm(api_key: Optional[str] = None):
    """
    Returns ChatGoogleGenerativeAI instance if API key is configured and valid, else None.
    Accepts optional session-specific api_key for multi-user isolation.
    """
    effective_key = (api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not effective_key:
        return None

    # Detect OAuth access tokens that cause 401 ACCESS_TOKEN_TYPE_UNSUPPORTED
    if effective_key.startswith("AQ.") or effective_key.startswith("ya29."):
        logger.warning(
            "Configured API key appears to be an OAuth access token rather than a Google AI Studio API key (starts with 'AIzaSy'). "
            "Running seamlessly with the built-in deterministic Senior Data Scientist heuristics engine."
        )
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=effective_key,
            temperature=config.llm_temperature,
            max_retries=1
        )
        return llm
    except Exception as e:
        logger.warning("Failed to initialize ChatGoogleGenerativeAI: %s. Using heuristics fallback.", e)
        return None


def invoke_structured_llm(
    prompt: str,
    output_schema: Type[BaseModel],
    system_instruction: str = "You are a Senior Machine Learning Engineer and AI Data Scientist.",
    api_key: Optional[str] = None
) -> Optional[BaseModel]:
    """
    Invokes Gemini 2.5 Flash with structured Pydantic schema enforcement.
    Returns parsed Pydantic object or None if LLM is unavailable or fails.
    Isolates errors per invocation without mutating process-wide configuration.
    """
    llm = get_llm(api_key=api_key)
    if not llm:
        return None

    try:
        structured_llm = llm.with_structured_output(output_schema)
        messages = [
            ("system", system_instruction),
            ("human", prompt)
        ]
        result = structured_llm.invoke(messages)
        if isinstance(result, output_schema):
            return result
        elif isinstance(result, dict):
            return output_schema.model_validate(result)
        return None
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "UNAUTHENTICATED" in err_msg or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in err_msg:
            logger.warning(
                "Gemini API authentication failed (401 UNAUTHENTICATED): "
                "The key is invalid or is an OAuth access token instead of a Google AI Studio API key ('AIzaSy...'). "
                "Falling back seamlessly to built-in deterministic heuristics."
            )
        else:
            logger.warning("LLM invocation error: %s. Falling back to built-in heuristics.", e)
        return None
