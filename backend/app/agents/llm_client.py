"""
LLM Client integration module.
Initializes ChatGoogleGenerativeAI (Gemini 2.5 Flash) via LangChain.
Provides structured output invocation with robust fallback for offline/test environments.
"""

import os
import logging
from typing import Any, Type, Optional
from pydantic import BaseModel
from backend.app.core.config import config

logger = logging.getLogger(__name__)


def get_llm():
    """
    Returns ChatGoogleGenerativeAI instance if API key is configured and valid, else None.
    """
    api_key = (config.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return None

    # Detect OAuth access tokens that cause 401 ACCESS_TOKEN_TYPE_UNSUPPORTED
    if api_key.startswith("AQ.") or api_key.startswith("ya29."):
        logger.warning(
            "Configured API key appears to be an OAuth access token rather than a Google AI Studio API key (starts with 'AIzaSy'). "
            "Running seamlessly with the built-in deterministic Senior Data Scientist heuristics engine."
        )
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=api_key,
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
    system_instruction: str = "You are a Senior Machine Learning Engineer and AI Data Scientist."
) -> Optional[BaseModel]:
    """
    Invokes Gemini 2.5 Flash with structured Pydantic schema enforcement.
    Returns parsed Pydantic object or None if LLM is unavailable or fails.
    """
    llm = get_llm()
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
                "Disabling LLM and running seamlessly on built-in deterministic Senior ML heuristics."
            )
            config.gemini_api_key = ""  # Disable to avoid repeated 401s on subsequent nodes
        else:
            logger.warning("LLM invocation error: %s. Falling back to built-in heuristics.", e)
        return None
