"""
LLM Client integration module.
Initializes ChatGoogleGenerativeAI (Gemini 2.5 Flash) via LangChain.
Provides structured output invocation with robust fallback for offline/test environments.
"""

import os
from typing import Any, Type, Optional
from pydantic import BaseModel
from backend.app.core.config import config


def get_llm():
    """
    Returns ChatGoogleGenerativeAI instance if API key is configured, else None.
    """
    api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            google_api_key=api_key,
            temperature=config.llm_temperature,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Warning: Failed to initialize ChatGoogleGenerativeAI: {e}")
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
        print(f"LLM invocation error: {e}")
        return None
