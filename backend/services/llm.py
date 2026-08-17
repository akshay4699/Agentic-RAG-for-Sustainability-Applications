"""
LLM service — initializes and caches the Groq LLM instance.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from backend.config import get_settings


@lru_cache()
def get_llm() -> ChatGroq:
    """Return a cached ChatGroq LLM instance."""
    settings = get_settings()
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
