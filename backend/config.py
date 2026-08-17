"""
Configuration module — centralizes all settings and environment variables.

Uses Pydantic Settings for validation and type safety.
All API keys and tunables are loaded from the .env file at the workspace root.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the .env file at the workspace root (two levels up from this file)
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API Keys ---
    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    NVIDIA_API_KEY: str

    # --- LangSmith (optional tracing) ---
    LANGSMITH_TRACING: bool = True
    LANGCHAIN_TRACING_V2: bool = True
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "Agentic_Rag_App"

    # --- LLM ---
    LLM_MODEL: str = "openai/gpt-oss-20b"
    LLM_TEMPERATURE: float = 0.0

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "nvidia/nemotron-3-embed-1b"

    # --- VectorStore ---
    VECTORSTORE_COLLECTION: str = "agentic-rag"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # --- Text Splitting ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Retrieval ---
    RETRIEVAL_K: int = 4

    # --- Web Search ---
    WEB_SEARCH_MAX_RESULTS: int = 3

    # --- Safety ---
    MAX_RETRY_COUNT: int = 3

    # --- Document Sources ---
    DOCUMENT_URLS: list[str] = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()  # type: ignore[call-arg]

    # Ensure API keys are also set as env vars (required by LangChain internals)
    os.environ.setdefault("GROQ_API_KEY", settings.GROQ_API_KEY)
    os.environ.setdefault("TAVILY_API_KEY", settings.TAVILY_API_KEY)
    os.environ.setdefault("NVIDIA_API_KEY", settings.NVIDIA_API_KEY)

    # Ensure LangSmith tracing variables are set in os.environ for LangChain internals
    api_key = settings.LANGSMITH_API_KEY or os.getenv("LANGCHAIN_API_KEY", "")
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT

    return settings
