"""
Pydantic models for structured LLM outputs and API request/response schemas.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# LLM Structured Output Models
# ──────────────────────────────────────────────

class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search", "direct_llm"] = Field(
        ...,
        description=(
            "Choose: 'vectorstore' for domain-specific questions about National Greenhouse Accounts factors, "
            "emissions factors, carbon accounting, energy, scope 1/2/3 emissions, LLMs, or technical documents; "
            "'web_search' for current events or unknown topics; "
            "'direct_llm' for simple greetings or general knowledge."
        ),
    )



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for answer quality."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# ──────────────────────────────────────────────
# API Schemas
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming query from the frontend."""

    question: str = Field(..., min_length=1, description="The user's question")
    thread_id: str = Field("default_thread", description="Thread / session ID for conversation memory")


class QueryResponse(BaseModel):
    """Response returned to the frontend."""

    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    route: str = ""
    steps: list[str] = Field(default_factory=list)
    thread_id: str = "default_thread"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    vectorstore_ready: bool = False
    graph_ready: bool = False


class IndexRequest(BaseModel):
    """Optional: trigger re-indexing with custom URLs."""

    urls: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    """Re-indexing result."""

    status: str
    chunks_indexed: int = 0
