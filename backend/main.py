"""
FastAPI Application — the backend API for the Agentic RAG system.

Endpoints:
  GET  /api/health       — Health check
  POST /api/query        — Run a question through the Agentic RAG graph (JSON)
  POST /api/query/stream — Stream graph steps and LLM tokens (NDJSON)
  POST /api/index        — Trigger document re-indexing
"""

import json
import logging
import sqlite3
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.config import get_settings
from backend.graph.graph import build_graph
from backend.models import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from backend.services import vectorstore as vs_service

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Synchronous SQLite connection for non-streaming endpoint
_sync_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
_sync_checkpointer = SqliteSaver(_sync_conn)


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("Starting Agentic RAG Backend...")

    # Load settings (validates env vars immediately)
    settings = get_settings()
    logger.info(f"   LLM model:    {settings.LLM_MODEL}")
    logger.info(f"   Embed model:  {settings.EMBEDDING_MODEL}")
    logger.info(f"   Collection:   {settings.VECTORSTORE_COLLECTION}")

    # Initialize vectorstore (loads persisted or indexes from scratch)
    vs_service.get_retriever()

    # Build and compile the graph
    build_graph()

    logger.info("Backend ready!")
    yield
    logger.info("Shutting down Agentic RAG Backend.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG API",
    description="Production-grade Agentic RAG system with LangGraph, Groq, Chroma, Tavily, and SqliteSaver.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        vectorstore_ready=vs_service.is_ready(),
        graph_ready=True,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run a question through the Agentic RAG graph.

    Returns the answer along with sources and pipeline steps.
    """
    try:
        graph = build_graph(checkpointer=_sync_checkpointer)

        # Invoke the graph with checkpointer thread_id config
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_state = {
            "question": request.question,
            "generation": "",
            "documents": [],
            "web_search_needed": False,
            "retry_count": 0,
            "messages": [HumanMessage(content=request.question)],
        }

        # Run with step tracking
        steps: list[str] = []
        final_state = None

        for step_output in graph.stream(initial_state, config=config):
            for node_name, node_state in step_output.items():
                steps.append(node_name)
                final_state = node_state
                logger.info(f"Step: {node_name}")

        if final_state is None:
            raise HTTPException(status_code=500, detail="Graph produced no output")

        # Extract sources from documents
        sources = []
        docs = final_state.get("documents", [])
        for doc in docs:
            source = doc.metadata.get("source", "")
            if source and source not in sources:
                sources.append(source)

        # Determine the route used
        route = steps[0] if steps else "unknown"

        return QueryResponse(
            question=request.question,
            answer=final_state.get("generation", "No answer generated."),
            sources=sources,
            route=route,
            steps=steps,
            thread_id=request.thread_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream graph execution steps and LLM generation tokens as NDJSON.
    """
    async def event_generator():
        try:
            async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": request.thread_id}}
                initial_state = {
                    "question": request.question,
                    "generation": "",
                    "documents": [],
                    "web_search_needed": False,
                    "retry_count": 0,
                    "messages": [HumanMessage(content=request.question)],
                }
                steps: list[str] = []
                sources: list[str] = []
                final_generation = ""

                VALID_NODES = {
                    "retrieve",
                    "grade_documents",
                    "transform_query",
                    "web_search",
                    "generate",
                    "direct_generate",
                }

                async for event in graph.astream_events(initial_state, config=config, version="v2"):
                    kind = event.get("event")
                    name = event.get("name", "")

                    # Node execution started
                    if kind == "on_chain_start" and name in VALID_NODES:
                        if not steps or steps[-1] != name:
                            steps.append(name)
                            yield json.dumps({"type": "step", "step": name, "steps": list(steps)}) + "\n"

                    # LLM streaming tokens during generation nodes
                    elif kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            if steps and steps[-1] in {"generate", "direct_generate"}:
                                content = chunk.content
                                if isinstance(content, str) and content:
                                    final_generation += content
                                    yield json.dumps({"type": "token", "content": content}) + "\n"

                    # Node execution finished — collect documents
                    elif kind == "on_chain_end" and name in VALID_NODES:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            if "documents" in output and output["documents"]:
                                for doc in output["documents"]:
                                    src = getattr(doc, "metadata", {}).get("source", "")
                                    if src and src not in sources:
                                        sources.append(src)
                            if "generation" in output and output["generation"]:
                                final_generation = output["generation"]

                route = steps[0] if steps else "unknown"
                yield json.dumps({
                    "type": "done",
                    "answer": final_generation,
                    "sources": sources,
                    "route": route,
                    "steps": steps,
                    "thread_id": request.thread_id,
                }) + "\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Trigger document re-indexing."""
    try:
        urls = request.urls if request.urls else None
        chunks = vs_service.load_and_index_documents(urls)
        return IndexResponse(status="ok", chunks_indexed=chunks)
    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
