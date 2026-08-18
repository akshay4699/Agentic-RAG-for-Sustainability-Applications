# Agentic RAG — Production App

A **production-grade** Agentic RAG (Retrieval-Augmented Generation) system built with **FastAPI** + **Streamlit**.

## Architecture

```
START → Route Question → [Vectorstore / Web Search / Direct LLM]
  → Retrieve Documents → Grade Documents → Decide Fallback
  → Generate Answer → Hallucination Check → Answer Quality → END
```

**Tech Stack:** LangGraph · Groq (LLM) · Pinecone (VectorStore) · NVIDIA (Embeddings) · Tavily (Web Search) · FastAPI · Streamlit

## Project Structure

```
agentic_rag_app/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py             # Settings & environment
│   ├── models.py             # Pydantic models
│   ├── state.py              # GraphState
│   ├── ingest_pinecone.py    # Pinecone document ingestion script
│   ├── services/
│   │   ├── llm.py            # Groq LLM
│   │   ├── vectorstore.py    # Pinecone vectorstore
│   │   └── search.py         # Tavily web search
│   ├── chains/
│   │   └── chains.py         # All LLM chains
│   └── graph/
│       ├── nodes.py           # Graph node functions
│       ├── edges.py           # Decision functions
│       └── graph.py           # Graph assembly
├── data/                     # PDF documents folder
├── frontend/
│   └── streamlit_app.py       # Chat UI
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.11+
- API keys for: **Groq**, **Tavily**, **NVIDIA**, **Pinecone**

### 2. Install Dependencies

```bash
cd agentic_rag_app
pip install -r requirements.txt
```

### 3. Configure Environment

Ensure your `.env` file (at the workspace root) contains:

```env
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
NVIDIA_API_KEY=your-nvidia-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=agentic-rag-au-emission-factor-app
```

### 4. Index PDF Documents into Pinecone

Run the Pinecone ingestion script to index all PDFs from `agentic_rag_app/data`:

```bash
cd agentic_rag_app
python -m backend.ingest_pinecone
```

### 5. Start the Backend

```bash
cd agentic_rag_app
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```


On first run, the backend will:
- Validate all API keys
- Load and index documents into Chroma (persisted to `./chroma_db`)
- Compile the LangGraph pipeline

### 5. Start the Frontend

```bash
cd agentic_rag_app
streamlit run frontend/streamlit_app.py
```

## API Endpoints

| Method | Endpoint       | Description                    |
|--------|---------------|--------------------------------|
| GET    | `/api/health` | Health check                   |
| POST   | `/api/query`  | Run a question through the RAG |
| POST   | `/api/index`  | Trigger document re-indexing   |

### Example Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are LLM agents?"}'
```

## Features

- **Intelligent Routing** — questions auto-routed to vectorstore, web search, or direct LLM
- **Document Retrieval** — Chroma vectorstore with NVIDIA embeddings
- **Relevance Grading** — filters irrelevant documents before generation
- **Hallucination Detection** — validates answers are grounded in sources
- **Query Rewriting** — rewrites poor queries for better retrieval
- **Web Search Fallback** — Tavily search when vectorstore fails
- **Persistent VectorStore** — Chroma persisted to disk
- **Premium UI** — Dark-themed Streamlit chat with pipeline tracing
