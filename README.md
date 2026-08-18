# Agentic RAG — Sustainability & Emission Factors App

A **production-grade** Agentic RAG (Retrieval-Augmented Generation) system built with **FastAPI** + **Streamlit** for querying Australian National Greenhouse Accounts Factors documents.

## Architecture

```
START → Route Question → [Vectorstore / Web Search / Direct LLM]
  → Retrieve Documents → Grade Documents → Decide Fallback
  → Generate Answer → Hallucination Check → Answer Quality → END
```

**Tech Stack:** LangGraph · Groq (LLM) · Pinecone (VectorStore) · NVIDIA Nemotron (Embeddings) · Tavily (Web Search) · FastAPI · Streamlit

## Project Structure

```
agentic_rag_app/
├── backend/
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                  # Settings & environment configuration
│   ├── models.py                  # Pydantic models & structured outputs
│   ├── state.py                   # GraphState definition
│   ├── ingest_pinecone.py         # Document loader & Pinecone indexer
│   ├── create_pinecone_index.py   # Pinecone index creation helper (2048 dims)
│   ├── services/
│   │   ├── llm.py                 # Groq LLM client
│   │   ├── vectorstore.py         # Pinecone vectorstore & retriever
│   │   └── search.py              # Tavily web search tool
│   ├── chains/
│   │   └── chains.py              # LangChain prompts & chains
│   └── graph/
│       ├── nodes.py                # LangGraph node functions
│       ├── edges.py                # LangGraph conditional edge logic
│       └── graph.py                # Graph assembly & compilation
├── data/                          # National Greenhouse Accounts Factors PDFs
├── frontend/
│   └── streamlit_app.py            # Streamlit chat interface
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

### 4. Create Pinecone Index (2048 Dimensions)

If you haven't created the Pinecone index with **2048 dimensions**, run:

```bash
cd agentic_rag_app
python -m backend.create_pinecone_index
```

### 5. Index PDF Documents into Pinecone

Run the Pinecone ingestion script to chunk and upload all PDFs from `agentic_rag_app/data`:

```bash
cd agentic_rag_app
python -m backend.ingest_pinecone
```

### 6. Start the Backend API

```bash
cd agentic_rag_app
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Start the Frontend UI

In a new terminal:

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
  -d '{"question": "What is the 2025 Scope 2 emission factor for electricity in New South Wales?"}'
```

## Key Features

- **Intelligent Routing** — Questions automatically route to vectorstore, web search, or direct LLM.
- **Document Retrieval** — Pinecone vectorstore powered by 2048-dimensional NVIDIA Nemotron embeddings.
- **PDF Ingestion Engine** — Parses and chunks multi-page Sustainability and Greenhouse Accounts PDFs.
- **Relevance & Fallback Grading** — Evaluates document relevance to avoid hallucinations and triggers query rewrites or web search when necessary.
- **Robust Flow Control** — Guardrails against infinite loops and handles API rate limits gracefully.
- **Interactive UI** — Streamlit chat interface with step-by-step pipeline execution tracing.

