"""
Streamlit Frontend — Premium chat UI for the Agentic RAG system with live streaming.

Communicates with the FastAPI backend via HTTP streaming.
"""

import json
import uuid
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

API_BASE = "http://localhost:8000"


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --text-primary: #e8e8ed;
        --text-secondary: #8b8b9e;
        --accent-purple: #7c3aed;
        --accent-blue: #3b82f6;
        --accent-teal: #14b8a6;
        --accent-gradient: linear-gradient(135deg, #7c3aed, #3b82f6, #14b8a6);
        --glow-purple: 0 0 20px rgba(124, 58, 237, 0.3);
        --glow-blue: 0 0 20px rgba(59, 130, 246, 0.2);
    }

    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 0.75rem !important;
        transition: all 0.3s ease !important;
    }

    .stChatMessage:hover {
        border-color: rgba(124, 58, 237, 0.2) !important;
        box-shadow: var(--glow-purple) !important;
    }

    /* ── Chat Input ── */
    .stChatInput {
        border-color: var(--border-subtle) !important;
    }

    .stChatInput > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.3s ease !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent-purple) !important;
        box-shadow: var(--glow-purple) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--glow-purple) !important;
    }

    /* ── Pipeline Step Cards ── */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        margin: 0.25rem 0;
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 10px;
        font-size: 0.82rem;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }

    .pipeline-step:hover {
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.3);
    }

    /* ── Route Badge ── */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
    }

    .route-vectorstore {
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.3);
        color: #a78bfa;
    }

    .route-web_search, .route-web-search {
        background: rgba(20, 184, 166, 0.12);
        border: 1px solid rgba(20, 184, 166, 0.3);
        color: #5eead4;
    }

    .route-direct_generate, .route-direct-llm {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93c5fd;
    }

    /* ── Sources ── */
    .source-link {
        display: block;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem 0;
        background: rgba(59, 130, 246, 0.06);
        border: 1px solid rgba(59, 130, 246, 0.12);
        border-radius: 8px;
        font-size: 0.78rem;
        color: #93c5fd;
        text-decoration: none;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .source-link:hover {
        background: rgba(59, 130, 246, 0.12);
        border-color: rgba(59, 130, 246, 0.25);
        color: #bfdbfe;
    }

    /* ── Status Indicator ── */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        animation: pulse-glow 2s ease-in-out infinite;
    }

    .status-online { background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.5); }
    .status-offline { background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ── Expander Styles ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* ── Divider ── */
    .subtle-divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 1rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

NODE_LABELS = {
    "retrieve": "Retrieve Documents",
    "grade_documents": "Grade Relevance",
    "generate": "Generate Answer",
    "transform_query": "Rewrite Query",
    "web_search": "Web Search",
    "direct_generate": "Direct LLM",
}


def get_route_badge(route: str) -> str:
    """Return an HTML badge for the route."""
    label = NODE_LABELS.get(route, route.replace("_", " ").title())
    css_class = f"route-{route.replace('_', '-')}"
    return f'<span class="route-badge {css_class}">{label}</span>'


def render_pipeline_steps(steps: list[str]) -> str:
    """Render pipeline steps as styled HTML."""
    html_parts = []
    for i, step in enumerate(steps):
        label = NODE_LABELS.get(step, step.replace("_", " ").title())
        arrow = " → " if i < len(steps) - 1 else ""
        html_parts.append(
            f'<div class="pipeline-step">'
            f'<span>{label}</span>'
            f'{arrow}'
            f'</div>'
        )
    return "".join(html_parts)


def check_backend_health() -> dict | None:
    """Check if the backend is healthy."""
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def query_backend_stream(question: str, thread_id: str = "default_thread"):
    """
    Generator that streams tokens from backend and updates session state with steps & thread_id.
    """
    try:
        response = requests.post(
            f"{API_BASE}/api/query/stream",
            json={"question": question, "thread_id": thread_id},
            stream=True,
            timeout=120,
        )
        if response.status_code != 200:
            st.error(f"Backend error: {response.status_code} — {response.text}")
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                try:
                    data = json.loads(line_str)
                except Exception:
                    continue

                event_type = data.get("type")

                if event_type == "step":
                    steps = data.get("steps", [])
                    route = steps[0] if steps else ""
                    st.session_state.last_response = {
                        "question": question,
                        "answer": "",
                        "sources": [],
                        "route": route,
                        "steps": steps,
                        "thread_id": thread_id,
                    }

                elif event_type == "token":
                    content = data.get("content", "")
                    yield content

                elif event_type == "done":
                    st.session_state.last_response = {
                        "question": question,
                        "answer": data.get("answer", ""),
                        "sources": data.get("sources", []),
                        "route": data.get("route", ""),
                        "steps": data.get("steps", []),
                        "thread_id": data.get("thread_id", thread_id),
                    }

                elif event_type == "error":
                    st.error(f"Error: {data.get('message')}")

    except requests.ConnectionError:
        st.error("Cannot connect to backend. Is the FastAPI server running?")
    except requests.Timeout:
        st.error("Request timed out. The query may be too complex.")


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("# Agentic RAG")
    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # Thread ID / Session control for SqliteSaver Memory
    thread_input = st.text_input(
        "Session ID (SqliteSaver)",
        value=st.session_state.thread_id,
        help="Thread ID used by LangGraph SqliteSaver checkpointer for conversation memory",
    )
    if thread_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_input

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("New Session", use_container_width=True):
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:6]}"
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()
    with col_btn2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # Backend status
    health = check_backend_health()
    if health:
        st.markdown(
            '<span class="status-dot status-online"></span> **Backend Online**',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            vs_status = "Ready" if health.get("vectorstore_ready") else "Pending"
            st.caption(f"VectorStore: {vs_status}")
        with col2:
            gr_status = "Ready" if health.get("graph_ready") else "Pending"
            st.caption(f"Graph: {gr_status}")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span> **Backend Offline**',
            unsafe_allow_html=True,
        )
        st.caption("Start the backend with:")
        st.code("uvicorn backend.main:app --reload", language="bash")

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # Pipeline trace
    st.markdown("### Pipeline Trace")

    if st.session_state.last_response:
        resp = st.session_state.last_response

        # Route badge
        route = resp.get("route", "")
        if route:
            st.markdown(get_route_badge(route), unsafe_allow_html=True)
            st.markdown("")

        # Steps
        steps = resp.get("steps", [])
        if steps:
            st.markdown(render_pipeline_steps(steps), unsafe_allow_html=True)

        # Sources
        sources = resp.get("sources", [])
        if sources:
            st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Sources")
            for src in sources:
                st.markdown(
                    f'<a href="{src}" target="_blank" class="source-link">{src}</a>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Ask a question to see the pipeline trace.")


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align: left; padding: 1rem 0 2rem;">
    <h1 style="
        background: linear-gradient(135deg, #7c3aed, #3b82f6, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        font-family: 'Inter', sans-serif;
    ">Agentic RAG Assistant</h1>
</div>
""", unsafe_allow_html=True)


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        answer = st.write_stream(query_backend_stream(prompt, thread_id=st.session_state.thread_id))

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
