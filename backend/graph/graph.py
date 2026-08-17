"""
Graph Builder — assembles the LangGraph StateGraph with all nodes and edges.

Architecture:
  START → route_question_decision
    ├── vectorstore  → retrieve → grade_documents → decide_to_generate
    │                                                 ├── generate → grade_generation
    │                                                 │               ├── useful      → END
    │                                                 │               ├── not_grounded → generate (loop)
    │                                                 │               └── not_useful   → transform_query → retrieve (loop)
    │                                                 ├── transform_query → retrieve (loop)
    │                                                 └── web_search → generate
    ├── web_search   → web_search → generate → grade_generation → ...
    └── direct_llm   → direct_generate → END
"""

import logging
from langgraph.graph import END, START, StateGraph

from backend.graph.edges import (
    decide_to_generate,
    grade_generation_decision,
    route_question_decision,
)
from backend.graph.nodes import (
    direct_generate,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from backend.state import GraphState

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None):
    """
    Build and compile the Agentic RAG StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (SqliteSaver / AsyncSqliteSaver)

    Returns:
        Compiled LangGraph application ready for multi-turn conversational invocation.
    """
    workflow = StateGraph(GraphState)

    # ----- Add Nodes -----
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("direct_generate", direct_generate)

    # ----- Add Edges -----

    # Entry point: Route the question
    workflow.add_conditional_edges(
        START,
        route_question_decision,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "direct_llm": "direct_generate",
        },
    )

    # Retrieve → Grade Documents
    workflow.add_edge("retrieve", "grade_documents")

    # Grade Documents → Decide what to do next
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    # Transform Query → Retrieve again (loop)
    workflow.add_edge("transform_query", "retrieve")

    # Web Search → Generate
    workflow.add_edge("web_search", "generate")

    # Generate → Grade the generation
    workflow.add_conditional_edges(
        "generate",
        grade_generation_decision,
        {
            "useful": END,                    # Answer is good → END
            "not_grounded": "generate",       # Hallucinating → regenerate
            "not_useful": "transform_query",  # Not useful → rewrite & retry
        },
    )

    # Direct Generate → END
    workflow.add_edge("direct_generate", END)

    # ----- Compile -----
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
