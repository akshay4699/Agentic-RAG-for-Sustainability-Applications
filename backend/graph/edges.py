"""
Conditional Edge Functions — decision points in the graph.

These determine which node to transition to based on the current state.

Decision Points:
  - route_question_decision: Routes to vectorstore / web_search / direct_llm
  - decide_to_generate: After grading, decides generate / transform_query / web_search
  - grade_generation_decision: Checks hallucination & answer quality
"""

import logging

from backend.chains.chains import (
    get_answer_grader,
    get_hallucination_grader,
    get_router_chain,
)
from backend.config import get_settings
from backend.state import GraphState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# DECISION 1: ROUTE QUESTION
# ──────────────────────────────────────────────

def route_question_decision(state: GraphState) -> str:
    """Route question to vectorstore, web_search, or direct_llm."""
    logger.info("--- DECISION: Route Question ---")
    question = state["question"]

    router_chain = get_router_chain()
    route = router_chain.invoke({"question": question})

    logger.info(f"    Routing to: {route.datasource}")
    return route.datasource


# ──────────────────────────────────────────────
# DECISION 2: DECIDE TO GENERATE
# ──────────────────────────────────────────────

def decide_to_generate(state: GraphState) -> str:
    """After grading documents, decide: generate / transform_query / web_search."""
    logger.info("--- DECISION: Decide Fallback ---")
    documents = state["documents"]
    web_search_needed = state.get("web_search_needed", False)

    if not documents:
        # No relevant docs at all → web search
        logger.info("    -> No relevant docs found, falling back to WEB SEARCH")
        return "web_search"
    elif web_search_needed:
        # Some docs irrelevant → try rewriting the query
        logger.info("    -> Some irrelevant docs, TRANSFORMING QUERY")
        return "transform_query"
    else:
        # All docs relevant → proceed to generate
        logger.info("    -> All docs relevant, proceeding to GENERATE")
        return "generate"


# ──────────────────────────────────────────────
# DECISION 3: GRADE GENERATION
# ──────────────────────────────────────────────

def grade_generation_decision(state: GraphState) -> str:
    """Grade the generation for hallucinations and answer quality."""
    logger.info("--- DECISION: Grade Generation ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    settings = get_settings()

    # Safety valve: max retries to prevent infinite loops
    if retry_count >= settings.MAX_RETRY_COUNT:
        logger.warning("    Max retries reached, returning current answer")
        return "useful"

    # --- Check 1: Hallucination Check ---
    docs_text = "\n\n".join(doc.page_content for doc in documents)
    hallucination_grader = get_hallucination_grader()
    hallucination_score = hallucination_grader.invoke({
        "documents": docs_text,
        "generation": generation,
    })

    if hallucination_score.binary_score == "no":
        logger.info("    NOT GROUNDED — regenerating...")
        return "not_grounded"  # → loop back to generate

    logger.info("    Grounded in documents")

    # --- Check 2: Answer Quality Check ---
    answer_grader = get_answer_grader()
    answer_score = answer_grader.invoke({
        "question": question,
        "generation": generation,
    })

    if answer_score.binary_score == "yes":
        logger.info("    Answer is USEFUL — done!")
        return "useful"  # → END
    else:
        logger.info("    Answer NOT USEFUL — rewriting query...")
        return "not_useful"  # → transform_query → retrieve again
