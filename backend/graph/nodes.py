"""
Graph Node Functions — each node reads from and writes to the GraphState.

Nodes:
  - retrieve: Fetches docs from vectorstore
  - grade_documents: Grades each doc for relevance
  - transform_query: Rewrites the query for better retrieval
  - web_search: Searches the web as a fallback
  - generate: Generates an answer using RAG chain
  - direct_generate: Direct LLM response (no retrieval)
"""

import logging

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.chains.chains import get_doc_grader, get_query_rewriter, get_rag_chain
from backend.services.llm import get_llm
from backend.services.search import get_web_search_tool
from backend.services.vectorstore import get_retriever
from backend.state import GraphState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# NODE 1: RETRIEVE
# ──────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from the vectorstore."""
    logger.info("--- NODE: Retrieve Documents ---")
    question = state["question"]

    retriever = get_retriever()
    documents = retriever.invoke(question)

    logger.info(f"    Retrieved {len(documents)} documents")
    return {
        "documents": documents,
        "question": question,
    }


# ──────────────────────────────────────────────
# NODE 2: GRADE DOCUMENTS
# ──────────────────────────────────────────────

def grade_documents(state: GraphState) -> GraphState:
    """Grade each retrieved document for relevance. Filter out irrelevant ones."""
    logger.info("--- NODE: Grade Documents ---")
    question = state["question"]
    documents = state.get("documents", [])

    doc_grader = get_doc_grader()
    filtered_docs = []

    for i, doc in enumerate(documents):
        try:
            score = doc_grader.invoke({
                "question": question,
                "document": doc.page_content,
            })
            if score.binary_score == "yes":
                logger.info(f"    Doc {i + 1}: RELEVANT")
                filtered_docs.append(doc)
            else:
                logger.info(f"    Doc {i + 1}: NOT RELEVANT")
        except Exception as e:
            logger.warning(f"    Doc {i + 1}: Error grading document ({e}), keeping document as fallback.")
            filtered_docs.append(doc)

    logger.info(f"    Kept {len(filtered_docs)}/{len(documents)} documents")
    return {
        "documents": filtered_docs,
        "question": question,
    }


# ──────────────────────────────────────────────
# NODE 3: TRANSFORM QUERY (Rewrite)
# ──────────────────────────────────────────────

def transform_query(state: GraphState) -> GraphState:
    """Re-write the query to produce a better question for retrieval."""
    logger.info("--- NODE: Transform Query ---")
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    try:
        query_rewriter = get_query_rewriter()
        better_question = query_rewriter.invoke({"question": question})
        logger.info(f"    Original:  {question}")
        logger.info(f"    Rewritten: {better_question}")
    except Exception as e:
        logger.warning(f"    Query rewriter failed: {e}. Keeping original question.")
        better_question = question

    return {
        "question": better_question,
        "documents": state.get("documents", []),
        "retry_count": retry_count + 1,
    }



# ──────────────────────────────────────────────
# NODE 4: WEB SEARCH
# ──────────────────────────────────────────────

def web_search(state: GraphState) -> GraphState:
    """Perform web search as a fallback data source."""
    logger.info("--- NODE: Web Search ---")
    question = state["question"]
    documents = state.get("documents", [])

    web_search_tool = get_web_search_tool()
    web_results = web_search_tool.invoke({"query": question})

    # Convert web results to Document objects
    web_docs = [
        Document(
            page_content=result["content"],
            metadata={"source": result["url"]},
        )
        for result in web_results
    ]

    # Append web results to existing documents
    documents = list(documents) + web_docs
    logger.info(f"    Found {len(web_docs)} web results, total docs: {len(documents)}")

    return {
        "documents": documents,
        "question": question,
    }


# ──────────────────────────────────────────────
# NODE 5: GENERATE ANSWER
# ──────────────────────────────────────────────

def generate(state: GraphState) -> GraphState:
    """Generate an answer using the retrieved documents as context."""
    logger.info("--- NODE: Generate Answer ---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    # Format documents into context string
    context = "\n\n".join(doc.page_content for doc in documents)

    rag_chain = get_rag_chain()
    generation = rag_chain.invoke({
        "question": question,
        "context": context,
    })

    logger.info(f"    Generated answer (attempt {retry_count + 1})")
    return {
        "generation": generation,
        "question": question,
        "documents": documents,
        "retry_count": retry_count + 1,
        "messages": [AIMessage(content=generation)],
    }


# ──────────────────────────────────────────────
# NODE 6: DIRECT GENERATE (No retrieval)
# ──────────────────────────────────────────────

def direct_generate(state: GraphState) -> GraphState:
    """Generate a direct LLM response without any retrieval."""
    logger.info("--- NODE: Direct Generate (No Retrieval) ---")
    question = state["question"]
    messages = state.get("messages", [])

    llm = get_llm()
    if messages and len(messages) > 1:
        generation = llm.invoke(messages).content
    else:
        generation = llm.invoke(question).content

    logger.info("    Direct answer generated")
    return {
        "generation": generation,
        "question": question,
        "documents": [],
        "messages": [AIMessage(content=generation)],
    }
