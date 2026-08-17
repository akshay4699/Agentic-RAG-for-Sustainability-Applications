"""
LLM Chains — all prompt + LLM + structured output chains used by the graph.

Each chain is a factory function returning a cached chain instance:
  - Router chain: routes questions to vectorstore / web_search / direct_llm
  - Document grader: grades document relevance
  - Hallucination grader: checks if generation is grounded
  - Answer grader: checks if answer resolves the question
  - Query rewriter: rewrites queries for better retrieval
  - RAG chain: generates answers from context
"""

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.models import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    RouteQuery,
)
from backend.services.llm import get_llm


# ──────────────────────────────────────────────
# 1. ROUTER CHAIN
# ──────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at routing a user question to the correct data source.

You have access to:
- 'vectorstore': Contains documents about LLM agents, prompt engineering, and adversarial attacks on LLMs.
- 'web_search': Use for current events, recent news, or topics NOT in the vectorstore.
- 'direct_llm': Use for simple greetings, general knowledge, or questions that don't need retrieval.

Route the question to the most appropriate source.""",
    ),
    ("human", "{question}"),
])


@lru_cache()
def get_router_chain() -> Runnable:
    """Return the question router chain."""
    return _ROUTER_PROMPT | get_llm().with_structured_output(RouteQuery)


# ──────────────────────────────────────────────
# 2. DOCUMENT GRADER CHAIN
# ──────────────────────────────────────────────

_DOC_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant.""",
    ),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
])


@lru_cache()
def get_doc_grader() -> Runnable:
    """Return the document grader chain."""
    return _DOC_GRADER_PROMPT | get_llm().with_structured_output(GradeDocuments)


# ──────────────────────────────────────────────
# 3. HALLUCINATION GRADER CHAIN
# ──────────────────────────────────────────────

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing whether an LLM generation is grounded in / supported by \
a set of retrieved facts. Give a binary score 'yes' or 'no'.
'Yes' means the answer is grounded in the set of facts.""",
    ),
    ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
])


@lru_cache()
def get_hallucination_grader() -> Runnable:
    """Return the hallucination grader chain."""
    return _HALLUCINATION_PROMPT | get_llm().with_structured_output(GradeHallucinations)


# ──────────────────────────────────────────────
# 4. ANSWER GRADER CHAIN
# ──────────────────────────────────────────────

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question.""",
    ),
    ("human", "User question:\n\n{question}\n\nLLM generation: {generation}"),
])


@lru_cache()
def get_answer_grader() -> Runnable:
    """Return the answer grader chain."""
    return _ANSWER_PROMPT | get_llm().with_structured_output(GradeAnswer)


# ──────────────────────────────────────────────
# 5. QUERY REWRITER CHAIN
# ──────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a question re-writer that converts an input question to a better version \
that is optimized for vectorstore retrieval. Look at the input and try to reason about the \
underlying semantic intent / meaning.""",
    ),
    ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved question."),
])


@lru_cache()
def get_query_rewriter() -> Runnable:
    """Return the query rewriter chain."""
    return _REWRITE_PROMPT | get_llm() | StrOutputParser()


# ──────────────────────────────────────────────
# 6. RAG GENERATION CHAIN
# ──────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an assistant for question-answering tasks. Use the following pieces of \
retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.""",
    ),
    ("human", "Question: {question}\n\nContext: {context}"),
])


@lru_cache()
def get_rag_chain() -> Runnable:
    """Return the RAG generation chain."""
    return _RAG_PROMPT | get_llm() | StrOutputParser()
