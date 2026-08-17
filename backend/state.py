"""
Graph state definition — the data structure that flows through every LangGraph node.
"""

from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Represents the state of the Agentic RAG graph.

    Attributes:
        question: The user's original question
        generation: LLM generation (answer)
        documents: List of retrieved documents
        web_search_needed: Flag for web search fallback
        retry_count: Number of regeneration attempts (safety valve)
        messages: Multi-turn conversation messages with add_messages reducer
    """

    question: str
    generation: str
    documents: List[Document]
    web_search_needed: bool
    retry_count: int
    messages: Annotated[List[BaseMessage], add_messages]
