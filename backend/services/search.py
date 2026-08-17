"""
Web Search service — configures and exposes the Tavily search tool.
"""

from functools import lru_cache

from langchain_community.tools.tavily_search import TavilySearchResults

from backend.config import get_settings


@lru_cache()
def get_web_search_tool() -> TavilySearchResults:
    """Return a cached TavilySearchResults tool instance."""
    settings = get_settings()
    return TavilySearchResults(max_results=settings.WEB_SEARCH_MAX_RESULTS)
