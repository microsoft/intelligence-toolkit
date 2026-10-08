"""
Search provider abstraction for Schemify.

Decouples web search from the extraction pipeline, enabling
alternative search backends (Bing, Google, Brave, etc.).
"""

from typing import Protocol, runtime_checkable
import logging

from .models import Citation

logger = logging.getLogger("schemify.search")


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for web search providers."""

    async def search(
        self,
        query: str,
        user_location: dict[str, str] | None = None,
    ) -> tuple[str, list[Citation]]:
        """
        Execute a web search and return synthesized text with citations.

        Args:
            query: The search query
            user_location: Optional location for geo-aware results

        Returns:
            Tuple of (response_text, citations)
        """
        ...


class OpenAISearchProvider:
    """Search provider using OpenAI Responses API with web_search tool."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance for making API calls
        """
        self.llm = llm_client

    async def search(
        self,
        query: str,
        user_location: dict[str, str] | None = None,
    ) -> tuple[str, list[Citation]]:
        return await self.llm.web_search_completion(query, user_location=user_location)
