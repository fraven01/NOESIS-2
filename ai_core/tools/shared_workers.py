"""Shared worker instances to avoid duplication across graphs."""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings
from pydantic import SecretStr

from ai_core.tools.search_adapters.google import GoogleSearchAdapter
from ai_core.tools.web_search import WebSearchWorker


@lru_cache(maxsize=1)
def get_web_search_worker() -> WebSearchWorker:
    """Return a shared WebSearchWorker instance.

    This worker is thread-safe and can be reused across all graphs that need
    web search functionality. The GoogleSearchAdapter credentials are loaded
    from Django settings.

    Returns:
        Configured WebSearchWorker with GoogleSearchAdapter.
    """
    search_adapter = GoogleSearchAdapter(
        api_key=SecretStr(settings.GOOGLE_CUSTOM_SEARCH_API_KEY),
        search_engine_id=settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
    )
    return WebSearchWorker(search_adapter)


__all__ = ["get_web_search_worker"]
