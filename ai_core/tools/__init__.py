"""Tooling helpers and error types for AI Core."""

from .errors import InputError, ToolError

__all__ = [
    "ToolError",
    "InputError",
    "BaseSearchAdapter",
    "SearchAdapter",
    "SearchAdapterResponse",
    "ProviderSearchResult",
    "RawSearchResult",
    "SearchProviderError",
    "SearchProviderTimeout",
    "SearchProviderQuotaExceeded",
    "SearchProviderBadResponse",
    "SearchResult",
    "ToolOutcome",
    "WebSearchInput",
    "WebSearchResponse",
    "WebSearchWorker",
    "GoogleSearchAdapter",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies with tool_contracts.

    web_search.py imports ToolContext, which is defined in tool_contracts/base.py.
    tool_contracts/base.py imports ToolErrorType from tools/errors.py.
    If we import web_search at module level, it creates a circular dependency.

    Solution: Delay web_search imports until actually needed via __getattr__.
    """
    if name in {
        "BaseSearchAdapter",
        "ProviderSearchResult",
        "RawSearchResult",
        "SearchAdapter",
        "SearchAdapterResponse",
        "SearchProviderBadResponse",
        "SearchProviderError",
        "SearchProviderQuotaExceeded",
        "SearchProviderTimeout",
        "SearchResult",
        "ToolOutcome",
        "WebSearchInput",
        "WebSearchResponse",
        "WebSearchWorker",
    }:
        from .web_search import (  # noqa: F401
            BaseSearchAdapter,
            ProviderSearchResult,
            RawSearchResult,
            SearchAdapter,
            SearchAdapterResponse,
            SearchProviderBadResponse,
            SearchProviderError,
            SearchProviderQuotaExceeded,
            SearchProviderTimeout,
            SearchResult,
            ToolOutcome,
            WebSearchInput,
            WebSearchResponse,
            WebSearchWorker,
        )

        return locals()[name]

    if name == "GoogleSearchAdapter":
        from .search_adapters import GoogleSearchAdapter

        return GoogleSearchAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
