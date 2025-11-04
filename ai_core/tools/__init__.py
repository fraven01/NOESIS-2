"""Tooling helpers and error types for AI Core."""

from .errors import InputError, ToolError
from .web_search import (
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
    WebSearchContext,
    WebSearchInput,
    WebSearchResponse,
    WebSearchWorker,
)
from .search_adapters import GoogleSearchAdapter

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
    "WebSearchContext",
    "WebSearchInput",
    "WebSearchResponse",
    "WebSearchWorker",
    "GoogleSearchAdapter",
]
