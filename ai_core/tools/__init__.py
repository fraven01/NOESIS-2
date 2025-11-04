"""Tooling helpers and error types for AI Core."""

from .errors import InputError, ToolError
from .web_search import (
    ProviderSearchResult,
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

__all__ = [
    "ToolError",
    "InputError",
    "SearchAdapter",
    "SearchAdapterResponse",
    "ProviderSearchResult",
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
]
