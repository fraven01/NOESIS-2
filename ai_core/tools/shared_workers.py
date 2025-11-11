"""Shared worker instances to avoid duplication across graphs."""

from __future__ import annotations

from functools import lru_cache

from ai_core.tools.search_adapter_factory import SearchAdapterFactory
from ai_core.tools.tenant_aware_search_worker import TenantAwareSearchWorker


@lru_cache(maxsize=1)
def get_web_search_worker() -> TenantAwareSearchWorker:
    """Return a shared TenantAwareSearchWorker instance.

    This worker is thread-safe and can be reused across all graphs that need
    web search functionality. It uses a SearchAdapterFactory to dynamically
    select the appropriate search provider adapter based on tenant configuration.

    For tenants without specific configuration, it falls back to the default
    Google Custom Search adapter configured in Django settings.

    Returns:
        Configured TenantAwareSearchWorker with SearchAdapterFactory.
    """
    factory = SearchAdapterFactory()
    return TenantAwareSearchWorker(factory)


__all__ = ["get_web_search_worker"]
