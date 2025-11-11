"""Tenant-aware web search worker implementation."""

from __future__ import annotations

import logging
from typing import Callable

from ai_core.tools.search_adapter_factory import SearchAdapterFactory
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    WebSearchContext,
    WebSearchResponse,
    WebSearchWorker,
)
from common.logging import get_logger

_LOGGER = get_logger(__name__)


class TenantAwareSearchWorker:
    """Execute web searches with tenant-specific search provider adapters.

    This worker uses a SearchAdapterFactory to dynamically select the
    appropriate search provider adapter based on the tenant_id in the
    search context. This enables different tenants to use different
    search providers (Google, Bing, Brave, etc.) with their own
    API credentials and configurations.

    The worker is designed to be used as a singleton, shared across
    all graphs that need web search functionality.
    """

    def __init__(
        self,
        factory: SearchAdapterFactory,
        *,
        max_results: int = 10,
        max_attempts: int = 3,
        backoff_factor: float = 0.6,
        oversample_factor: int = 2,
        sleep: Callable[[float], None] | None = None,
        timer: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the tenant-aware search worker.

        Args:
            factory: SearchAdapterFactory instance for creating tenant-specific adapters
            max_results: Maximum number of search results to return
            max_attempts: Maximum number of retry attempts for transient failures
            backoff_factor: Exponential backoff factor for retries
            oversample_factor: Factor to oversample results before deduplication
            sleep: Optional sleep function for testing
            timer: Optional timer function for testing
            logger: Optional logger instance
        """
        self._factory = factory
        self._max_results = max_results
        self._max_attempts = max_attempts
        self._backoff_factor = backoff_factor
        self._oversample_factor = oversample_factor
        self._sleep = sleep
        self._timer = timer
        self._logger = logger or _LOGGER
        self._worker_cache: dict[str, WebSearchWorker] = {}

    def run(
        self, *, query: str, context: WebSearchContext | dict[str, object]
    ) -> WebSearchResponse:
        """Execute a web search with tenant-specific adapter.

        This method extracts the tenant_id from the context, obtains
        the appropriate search adapter from the factory, and delegates
        to a WebSearchWorker configured with that adapter.

        Args:
            query: The search query string
            context: WebSearchContext or dict with tenant_id, trace_id, etc.

        Returns:
            WebSearchResponse with results and metadata
        """
        # Validate and extract tenant_id from context
        if isinstance(context, dict):
            tenant_id = context.get("tenant_id")
        else:
            tenant_id = context.tenant_id

        if not tenant_id:
            self._logger.error(
                "tenant_aware_search.missing_tenant_id",
                extra={"context": context},
            )
            raise ValueError("tenant_id is required in search context")

        tenant_id = str(tenant_id)

        # Get tenant-specific adapter from factory
        try:
            adapter = self._factory.get_adapter_for_tenant(tenant_id)
        except Exception as exc:
            self._logger.error(
                "tenant_aware_search.adapter_error",
                extra={"tenant_id": tenant_id, "error": str(exc)},
                exc_info=exc,
            )
            raise

        # Get or create a worker for this adapter
        worker = self._get_worker_for_adapter(adapter)

        # Execute the search
        self._logger.debug(
            "tenant_aware_search.executing",
            extra={
                "tenant_id": tenant_id,
                "query": query,
                "provider": adapter.provider_name,
            },
        )

        return worker.run(query=query, context=context)

    def _get_worker_for_adapter(self, adapter: BaseSearchAdapter) -> WebSearchWorker:
        """Get or create a WebSearchWorker for the given adapter.

        Workers are cached by adapter provider name to avoid creating
        multiple workers for the same adapter type.

        Args:
            adapter: The search adapter instance

        Returns:
            Configured WebSearchWorker instance
        """
        provider_name = adapter.provider_name
        if provider_name in self._worker_cache:
            # Check if the cached worker uses the same adapter
            cached_worker = self._worker_cache[provider_name]
            if cached_worker._adapter is adapter:
                return cached_worker

        # Create new worker
        worker = WebSearchWorker(
            adapter,
            max_results=self._max_results,
            max_attempts=self._max_attempts,
            backoff_factor=self._backoff_factor,
            oversample_factor=self._oversample_factor,
            sleep=self._sleep,
            timer=self._timer,
            logger=self._logger,
        )

        self._worker_cache[provider_name] = worker
        return worker

    def clear_cache(self) -> None:
        """Clear the internal worker cache.

        This should be called if adapter configurations change and
        workers need to be recreated.
        """
        self._worker_cache.clear()


__all__ = ["TenantAwareSearchWorker"]
