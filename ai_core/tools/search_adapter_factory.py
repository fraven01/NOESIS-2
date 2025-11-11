"""Factory for creating tenant-specific search adapters."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from django.conf import settings
from pydantic import SecretStr

from ai_core.tools.search_adapters.google import GoogleSearchAdapter
from ai_core.tools.web_search import BaseSearchAdapter
from common.logging import get_logger

_LOGGER = get_logger(__name__)


class SearchAdapterFactory:
    """Factory for creating tenant-aware search adapters.

    This factory retrieves tenant-specific search provider configurations
    from the database and creates the appropriate adapter instance with
    the tenant's custom settings.

    For development purposes, if no tenant configuration is found, it falls
    back to the default configuration from Django settings.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the factory.

        Args:
            logger: Optional logger instance for debugging
        """
        self._logger = logger or _LOGGER
        self._adapter_cache: dict[str, BaseSearchAdapter] = {}

    @lru_cache(maxsize=128)
    def get_adapter_for_tenant(self, tenant_id: str) -> BaseSearchAdapter:
        """Get the appropriate search adapter for a tenant.

        This method checks for tenant-specific configuration in the database.
        If found, it creates an adapter with those settings. Otherwise, it
        falls back to the default configuration from Django settings (primarily
        for the 'dev' tenant during development).

        Args:
            tenant_id: The UUID of the tenant

        Returns:
            A configured search adapter instance

        Raises:
            ValueError: If no configuration is found and no fallback is available
        """
        # Check cache first
        if tenant_id in self._adapter_cache:
            self._logger.debug(
                "search_adapter.cache_hit", extra={"tenant_id": tenant_id}
            )
            return self._adapter_cache[tenant_id]

        # Try to get tenant-specific configuration from database
        adapter = self._get_adapter_from_db(tenant_id)

        if adapter is None:
            # Fallback to default configuration (mainly for dev tenant)
            self._logger.info(
                "search_adapter.fallback_to_default",
                extra={"tenant_id": tenant_id},
            )
            adapter = self._get_default_adapter()

        # Cache the adapter
        self._adapter_cache[tenant_id] = adapter

        return adapter

    def _get_adapter_from_db(self, tenant_id: str) -> BaseSearchAdapter | None:
        """Retrieve tenant configuration from database and create adapter.

        Args:
            tenant_id: The UUID of the tenant

        Returns:
            Configured adapter or None if no configuration found
        """
        try:
            from ai_core.models import SearchProviderConfiguration

            # Get active configuration for tenant
            config = SearchProviderConfiguration.objects.filter(
                tenant_id=tenant_id, is_active=True
            ).first()

            if config is None:
                self._logger.debug(
                    "search_adapter.no_config_found",
                    extra={"tenant_id": tenant_id},
                )
                return None

            return self._create_adapter_from_config(config)

        except Exception as exc:
            self._logger.error(
                "search_adapter.db_error",
                extra={"tenant_id": tenant_id, "error": str(exc)},
                exc_info=exc,
            )
            return None

    def _create_adapter_from_config(
        self, config: Any
    ) -> BaseSearchAdapter:
        """Create an adapter instance from a configuration object.

        Args:
            config: SearchProviderConfiguration instance

        Returns:
            Configured adapter instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_type = config.provider_type
        provider_config = config.configuration

        if provider_type == "google_cse":
            return GoogleSearchAdapter(
                api_key=SecretStr(provider_config.get("api_key", "")),
                search_engine_id=provider_config.get("search_engine_id", ""),
            )
        elif provider_type == "bing_search":
            # Placeholder for Bing adapter (to be implemented)
            self._logger.warning(
                "search_adapter.unsupported_provider",
                extra={"provider_type": provider_type},
            )
            raise ValueError(f"Provider {provider_type} not yet implemented")
        elif provider_type == "brave_search":
            # Placeholder for Brave adapter (to be implemented)
            self._logger.warning(
                "search_adapter.unsupported_provider",
                extra={"provider_type": provider_type},
            )
            raise ValueError(f"Provider {provider_type} not yet implemented")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def _get_default_adapter(self) -> BaseSearchAdapter:
        """Get the default search adapter from Django settings.

        This is used as a fallback when no tenant-specific configuration
        is found, primarily for the 'dev' tenant during development.

        Returns:
            Default configured adapter

        Raises:
            ValueError: If required settings are not configured
        """
        try:
            api_key = getattr(settings, "GOOGLE_CUSTOM_SEARCH_API_KEY", None)
            search_engine_id = getattr(
                settings, "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", None
            )

            if not api_key or not search_engine_id:
                raise ValueError(
                    "Default search configuration missing: "
                    "GOOGLE_CUSTOM_SEARCH_API_KEY and/or "
                    "GOOGLE_CUSTOM_SEARCH_ENGINE_ID not set in settings"
                )

            return GoogleSearchAdapter(
                api_key=SecretStr(api_key),
                search_engine_id=search_engine_id,
            )
        except AttributeError as exc:
            raise ValueError(
                "Default search configuration missing from settings"
            ) from exc

    def clear_cache(self, tenant_id: str | None = None) -> None:
        """Clear the adapter cache.

        This is useful when tenant configurations are updated and the
        factory should reload them from the database.

        Args:
            tenant_id: If provided, clear only this tenant's cache.
                      If None, clear all cached adapters.
        """
        if tenant_id:
            self._adapter_cache.pop(tenant_id, None)
            # Also clear LRU cache for this specific tenant
            self.get_adapter_for_tenant.cache_clear()
        else:
            self._adapter_cache.clear()
            self.get_adapter_for_tenant.cache_clear()


__all__ = ["SearchAdapterFactory"]
