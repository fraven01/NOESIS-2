"""Tests for SearchAdapterFactory."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from django.conf import settings

from ai_core.tools.search_adapter_factory import SearchAdapterFactory
from ai_core.tools.search_adapters.google import GoogleSearchAdapter


@pytest.fixture
def factory() -> SearchAdapterFactory:
    """Create a SearchAdapterFactory instance."""
    return SearchAdapterFactory()


@pytest.fixture
def mock_config():
    """Create a mock SearchProviderConfiguration."""
    config = Mock()
    config.provider_type = "google_cse"
    config.configuration = {
        "api_key": "test-api-key",
        "search_engine_id": "test-engine-id",
    }
    return config


def test_get_adapter_for_tenant_with_db_config(
    factory: SearchAdapterFactory, mock_config: Mock
) -> None:
    """Test getting adapter when tenant has database configuration."""
    tenant_id = "test-tenant-123"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.return_value.first.return_value = mock_config

        adapter = factory.get_adapter_for_tenant(tenant_id)

        assert isinstance(adapter, GoogleSearchAdapter)
        assert adapter.provider_name == "google"
        MockModel.objects.filter.assert_called_once_with(
            tenant_id=tenant_id, is_active=True
        )


def test_get_adapter_for_tenant_fallback_to_default(
    factory: SearchAdapterFactory,
) -> None:
    """Test fallback to default adapter when no DB config exists."""
    tenant_id = "dev-tenant"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.return_value.first.return_value = None

        with patch.object(settings, "GOOGLE_CUSTOM_SEARCH_API_KEY", "default-key"):
            with patch.object(
                settings, "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", "default-engine"
            ):
                adapter = factory.get_adapter_for_tenant(tenant_id)

                assert isinstance(adapter, GoogleSearchAdapter)
                assert adapter.provider_name == "google"


def test_get_adapter_for_tenant_caches_result(
    factory: SearchAdapterFactory, mock_config: Mock
) -> None:
    """Test that adapters are cached per tenant."""
    tenant_id = "test-tenant-456"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.return_value.first.return_value = mock_config

        # First call
        adapter1 = factory.get_adapter_for_tenant(tenant_id)

        # Second call should return cached adapter
        adapter2 = factory.get_adapter_for_tenant(tenant_id)

        assert adapter1 is adapter2
        # Database should only be queried once
        assert MockModel.objects.filter.call_count == 1


def test_create_adapter_unsupported_provider(factory: SearchAdapterFactory) -> None:
    """Test error handling for unsupported provider types."""
    config = Mock()
    config.provider_type = "bing_search"
    config.configuration = {"api_key": "test-key"}

    with pytest.raises(ValueError, match="not yet implemented"):
        factory._create_adapter_from_config(config)


def test_get_default_adapter_missing_settings(factory: SearchAdapterFactory) -> None:
    """Test error when default settings are missing."""
    with patch.object(settings, "GOOGLE_CUSTOM_SEARCH_API_KEY", None):
        with pytest.raises(ValueError, match="Default search configuration missing"):
            factory._get_default_adapter()


def test_clear_cache_specific_tenant(
    factory: SearchAdapterFactory, mock_config: Mock
) -> None:
    """Test clearing cache for a specific tenant."""
    tenant_id = "test-tenant-789"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.return_value.first.return_value = mock_config

        # Get adapter (will be cached)
        adapter1 = factory.get_adapter_for_tenant(tenant_id)

        # Clear cache for this tenant
        factory.clear_cache(tenant_id)

        # Next call should create new adapter
        adapter2 = factory.get_adapter_for_tenant(tenant_id)

        # Should be different instances
        assert adapter1 is not adapter2


def test_clear_cache_all(factory: SearchAdapterFactory, mock_config: Mock) -> None:
    """Test clearing all cached adapters."""
    tenant_id1 = "tenant-1"
    tenant_id2 = "tenant-2"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.return_value.first.return_value = mock_config

        # Get adapters for two tenants
        factory.get_adapter_for_tenant(tenant_id1)
        factory.get_adapter_for_tenant(tenant_id2)

        assert len(factory._adapter_cache) == 2

        # Clear all cache
        factory.clear_cache()

        assert len(factory._adapter_cache) == 0


def test_get_adapter_db_error_returns_none(factory: SearchAdapterFactory) -> None:
    """Test that database errors are handled gracefully."""
    tenant_id = "error-tenant"

    with patch("ai_core.tools.search_adapter_factory.SearchProviderConfiguration") as MockModel:
        MockModel.objects.filter.side_effect = Exception("Database error")

        # Should fall back to default adapter
        with patch.object(settings, "GOOGLE_CUSTOM_SEARCH_API_KEY", "default-key"):
            with patch.object(
                settings, "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", "default-engine"
            ):
                adapter = factory.get_adapter_for_tenant(tenant_id)
                assert isinstance(adapter, GoogleSearchAdapter)
