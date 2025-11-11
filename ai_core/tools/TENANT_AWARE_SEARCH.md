# Tenant-Aware Search Architecture

## Overview

The tenant-aware search system enables each tenant in NOESIS 2 to configure their own web search provider with custom credentials and settings. This architecture provides flexibility, security, and scalability for multi-tenant search operations.

## Architecture Components

### 1. SearchProviderConfiguration Model

**Location**: `ai_core/models.py`

A Django model that stores tenant-specific search provider configurations in the database.

**Fields**:
- `tenant`: ForeignKey to `customers.Tenant`
- `provider_type`: Choice field (google_cse, bing_search, brave_search)
- `configuration`: JSONField for provider-specific settings (e.g., api_key, search_engine_id)
- `is_active`: Boolean to enable/disable configurations

**Features**:
- One active configuration per tenant per provider type
- Indexed by tenant_id and is_active for fast lookups
- Flexible JSONField allows different providers to have different configuration schemas

### 2. SearchAdapterFactory

**Location**: `ai_core/tools/search_adapter_factory.py`

A service class that creates tenant-specific search adapters dynamically.

**Key Methods**:
- `get_adapter_for_tenant(tenant_id)`: Returns the appropriate adapter for a tenant
- `clear_cache(tenant_id)`: Clears cached adapters when configurations change

**Behavior**:
1. Checks database for tenant-specific configuration
2. If found, creates adapter with tenant's custom settings
3. If not found, falls back to default settings from Django settings (primarily for dev tenant)
4. Caches adapters using `@lru_cache` to minimize database queries

**Supported Providers**:
- Google Custom Search (`google_cse`)
- Bing Search (`bing_search`) - placeholder for future implementation
- Brave Search (`brave_search`) - placeholder for future implementation

### 3. TenantAwareSearchWorker

**Location**: `ai_core/tools/tenant_aware_search_worker.py`

A worker that orchestrates web searches with tenant-specific adapters.

**Key Features**:
- Extracts `tenant_id` from the search context
- Uses `SearchAdapterFactory` to get the correct adapter
- Delegates to `WebSearchWorker` for actual search execution
- Caches workers by provider name for performance

**Flow**:
```
Graph → TenantAwareSearchWorker
         ↓
         Extract tenant_id from context
         ↓
         SearchAdapterFactory.get_adapter_for_tenant()
         ↓
         WebSearchWorker.run() with tenant-specific adapter
         ↓
         Return results
```

### 4. Updated Shared Workers

**Location**: `ai_core/tools/shared_workers.py`

The `get_web_search_worker()` function now returns a `TenantAwareSearchWorker` instance instead of a basic `WebSearchWorker`.

**Impact**: All graphs using `get_web_search_worker()` automatically benefit from tenant-aware search without code changes.

## Usage

### For Graph Developers

No changes required! The existing API remains unchanged:

```python
from ai_core.tools.shared_workers import get_web_search_worker

# In your graph
worker = get_web_search_worker()
response = worker.run(
    query="search query",
    context={
        "tenant_id": "uuid-here",  # Required!
        "trace_id": "trace-123",
        "workflow_id": "workflow-456",
        "case_id": "case-789",
        "run_id": "run-abc",
    }
)
```

**Important**: The `tenant_id` field in the context is now **mandatory**. The worker will raise a `ValueError` if it's missing.

### For Administrators

#### Adding a Tenant Configuration

Use the Django admin interface or shell:

```python
from customers.models import Tenant
from ai_core.models import SearchProviderConfiguration

tenant = Tenant.objects.get(schema_name="demo")

config = SearchProviderConfiguration.objects.create(
    tenant=tenant,
    provider_type="google_cse",
    configuration={
        "api_key": "your-api-key-here",
        "search_engine_id": "your-engine-id-here"
    },
    is_active=True
)
```

#### Updating Configuration

```python
config = SearchProviderConfiguration.objects.get(tenant=tenant, is_active=True)
config.configuration["api_key"] = "new-api-key"
config.save()

# Clear the factory cache to reload configuration
from ai_core.tools.shared_workers import get_web_search_worker
worker = get_web_search_worker()
worker._factory.clear_cache(str(tenant.id))
```

#### Switching Providers

```python
# Deactivate old configuration
old_config.is_active = False
old_config.save()

# Create new configuration
new_config = SearchProviderConfiguration.objects.create(
    tenant=tenant,
    provider_type="bing_search",  # Different provider
    configuration={
        "api_key": "bing-api-key",
        # Bing-specific settings...
    },
    is_active=True
)

# Clear cache
worker._factory.clear_cache(str(tenant.id))
```

## Fallback Behavior

If a tenant has no configuration in the database, the system falls back to the default Google Custom Search configuration from Django settings:

- `settings.GOOGLE_CUSTOM_SEARCH_API_KEY`
- `settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID`

This is primarily intended for the `dev` tenant during local development.

## Security Considerations

1. **Credentials Isolation**: Each tenant's API keys are stored separately and never shared
2. **Database Storage**: Sensitive credentials are stored in the `configuration` JSONField
3. **Future Enhancement**: Consider encrypting the `configuration` field for additional security
4. **Access Control**: Only tenant admins should be able to configure search providers

## Performance

1. **Adapter Caching**: The `SearchAdapterFactory` uses `@lru_cache` to cache adapters
2. **Worker Caching**: The `TenantAwareSearchWorker` caches `WebSearchWorker` instances by provider
3. **Database Queries**: Minimized through intelligent caching strategies

## Testing

### Unit Tests

- `ai_core/tests/tools/test_search_adapter_factory.py`: Tests for factory logic
- `ai_core/tests/tools/test_tenant_aware_search_worker.py`: Tests for worker orchestration

### Integration Tests

To test with a real tenant:

```python
from django.test import TestCase
from customers.models import Tenant
from ai_core.models import SearchProviderConfiguration
from ai_core.tools.shared_workers import get_web_search_worker

class TenantSearchIntegrationTest(TestCase):
    def test_tenant_specific_search(self):
        tenant = Tenant.objects.create(schema_name="test")
        config = SearchProviderConfiguration.objects.create(
            tenant=tenant,
            provider_type="google_cse",
            configuration={"api_key": "test", "search_engine_id": "test"},
            is_active=True,
        )

        worker = get_web_search_worker()
        response = worker.run(
            query="test",
            context={
                "tenant_id": str(tenant.id),
                "trace_id": "trace",
                "workflow_id": "wf",
                "case_id": "case",
                "run_id": "run",
            }
        )

        assert response.outcome.decision == "ok"
```

## Migration Path

The system is backward compatible. Existing graphs will continue to work because:

1. The `get_web_search_worker()` API is unchanged
2. Tenants without configuration automatically fall back to default settings
3. The `tenant_id` field was already part of the context in most graphs

## Future Enhancements

1. **Additional Providers**: Implement Bing and Brave search adapters
2. **Admin UI**: Create a dedicated admin interface for managing search configurations
3. **Configuration Validation**: Add validation for provider-specific configuration schemas
4. **Encryption**: Encrypt sensitive fields in the database
5. **Usage Tracking**: Track search usage per tenant for billing/monitoring
6. **Rate Limiting**: Implement per-tenant rate limits

## Troubleshooting

### Error: "tenant_id is required in search context"

**Cause**: The context passed to the worker is missing the `tenant_id` field.

**Solution**: Ensure your graph passes a complete context:

```python
context = {
    "tenant_id": "your-tenant-uuid",  # Required!
    "trace_id": "...",
    "workflow_id": "...",
    "case_id": "...",
    "run_id": "...",
}
```

### Error: "Default search configuration missing"

**Cause**: No tenant configuration exists and default settings are not configured.

**Solution**: Set the following in your Django settings:

```python
GOOGLE_CUSTOM_SEARCH_API_KEY = "your-key"
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = "your-engine-id"
```

### Search returns no results for a tenant

**Possible Causes**:
1. Invalid API credentials in the tenant configuration
2. Quota exceeded for the search provider
3. Search provider service is down

**Debugging**:
1. Check the `outcome.meta` in the response for error details
2. Review Langfuse traces for the search operation
3. Verify credentials in the Django admin
4. Check provider quota/billing status

## Related Documentation

- [ai_core/tools/web_search.py](web_search.py): Core search worker implementation
- [ai_core/tools/search_adapters/google.py](search_adapters/google.py): Google CSE adapter
- [docs/agents/tool-contracts.md](../../docs/agents/tool-contracts.md): Tool contract specifications
- [docs/multi-tenancy.md](../../docs/multi-tenancy.md): Multi-tenancy architecture

## Changelog

### Version 1.0 (2025-11-11)

- Initial implementation of tenant-aware search architecture
- Added `SearchProviderConfiguration` model
- Implemented `SearchAdapterFactory` service
- Created `TenantAwareSearchWorker` orchestrator
- Updated `get_web_search_worker()` to use new architecture
- Added comprehensive unit tests
- Documented architecture and usage
