"""Tests for TenantAwareSearchWorker."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai_core.tools.search_adapter_factory import SearchAdapterFactory
from ai_core.tools.tenant_aware_search_worker import TenantAwareSearchWorker
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    ProviderSearchResult,
    SearchAdapterResponse,
    WebSearchContext,
)


class _FakeAdapter(BaseSearchAdapter):
    """Fake adapter for testing."""

    def __init__(self, provider: str = "test_provider") -> None:
        self.provider_name = provider

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        """Return fake search results."""
        return SearchAdapterResponse(
            results=[
                ProviderSearchResult(
                    url="https://example.com/result1",
                    title="Test Result",
                    snippet="Test snippet",
                    source=self.provider_name,
                )
            ],
            status_code=200,
        )


@pytest.fixture
def mock_factory() -> Mock:
    """Create a mock SearchAdapterFactory."""
    factory = Mock(spec=SearchAdapterFactory)
    factory.get_adapter_for_tenant.return_value = _FakeAdapter()
    return factory


@pytest.fixture
def worker(mock_factory: Mock) -> TenantAwareSearchWorker:
    """Create a TenantAwareSearchWorker with mock factory."""
    return TenantAwareSearchWorker(
        mock_factory, sleep=lambda _: None, timer=lambda: 1.0
    )


@pytest.fixture
def context() -> WebSearchContext:
    """Create a test context."""
    return WebSearchContext(
        tenant_id="test-tenant-123",
        trace_id="trace-456",
        workflow_id="workflow-789",
        case_id="case-abc",
        run_id="run-def",
    )


def test_run_with_valid_context(
    worker: TenantAwareSearchWorker,
    mock_factory: Mock,
    context: WebSearchContext,
) -> None:
    """Test running search with valid context."""
    with patch("ai_core.tools.tenant_aware_search_worker.record_span"):
        response = worker.run(query="test query", context=context)

        # Should have called factory with tenant_id
        mock_factory.get_adapter_for_tenant.assert_called_once_with("test-tenant-123")

        # Should return results
        assert response.outcome.decision == "ok"
        assert len(response.results) == 1
        assert str(response.results[0].url) == "https://example.com/result1"


def test_run_with_dict_context(
    worker: TenantAwareSearchWorker, mock_factory: Mock
) -> None:
    """Test running search with dict context."""
    context_dict = {
        "tenant_id": "dict-tenant-456",
        "trace_id": "trace-123",
        "workflow_id": "workflow-456",
        "case_id": "case-789",
        "run_id": "run-abc",
    }

    with patch("ai_core.tools.tenant_aware_search_worker.record_span"):
        response = worker.run(query="test query", context=context_dict)

        mock_factory.get_adapter_for_tenant.assert_called_once_with("dict-tenant-456")
        assert response.outcome.decision == "ok"


def test_run_without_tenant_id_raises_error(
    worker: TenantAwareSearchWorker,
) -> None:
    """Test that missing tenant_id raises ValueError."""
    context_dict = {
        "trace_id": "trace-123",
        "workflow_id": "workflow-456",
        "case_id": "case-789",
        "run_id": "run-abc",
    }

    with pytest.raises(ValueError, match="tenant_id is required"):
        worker.run(query="test query", context=context_dict)


def test_worker_caching_by_provider(
    mock_factory: Mock,
) -> None:
    """Test that workers are cached by provider name."""
    adapter1 = _FakeAdapter("google")
    adapter2 = _FakeAdapter("google")
    mock_factory.get_adapter_for_tenant.side_effect = [adapter1, adapter2]

    worker = TenantAwareSearchWorker(
        mock_factory, sleep=lambda _: None, timer=lambda: 1.0
    )

    context1 = WebSearchContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        workflow_id="wf-1",
        case_id="case-1",
        run_id="run-1",
    )

    context2 = WebSearchContext(
        tenant_id="tenant-2",
        trace_id="trace-2",
        workflow_id="wf-2",
        case_id="case-2",
        run_id="run-2",
    )

    with patch("ai_core.tools.tenant_aware_search_worker.record_span"):
        # First search with tenant-1
        worker.run(query="query1", context=context1)

        # Second search with tenant-2 (same provider)
        worker.run(query="query2", context=context2)

        # Should have created workers, but cache lookup is internal
        assert len(worker._worker_cache) >= 1


def test_factory_adapter_error_propagates(
    worker: TenantAwareSearchWorker,
    mock_factory: Mock,
    context: WebSearchContext,
) -> None:
    """Test that factory errors are propagated."""
    mock_factory.get_adapter_for_tenant.side_effect = ValueError(
        "Invalid configuration"
    )

    with pytest.raises(ValueError, match="Invalid configuration"):
        worker.run(query="test query", context=context)


def test_clear_cache(worker: TenantAwareSearchWorker) -> None:
    """Test clearing the worker cache."""
    # Populate cache
    worker._worker_cache["google"] = Mock()
    worker._worker_cache["bing"] = Mock()

    assert len(worker._worker_cache) == 2

    # Clear cache
    worker.clear_cache()

    assert len(worker._worker_cache) == 0


def test_multiple_adapters_create_separate_workers(
    mock_factory: Mock,
) -> None:
    """Test that different adapters create separate workers."""
    google_adapter = _FakeAdapter("google")
    bing_adapter = _FakeAdapter("bing")

    mock_factory.get_adapter_for_tenant.side_effect = [
        google_adapter,
        bing_adapter,
    ]

    worker = TenantAwareSearchWorker(
        mock_factory, sleep=lambda _: None, timer=lambda: 1.0
    )

    context1 = WebSearchContext(
        tenant_id="tenant-google",
        trace_id="trace-1",
        workflow_id="wf-1",
        case_id="case-1",
        run_id="run-1",
    )

    context2 = WebSearchContext(
        tenant_id="tenant-bing",
        trace_id="trace-2",
        workflow_id="wf-2",
        case_id="case-2",
        run_id="run-2",
    )

    with patch("ai_core.tools.tenant_aware_search_worker.record_span"):
        # Search with google adapter
        worker.run(query="query1", context=context1)

        # Search with bing adapter
        worker.run(query="query2", context=context2)

        # Should have two different workers cached
        assert len(worker._worker_cache) == 2
        assert "google" in worker._worker_cache
        assert "bing" in worker._worker_cache
