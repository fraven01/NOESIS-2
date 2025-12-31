import json
from unittest.mock import patch

import pytest
from django.core.cache import cache
from django.test import RequestFactory
from django.urls import reverse
from django_tenants.utils import get_public_schema_name, schema_context

from ai_core.rag.collections import manual_collection_uuid
from customers.tests.factories import TenantFactory
from theme.views import (
    rag_tools,
    start_rerank_workflow,
    web_search,
    web_search_ingest_selected,
)


@pytest.mark.django_db
def test_rag_tools_page_is_accessible():
    tenant_schema = "workbench"
    tenant = TenantFactory(schema_name=tenant_schema)
    # tenant_id defaults to schema_name if no explicit tenant_id attribute exists
    tenant_id = getattr(tenant, "tenant_id", None) or tenant.schema_name

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = tenant
    request.tenant_schema = tenant_schema
    # RequestFactory doesn't create sessions, so we need to add one manually
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    response = rag_tools(request)

    assert response.status_code == 200
    content = response.content.decode()

    assert "RAG Developer Workbench" in content
    assert "hx-post" in content
    assert "hx-target" in content
    # Verify HTMX headers are set on the body/container (JSON format)
    assert 'hx-headers=\'{"X-Tenant-ID": "' + tenant_id + '"' in content
    assert (
        'hx-headers=\'{"X-Tenant-ID": "'
        + tenant_id
        + '", "X-Tenant-Schema": "'
        + tenant_schema
        + '"'
        in content
    )


@pytest.mark.django_db
def test_rag_tools_requires_tenant():
    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))

    with schema_context(get_public_schema_name()):
        response = rag_tools(request)

    assert response.status_code == 403
    assert (
        json.loads(response.content)["detail"]
        == "Tenant could not be resolved from request"
    )


@pytest.mark.django_db
def test_rag_tools_rejects_spoofed_headers():
    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"), HTTP_X_TENANT_ID="spoofed")

    with schema_context(get_public_schema_name()):
        response = rag_tools(request)

    assert response.status_code == 403
    assert (
        json.loads(response.content)["detail"]
        == "Tenant could not be resolved from request"
    )


@pytest.mark.django_db
@patch("ai_core.graphs.web_acquisition_graph.build_web_acquisition_graph")
def test_web_search_uses_external_knowledge_graph(mock_build_graph):
    """Test that web_search view uses WebAcquisitionGraph for search."""
    tenant_schema = "test"

    # Create a mock graph object that the factory returns
    from unittest.mock import MagicMock

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "output": {
            "decision": "acquired",
            "search_results": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "snippet": "Test snippet",
                }
            ],
            "selected_result": None,
            "ingestion_result": None,
            "error": None,
            "auto_ingest": False,
        }
    }
    # Factory function returns the mock graph
    mock_build_graph.return_value = mock_graph

    tenant = TenantFactory(schema_name=tenant_schema)

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "test query", "collection_id": "test-collection"}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["outcome"] == "completed"
    assert "results" in response_data
    assert "trace_id" in response_data

    # Verify that graph.invoke was called with correct parameters
    mock_graph.invoke.assert_called_once()
    call_args = mock_graph.invoke.call_args
    state = call_args[0][0]
    # Web Acquisition Graph Input Structure
    assert state["input"]["query"] == "test query"
    assert state["input"]["mode"] == "search_only"
    assert "collection_id" not in state["input"]  # collection_id is in Context now


@pytest.mark.django_db
@patch("ai_core.graphs.web_acquisition_graph.build_web_acquisition_graph")
def test_web_search_htmx_returns_partial(mock_build_graph):
    """Test that web_search returns HTML partial for HTMX requests."""
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    from unittest.mock import MagicMock

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "output": {
            "decision": "acquired",
            "search_results": [
                {
                    "url": "https://example.com",
                    "title": "HTMX Result",
                    "snippet": "Snippet",
                }
            ],
            "selected_result": None,
            "ingestion_result": None,
            "error": None,
            "auto_ingest": False,
        }
    }
    mock_build_graph.return_value = mock_graph

    factory = RequestFactory()
    # Simulate HTMX request with form-encoded data (default for hx-post)
    request = factory.post(
        reverse("web-search"),
        data={"query": "htmx query", "collection_id": "test-collection"},
    )
    request.headers = {"HX-Request": "true"}
    request.tenant = tenant

    response = web_search(request)

    assert response.status_code == 200
    content = response.content.decode()
    # Should return the partial, not JSON
    assert "HTMX Result" in content
    assert "class=" in content  # Basic HTML check
    assert "Snippet" in content


@pytest.mark.django_db
@patch("ai_core.graphs.web_acquisition_graph.build_web_acquisition_graph")
def test_web_search_defaults_to_manual_collection(mock_build_graph):
    tenant_schema = "fallback"
    tenant = TenantFactory(schema_name=tenant_schema)
    tenant_id = tenant.schema_name

    from unittest.mock import MagicMock

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "output": {
            "decision": "no_results",
            "search_results": [],
            "selected_result": None,
            "ingestion_result": None,
            "error": None,
            "auto_ingest": False,
        }
    }
    mock_build_graph.return_value = mock_graph

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "fallback query"}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search(request)

    assert response.status_code == 200
    mock_graph.invoke.assert_called_once()
    manual_id = str(manual_collection_uuid(tenant_id))
    call_args = mock_graph.invoke.call_args
    state = call_args[0][0]
    # Expect "default" as per view logic which preserves user intent or falls back to default logic
    # But view says: collection_id = resolved_collection_id or manual_collection_id or "default"
    # Wait, if I send blank/none, it gets manual_collection_id.
    # The view code I saw earlier:
    # manual_collection_id, resolved_collection_id = _resolve_manual_collection(tenant_id, data.get("collection_id"))
    # collection_id = resolved_collection_id or manual_collection_id or "default"
    # If data.get("collection_id") is None, resolved is None. manual_collection_id should be returned by _resolve_manual_collection if tenant exists.
    # So expectation should be manual_id.
    # So expectation should be manual_id.
    # Expect manual_collection_id via BusinessContext
    # We can check tool_context.business.collection_id
    tool_context = state["tool_context"]
    assert tool_context.business.collection_id == manual_id


@pytest.mark.django_db
@patch("documents.collection_service.CollectionService.ensure_manual_collection")
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected_defaults_to_manual_collection(
    mock_dispatch, mock_ensure
):
    tenant_schema = "auto"
    tenant = TenantFactory(schema_name=tenant_schema)
    tenant_id = tenant.schema_name

    # Must be a valid UUID for CrawlerRunRequest validation
    manual_uuid = "11111111-1111-1111-1111-111111111111"
    mock_ensure.return_value = manual_uuid
    mock_dispatch.return_value = {"count": 1, "tasks": []}

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps({"urls": ["https://fallback.example"]}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    assert mock_dispatch.call_count == 1
    call_args = mock_dispatch.call_args
    request_model = call_args.args[0]
    assert request_model.collection_id == manual_uuid
    mock_ensure.assert_called_once_with(tenant_id)


@pytest.mark.django_db
@patch("theme.views.llm_routing.resolve")
@patch("theme.views.submit_worker_task")
@patch("ai_core.graphs.web_acquisition_graph.build_web_acquisition_graph")
def test_web_search_rerank_applies_scores(
    mock_build_graph, mock_submit_task, mock_resolve, settings
):
    settings.RERANK_MODEL_PRESET = "meta/llama-3.1-70b-instruct"

    def fake_resolve(label: str):
        if label == "meta/llama-3.1-70b-instruct":
            raise ValueError("unknown label")
        return "resolved-model"

    mock_resolve.side_effect = fake_resolve

    cache.clear()
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    from unittest.mock import MagicMock

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "output": {
            "decision": "acquired",
            "search_results": [
                {
                    "document_id": "doc-a",
                    "title": "Alpha",
                    "snippet": "Snippet A",
                    "source": "crawler",
                    "url": "https://a.example",
                    "score": 0.3,
                },
                {
                    "document_id": "doc-b",
                    "title": "Beta",
                    "snippet": "Snippet B",
                    "source": "crawler",
                    "url": "https://b.example",
                    "score": 0.2,
                },
            ],
            "selected_result": None,
            "ingestion_result": None,
            "error": None,
            "auto_ingest": False,
        }
    }
    mock_build_graph.return_value = mock_graph
    mock_submit_task.return_value = (
        {
            "task_id": "task-1",
            "result": {
                "ranked": [
                    {"id": "doc-b", "score": 88, "reasons": ["präzise"]},
                    {"id": "doc-a", "score": 40, "reasons": []},
                ],
                "latency_s": 0.5,
                "model": "demo",
            },
        },
        True,
    )

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "test", "rerank": True}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "succeeded"
    assert data["results"][0]["title"] == "Beta"
    assert data["results"][0]["rerank"]["score"] == 88
    task_payload = mock_submit_task.call_args.kwargs["task_payload"]
    assert task_payload["control"]["model_preset"] == "fast"


@pytest.mark.django_db
@patch("theme.views.submit_worker_task", return_value=({"task_id": "task-q"}, False))
@patch("ai_core.graphs.web_acquisition_graph.build_web_acquisition_graph")
def test_web_search_rerank_returns_queue_status(mock_build_graph, _mock_submit_task):
    cache.clear()
    tenant_schema = "queued"
    tenant = TenantFactory(schema_name=tenant_schema)

    from unittest.mock import MagicMock

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "output": {
            "decision": "acquired",
            "search_results": [
                {
                    "document_id": "doc-a",
                    "title": "Alpha",
                    "snippet": "Snippet A",
                    "url": "https://a.example",
                }
            ],
            "selected_result": None,
            "ingestion_result": None,
            "error": None,
            "auto_ingest": False,
        }
    }
    mock_build_graph.return_value = mock_graph

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "test", "rerank": True}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "queued"
    assert data["rerank"]["status_url"].endswith("/api/llm/tasks/task-q/")


@pytest.mark.django_db
@patch(
    "documents.collection_service.CollectionService.ensure_manual_collection",
    return_value="22222222-2222-2222-2222-222222222222",
)
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected(mock_dispatch, _mock_ensure):
    """Test that web_search_ingest_selected dispatches via CrawlerManager."""
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    mock_dispatch.return_value = {
        "count": 2,
        "tasks": [
            {"task_id": "task-1", "origin": {"url": "https://example.com"}},
            {"task_id": "task-2", "origin": {"url": "https://test.com"}},
        ],
    }

    factory = RequestFactory()
    # Use valid UUID for collection_id
    test_collection_uuid = "33333333-3333-3333-3333-333333333333"
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps(
            {
                "urls": ["https://example.com", "https://test.com"],
                "collection_id": test_collection_uuid,
            }
        ),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert mock_dispatch.call_count == 1


@pytest.mark.django_db
@patch(
    "documents.collection_service.CollectionService.ensure_manual_collection",
    return_value="44444444-4444-4444-4444-444444444444",
)
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected_passes_correct_params_to_crawler(
    mock_dispatch,
    _mock_ensure,
):
    """Ensure CrawlerManager receives the expected request model.

    This test validates that:
    1. collection_id from the request payload is passed through to CrawlerManager
    2. workflow_id is set to "web-search-ingestion"
    3. origins list is properly constructed from URLs
    """
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    mock_dispatch.return_value = {"count": 1, "tasks": []}

    test_collection_uuid = "55555555-5555-5555-5555-555555555555"

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps(
            {
                "urls": ["https://example.com"],
                "collection_id": test_collection_uuid,
                "case_id": "TEST-CASE",
            }
        ),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert mock_dispatch.call_count == 1

    # Verify the request model passed to CrawlerManager
    call_args = mock_dispatch.call_args
    request_model = call_args.args[0]
    assert request_model.workflow_id == "web-search-ingestion"
    assert request_model.mode == "live"
    assert request_model.collection_id == test_collection_uuid
    assert len(request_model.origins) == 1
    assert request_model.origins[0].url == "https://example.com"

    # Verify the meta payload complies with UniversalIngestionGraph contract
    meta = call_args.args[1]
    assert "scope_context" in meta, "CrawlerManager expects 'scope_context' in meta"
    assert meta["scope_context"]["tenant_id"] == tenant.schema_name


@pytest.mark.django_db
@patch("theme.views.submit_worker_task")
def test_start_rerank_workflow_returns_completed(mock_submit_worker_task):
    tenant_schema = "test"
    telemetry_payload = {
        "graph": "collection_search",
        "nodes": {"k_generate_strategy": {"status": "completed"}},
    }
    search_payload = {
        "results": [{"title": "Alpha", "url": "https://example.com", "score": 0.3}],
        "responses": [],
    }
    mock_submit_worker_task.return_value = (
        {
            "task_id": "task-123",
            "result": {
                "outcome": "search_completed",
                "telemetry": telemetry_payload,
                "search": search_payload,
            },
            "cost_summary": {"total_usd": 0.001},
        },
        True,
    )

    tenant = TenantFactory(schema_name=tenant_schema)

    factory = RequestFactory()
    request = factory.post(
        reverse("rag_tools_start_rerank"),
        data=json.dumps(
            {
                "query": "Alpha docs",
                "purpose": "collection_search",
                "collection_id": "manual",
                "max_candidates": 20,
                "quality_mode": "standard",
                "case_id": None,
            }
        ),
        content_type="application/json",
    )
    request.tenant = tenant

    response = start_rerank_workflow(request)
    assert response.status_code == 200
    data = json.loads(response.content)

    assert data["status"] == "completed"
    assert data["results"][0]["title"] == "Alpha"
    assert data["telemetry"]["nodes"]["k_generate_strategy"]["status"] == "completed"
