import json
from unittest.mock import MagicMock, patch

import pytest
from django.core.cache import cache
from django.http import JsonResponse
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
@patch("theme.views.build_graph")
def test_web_search_uses_external_knowledge_graph(mock_build_graph):
    """Test that web_search view uses ExternalKnowledgeGraph orchestration.

        The view only performs search phase; ingestion is handled separately.
    x"""
    tenant_schema = "test"

    # Mock graph
    mock_graph = MagicMock()
    mock_graph.run.return_value = (
        {
            "search": {
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test",
                        "snippet": "Test snippet",
                    }
                ]
            },
            "selection": {"selected": [], "shortlisted": []},
            "ingestion": {},
        },
        {"outcome": "completed", "telemetry": {}},
    )
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

    # Verify that build_graph was called
    mock_build_graph.assert_called_once()

    # Verify that graph.run was called with correct parameters
    mock_graph.run.assert_called_once()
    call_args = mock_graph.run.call_args
    assert call_args[1]["state"]["query"] == "test query"
    # Always runs until after_search for manual testing
    assert call_args[1]["state"]["run_until"] == "after_search"
    assert call_args[1]["state"]["collection_id"] == "test-collection"


@pytest.mark.django_db
@patch("theme.views.build_graph")
def test_web_search_htmx_returns_partial(mock_build_graph):
    """Test that web_search returns HTML partial for HTMX requests."""
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    mock_graph = MagicMock()
    mock_graph.run.return_value = (
        {
            "search": {
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "HTMX Result",
                        "snippet": "Snippet",
                    }
                ]
            },
            "selection": {"selected": [], "shortlisted": []},
            "ingestion": {},
        },
        {"outcome": "completed", "telemetry": {}},
    )
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
@patch("theme.views.build_graph")
def test_web_search_defaults_to_manual_collection(mock_build_graph):
    tenant_schema = "fallback"
    tenant = TenantFactory(schema_name=tenant_schema)
    tenant_id = tenant.schema_name

    mock_graph = MagicMock()
    mock_graph.run.return_value = (
        {"search": {"results": []}, "selection": {"selected": [], "shortlisted": []}},
        {"outcome": "completed", "telemetry": {}},
    )
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
    mock_graph.run.assert_called_once()
    manual_id = str(manual_collection_uuid(tenant_id))
    call_args = mock_graph.run.call_args
    assert call_args[1]["state"]["collection_id"] == manual_id


@pytest.mark.django_db
@patch("documents.collection_service.CollectionService.ensure_manual_collection")
@patch("theme.views.crawl_selected")
def test_web_search_ingest_selected_defaults_to_manual_collection(
    mock_crawl_selected, mock_ensure
):
    tenant_schema = "auto"
    tenant = TenantFactory(schema_name=tenant_schema)
    tenant_id = tenant.schema_name

    mock_ensure.return_value = "manual-uuid"

    mock_crawl_selected.return_value = JsonResponse({"task_ids": []})

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps({"urls": ["https://fallback.example"]}),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    assert mock_crawl_selected.call_count == 1
    forward_request = mock_crawl_selected.call_args.args[0]
    forwarded_payload = json.loads(forward_request.body.decode())
    assert forwarded_payload["collection_id"] == "manual-uuid"
    mock_ensure.assert_called_once_with(tenant_id)
    assert forward_request.META["HTTP_X_TENANT_SCHEMA"] == tenant_schema


@pytest.mark.django_db
@patch("theme.views.llm_routing.resolve")
@patch("theme.views.submit_worker_task")
@patch("theme.views.build_graph")
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

    mock_graph = MagicMock()
    mock_graph.run.return_value = (
        {
            "search": {
                "results": [
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
                ]
            },
            "selection": {"selected": [], "shortlisted": []},
            "ingestion": {},
        },
        {"outcome": {"meta": {"review_required": False}}},
    )
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
@patch("theme.views.build_graph")
def test_web_search_rerank_returns_queue_status(mock_build_graph, _mock_submit_task):
    cache.clear()
    tenant_schema = "queued"
    tenant = TenantFactory(schema_name=tenant_schema)

    mock_graph = MagicMock()
    mock_graph.run.return_value = (
        {
            "search": {
                "results": [
                    {
                        "document_id": "doc-a",
                        "title": "Alpha",
                        "snippet": "Snippet A",
                        "url": "https://a.example",
                    }
                ]
            },
            "selection": {"selected": [], "shortlisted": []},
            "ingestion": {},
        },
        {"outcome": {"meta": {"review_required": False}}},
    )
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
    return_value="manual-uuid",
)
@patch("theme.views.crawl_selected")
def test_web_search_ingest_selected(mock_crawl_selected, _mock_ensure):
    """Test that web_search_ingest_selected dispatches crawl_selected for each URL."""
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)

    mock_crawl_selected.return_value = JsonResponse(
        {
            "status": "accepted",
            "task_ids": [
                {"task_id": "task-1", "origin": {"url": "https://example.com"}},
                {"task_id": "task-2", "origin": {"url": "https://test.com"}},
            ],
        }
    )

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps(
            {
                "urls": ["https://example.com", "https://test.com"],
                "collection_id": "test-collection",
            }
        ),
        content_type="application/json",
    )
    request.tenant = tenant

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert response_data["url_count"] == 2
    assert "result" in response_data
    assert mock_crawl_selected.call_count == 1


@pytest.mark.django_db
@patch(
    "documents.collection_service.CollectionService.ensure_manual_collection",
    return_value="manual-uuid",
)
@patch("theme.views.crawl_selected")
def test_web_search_ingest_selected_passes_correct_params_to_crawler(
    mock_crawl_selected,
    _mock_ensure,
):
    """Ensure crawler view receives the expected payload and headers.

    This test validates that:
    1. collection_id from the request payload is passed through to crawler_runner
    2. tenant_id is correctly propagated via HTTP headers
    3. workflow_id is set to "web-search-ingestion"
    4. case_id is passed through via HTTP headers
    5. The crawler_runner API receives properly formatted payload
    """
    tenant_schema = "test"
    tenant = TenantFactory(schema_name=tenant_schema)
    tenant_id = tenant.schema_name

    mock_crawl_selected.return_value = JsonResponse({"task_ids": []})

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps(
            {
                "urls": ["https://example.com"],
                "collection_id": "test-collection",
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
    assert response_data["url_count"] == 1
    assert mock_crawl_selected.call_count == 1

    # Verify the payload and headers passed to crawler_runner API
    call_request = mock_crawl_selected.call_args.args[0]
    payload = json.loads(call_request.body.decode())
    assert payload["workflow_id"] == "web-search-ingestion"
    assert payload["mode"] == "live"
    assert payload["collection_id"] == "test-collection"
    assert len(payload["urls"]) == 1
    assert payload["urls"][0] == "https://example.com"

    headers = call_request.META
    assert headers["HTTP_X_TENANT_ID"] == tenant_id
    assert headers["HTTP_X_TENANT_SCHEMA"] == tenant_schema
    assert headers["HTTP_X_CASE_ID"] == "TEST-CASE"
    assert "HTTP_X_TRACE_ID" in headers


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
