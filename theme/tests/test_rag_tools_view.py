import json
from unittest.mock import patch
from django.contrib.auth import get_user_model

import pytest
from django.core.cache import cache
from django.test import RequestFactory
from django.urls import reverse
from django_tenants.utils import get_public_schema_name, schema_context

from ai_core.rag.collections import manual_collection_uuid
from theme.views import (
    rag_tools,
    start_rerank_workflow,
    web_search,
    web_search_ingest_selected,
)


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_rag_tools_page_is_accessible(tenant_pool):
    tenant = tenant_pool["alpha"]
    tenant_schema = tenant.schema_name
    # tenant_id defaults to schema_name if no explicit tenant_id attribute exists
    tenant_id = getattr(tenant, "tenant_id", None) or tenant.schema_name

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = tenant
    request.tenant_schema = tenant_schema
    # RequestFactory doesn't create sessions, so we need to add one manually
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    request.session = SessionStore()

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff", "staff@example.com", "password", is_staff=True
    )
    request.user = user

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


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_rag_tools_requires_tenant():
    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff2", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    with schema_context(get_public_schema_name()):
        response = rag_tools(request)

    assert response.status_code == 403
    assert (
        json.loads(response.content)["error"]["message"]
        == "Tenant could not be resolved from request"
    )


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_rag_tools_rejects_spoofed_headers():
    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"), HTTP_X_TENANT_ID="spoofed")

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff3", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    with schema_context(get_public_schema_name()):
        response = rag_tools(request)

    assert response.status_code == 403
    assert (
        json.loads(response.content)["error"]["message"]
        == "Tenant could not be resolved from request"
    )


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views_web_search.submit_business_graph")
def test_web_search_uses_external_knowledge_graph(mock_submit, tenant_pool):
    """Test that web_search view uses Generic Worker for search."""

    mock_submit.return_value = (
        {
            "status": "success",
            "data": {
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
            },
        },
        True,
    )

    tenant = tenant_pool["alpha"]

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "test query", "collection_id": "test-collection"}),
        content_type="application/json",
    )
    request.tenant = tenant

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff4", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["outcome"] == "completed"
    assert "results" in response_data
    assert "trace_id" in response_data

    # Verify that submit was called with correct parameters
    mock_submit.assert_called_once()
    kwargs = mock_submit.call_args.kwargs
    assert kwargs["graph_name"] == "web_acquisition"
    state = kwargs["state"]
    # Web Acquisition Graph Input Structure
    assert state["input"]["query"] == "test query"
    assert "collection_id" not in state["input"]  # collection_id is in Context now


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views_web_search.submit_business_graph")
def test_web_search_htmx_returns_partial(mock_submit, tenant_pool):
    """Test that web_search returns HTML partial for HTMX requests."""
    tenant = tenant_pool["alpha"]

    mock_submit.return_value = (
        {
            "status": "success",
            "data": {
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
            },
        },
        True,
    )

    factory = RequestFactory()
    # Simulate HTMX request with form-encoded data (default for hx-post)
    request = factory.post(
        reverse("web-search"),
        data={"query": "htmx query", "collection_id": "test-collection"},
    )
    request.headers = {"HX-Request": "true"}
    request.tenant = tenant

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff5", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search(request)

    assert response.status_code == 200
    content = response.content.decode()
    # Should return the partial, not JSON
    assert "HTMX Result" in content
    assert "class=" in content  # Basic HTML check
    assert "Snippet" in content


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views_web_search.submit_business_graph")
def test_web_search_defaults_to_manual_collection(mock_submit, tenant_pool):
    tenant = tenant_pool["alpha"]
    tenant_id = tenant.schema_name

    mock_submit.return_value = (
        {
            "status": "success",
            "data": {
                "output": {
                    "decision": "no_results",
                    "search_results": [],
                    "selected_result": None,
                    "ingestion_result": None,
                    "error": None,
                    "auto_ingest": False,
                }
            },
        },
        True,
    )

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "fallback query"}),
        content_type="application/json",
    )
    request.tenant = tenant

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff6", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search(request)

    assert response.status_code == 200
    mock_submit.assert_called_once()
    manual_id = str(manual_collection_uuid(tenant_id))

    kwargs = mock_submit.call_args.kwargs
    # Check tool_context argument
    tool_context = kwargs["tool_context"]
    assert tool_context.business.collection_id == manual_id


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("documents.collection_service.CollectionService.ensure_manual_collection")
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected_defaults_to_manual_collection(
    mock_dispatch, mock_ensure, tenant_pool
):
    tenant = tenant_pool["alpha"]
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

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff7", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    assert mock_dispatch.call_count == 1
    call_args = mock_dispatch.call_args
    request_model = call_args.args[0]
    assert request_model.collection_id == manual_uuid
    mock_ensure.assert_called_once_with(tenant_id)


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views.llm_routing.resolve")
@patch("theme.views.submit_business_graph")
@patch("theme.views_web_search.submit_business_graph")
def test_web_search_rerank_applies_scores(
    mock_submit_graph, mock_submit_task, mock_resolve, settings, tenant_pool
):
    settings.RERANK_MODEL_PRESET = "meta/llama-3.1-70b-instruct"

    def fake_resolve(label: str):
        if label == "meta/llama-3.1-70b-instruct":
            raise ValueError("unknown label")
        return "resolved-model"

    mock_resolve.side_effect = fake_resolve

    cache.clear()
    tenant = tenant_pool["alpha"]

    # Mock the initial search graph call (M-2)
    mock_submit_graph.return_value = (
        {
            "status": "success",
            "data": {
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
            },
        },
        True,
    )

    # Mock the rerank worker call
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

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff8", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "succeeded"
    assert data["results"][0]["title"] == "Beta"
    assert data["results"][0]["rerank"]["score"] == 88
    assert data["results"][0]["rerank"]["score"] == 88

    # Verify graph input
    call_kwargs = mock_submit_task.call_args.kwargs
    graph_state = call_kwargs["state"]
    # Provide backward compatibility check or simply check known fields
    assert graph_state["input"]["search"]["results"][0]["document_id"] == "doc-a"


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views.submit_business_graph", return_value=({"task_id": "task-q"}, False))
@patch("theme.views_web_search.submit_business_graph")
def test_web_search_rerank_returns_queue_status(
    mock_submit_graph, _mock_submit_task, tenant_pool
):
    cache.clear()
    tenant = tenant_pool["alpha"]

    mock_submit_graph.return_value = (
        {
            "status": "success",
            "data": {
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

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff9", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "queued"
    assert data["rerank"]["status_url"].endswith("/api/llm/tasks/task-q/")


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch(
    "documents.collection_service.CollectionService.ensure_manual_collection",
    return_value="22222222-2222-2222-2222-222222222222",
)
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected(mock_dispatch, _mock_ensure, tenant_pool):
    """Test that web_search_ingest_selected dispatches via CrawlerManager."""
    tenant = tenant_pool["alpha"]

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

    request.tenant = tenant

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff10", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert mock_dispatch.call_count == 1


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch(
    "documents.collection_service.CollectionService.ensure_manual_collection",
    return_value="44444444-4444-4444-4444-444444444444",
)
@patch("crawler.manager.CrawlerManager.dispatch_crawl_request")
def test_web_search_ingest_selected_passes_correct_params_to_crawler(
    mock_dispatch,
    _mock_ensure,
    tenant_pool,
):
    """Ensure CrawlerManager receives the expected request model.

    This test validates that:
    1. collection_id from the request payload is passed through to CrawlerManager
    2. workflow_id is set to "web-search-ingestion"
    3. origins list is properly constructed from URLs
    """
    tenant = tenant_pool["alpha"]

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

    request.tenant = tenant

    # Auth setup
    # Re-use staff10 if strictly sequential or create new
    User = get_user_model()
    user = User.objects.create_user(
        "staff11", "staff@example.com", "password", is_staff=True
    )
    request.user = user

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


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@patch("theme.views_rag_tools.submit_business_graph")
def test_start_rerank_workflow_returns_completed(
    mock_submit_business_graph, tenant_pool
):
    telemetry_payload = {
        "graph": "collection_search",
        "nodes": {"k_generate_strategy": {"status": "completed"}},
    }
    search_payload = {
        "results": [{"title": "Alpha", "url": "https://example.com", "score": 0.3}],
        "responses": [],
    }
    mock_submit_business_graph.return_value = (
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

    tenant = tenant_pool["alpha"]

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
    # RequestFactory doesn't create sessions, but prepare_workbench_context needs one
    from django.contrib.sessions.backends.db import SessionStore

    request.session = SessionStore()

    request.session = SessionStore()

    # Auth setup
    User = get_user_model()
    user = User.objects.create_user(
        "staff12", "staff@example.com", "password", is_staff=True
    )
    request.user = user

    response = start_rerank_workflow(request)
    assert response.status_code == 200
    data = json.loads(response.content)

    assert data["status"] == "completed"
    assert data["results"][0]["title"] == "Alpha"
    assert data["telemetry"]["nodes"]["k_generate_strategy"]["status"] == "completed"
