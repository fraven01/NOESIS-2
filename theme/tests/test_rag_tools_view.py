import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core.cache import cache
from django.test import RequestFactory
from django.urls import reverse
from django.utils.html import escapejs

from ai_core.rag.collections import manual_collection_uuid
from theme.views import (
    rag_tools,
    start_rerank_workflow,
    web_search,
    web_search_ingest_selected,
)


def test_rag_tools_page_is_accessible():
    tenant_id = "tenant-workbench"
    tenant_schema = "workbench"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)
    request.tenant_schema = tenant_schema

    response = rag_tools(request)
    manual_collection_id = str(manual_collection_uuid(tenant_id))

    assert response.status_code == 200
    content = response.content.decode()

    assert "RAG Manual Testing" in content
    assert "Upload Document" in content
    assert "Ingestion Control" in content
    assert "Ingestion Status" in content
    assert "Query" in content
    assert "Aktive Collection" in content
    assert "query-collection-options" in content
    assert "Crawler Runner" in content
    assert "crawler-form" in content
    assert f"X-Tenant-ID: {tenant_id}" in content
    assert f"X-Tenant-Schema: {tenant_schema}" in content
    assert f'const derivedTenantId = "{escapejs(tenant_id)}"' in content
    assert f'const derivedTenantSchema = "{escapejs(tenant_schema)}"' in content
    assert "const defaultEmbeddingProfile" in content
    assert "const allowDocClassAlias" in content
    assert "const crawlerRunnerUrl" in content
    assert "const crawlerDefaultWorkflow" in content
    assert f'const manualCollectionId = "{escapejs(manual_collection_id)}"' in content
    assert "function requireCollection" in content


@patch("theme.views.build_graph")
def test_web_search_uses_external_knowledge_graph(mock_build_graph):
    """Test that web_search view uses ExternalKnowledgeGraph orchestration.

    The view only performs search phase; ingestion is handled separately.
    """
    tenant_id = "tenant-test"
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

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search"),
        data=json.dumps({"query": "test query", "collection_id": "test-collection"}),
        content_type="application/json",
    )
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

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


@patch("theme.views.build_graph")
def test_web_search_defaults_to_manual_collection(mock_build_graph):
    tenant_id = "tenant-fallback"
    tenant_schema = "fallback"

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
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search(request)

    assert response.status_code == 200
    mock_graph.run.assert_called_once()
    manual_id = str(manual_collection_uuid(tenant_id))
    call_args = mock_graph.run.call_args
    assert call_args[1]["state"]["collection_id"] == manual_id


@patch("theme.views.ensure_manual_collection")
@patch("httpx.Client")
def test_web_search_ingest_selected_defaults_to_manual_collection(
    mock_client_class, mock_ensure
):
    tenant_id = "tenant-auto"
    tenant_schema = "auto"

    mock_ensure.return_value = "manual-uuid"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"origins": []}
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client_class.return_value = mock_client

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps({"urls": ["https://fallback.example"]}),
        content_type="application/json",
    )
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    assert mock_client.post.call_count == 1
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["collection_id"] == "manual-uuid"
    mock_ensure.assert_called_once_with(tenant_id)
    headers = mock_client.post.call_args.kwargs["headers"]
    assert headers["X-Tenant-Schema"] == tenant_schema


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
    tenant_id = "tenant-test"
    tenant_schema = "test"

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
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "succeeded"
    assert data["results"][0]["title"] == "Beta"
    assert data["results"][0]["rerank"]["score"] == 88
    task_payload = mock_submit_task.call_args.kwargs["task_payload"]
    assert task_payload["control"]["model_preset"] == "fast"


@patch("theme.views.submit_worker_task", return_value=({"task_id": "task-q"}, False))
@patch("theme.views.build_graph")
def test_web_search_rerank_returns_queue_status(mock_build_graph, _mock_submit_task):
    cache.clear()
    tenant_id = "tenant-queued"
    tenant_schema = "queued"

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
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search(request)
    data = json.loads(response.content)

    assert response.status_code == 200
    assert data["rerank"]["status"] == "queued"
    assert data["rerank"]["status_url"].endswith("/api/llm/tasks/task-q/")


@patch("theme.views.ensure_manual_collection", return_value="manual-uuid")
@patch("httpx.Client")
def test_web_search_ingest_selected(mock_client_class, _mock_ensure):
    """Test that web_search_ingest_selected calls crawler_runner API for selected URLs."""
    tenant_id = "tenant-test"
    tenant_schema = "test"

    # Mock HTTP client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "origins": [
            {"url": "https://example.com", "status": "completed"},
            {"url": "https://test.com", "status": "completed"},
        ]
    }
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client_class.return_value = mock_client

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
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert response_data["url_count"] == 2
    assert "result" in response_data

    # Verify httpx.Client().post was called once
    assert mock_client.post.call_count == 1


@patch("theme.views.ensure_manual_collection", return_value="manual-uuid")
@patch("httpx.Client")
def test_web_search_ingest_selected_passes_correct_params_to_crawler(
    mock_client_class,
    _mock_ensure,
):
    """Test that web_search_ingest_selected passes all required parameters to crawler_runner API.

    This test validates that:
    1. collection_id from the request payload is passed through to crawler_runner
    2. tenant_id is correctly propagated via HTTP headers
    3. workflow_id is set to "web-search-ingestion"
    4. case_id is passed through via HTTP headers
    5. The crawler_runner API receives properly formatted payload
    """
    tenant_id = "tenant-test"
    tenant_schema = "test"

    # Mock HTTP client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"document_id": "doc-123"}
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client_class.return_value = mock_client

    factory = RequestFactory()
    request = factory.post(
        reverse("web-search-ingest-selected"),
        data=json.dumps(
            {
                "urls": ["https://example.com"],
                "collection_id": "test-collection",
                "case_id": "test-case",
            }
        ),
        content_type="application/json",
    )
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = web_search_ingest_selected(request)

    assert response.status_code == 200
    response_data = json.loads(response.content)
    assert response_data["status"] == "completed"
    assert response_data["url_count"] == 1

    # Verify httpx.Client().post was called once
    assert mock_client.post.call_count == 1

    # Verify the payload and headers passed to crawler_runner API
    call_kwargs = mock_client.post.call_args.kwargs

    # Validate JSON payload
    payload = call_kwargs["json"]
    assert payload["workflow_id"] == "web-search-ingestion"
    assert payload["mode"] == "live"
    assert payload["collection_id"] == "test-collection"
    assert len(payload["origins"]) == 1
    assert payload["origins"][0]["url"] == "https://example.com"

    # Validate HTTP headers
    headers = call_kwargs["headers"]
    assert headers["X-Tenant-ID"] == tenant_id
    assert headers["X-Tenant-Schema"] == tenant_schema
    assert headers["X-Case-ID"] == "test-case"
    assert "X-Trace-ID" in headers


@patch("theme.views.submit_worker_task")
def test_start_rerank_workflow_returns_completed(mock_submit_worker_task):
    tenant_id = "tenant-test"
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
                "case_id": "local",
            }
        ),
        content_type="application/json",
    )
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = start_rerank_workflow(request)
    assert response.status_code == 200
    data = json.loads(response.content)

    assert data["status"] == "completed"
    assert data["results"][0]["title"] == "Alpha"
    assert data["telemetry"]["nodes"]["k_generate_strategy"]["status"] == "completed"


def test_rag_tools_page_includes_source_transformation_logic():
    """Test that the RAG tools page includes JavaScript logic for source transformation.

    This test verifies that the transformSnippetSource function is present in the
    rendered HTML, which is responsible for converting internal source metadata
    (like file paths) into user-facing source information (like clickable links).
    """
    tenant_id = "tenant-test"
    tenant_schema = "test"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = rag_tools(request)

    assert response.status_code == 200
    content = response.content.decode()

    # Verify that the source transformation function exists
    assert "function transformSnippetSource" in content

    # Verify key transformation logic patterns are present
    assert "meta.external_id" in content  # Web source detection
    assert "startsWith('web::')" in content  # Web source prefix check
    assert "meta.workflow_id" in content  # Upload source detection
    assert "meta.title" in content  # Filename extraction
    assert "meta.filename" in content  # Alternative filename field
    assert "/documents/download/" in content  # Download URL pattern

    # Verify that the rendering logic uses the transformation
    assert "transformSnippetSource(snippet)" in content
    assert "sourceInfo.url" in content
    assert "sourceInfo.displayText" in content

    # Verify that links are created for sources with URLs
    assert "link.target = '_blank'" in content
    assert "link.rel = 'noopener noreferrer'" in content


def test_rag_tools_javascript_source_transformation_web_sources():
    """Test that the JavaScript source transformation handles web sources correctly.

    This is a documentation test that verifies the expected behavior for web sources:
    - Sources with meta.external_id starting with "web::" should be extracted as URLs
    - The display text should be derived from the URL hostname
    - A clickable link should be created
    """
    tenant_id = "tenant-test"
    tenant_schema = "test"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = rag_tools(request)
    content = response.content.decode()

    # Verify web source detection logic
    assert "if (externalId.startsWith('web::'))" in content
    assert "const url = externalId.substring(5)" in content  # Remove "web::" prefix
    assert "new URL(url)" in content  # Parse URL
    assert "urlObj.hostname" in content  # Extract hostname for display


def test_rag_tools_javascript_source_transformation_upload_sources():
    """Test that the JavaScript source transformation handles upload sources correctly.

    This is a documentation test that verifies the expected behavior for uploaded files:
    - Sources with meta.workflow_id should be treated as uploaded documents
    - The display text should be extracted from meta.title or meta.filename
    - A download URL should be constructed from meta.document_id
    """
    tenant_id = "tenant-test"
    tenant_schema = "test"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = rag_tools(request)
    content = response.content.decode()

    # Verify upload source detection logic
    assert "if (meta.workflow_id" in content
    assert "if (meta.title" in content  # Title extraction
    assert "if (meta.filename" in content  # Filename extraction
    assert "url = '/documents/download/' + meta.document_id + '/'" in content


def test_rag_tools_javascript_source_transformation_fallback():
    """Test that the JavaScript source transformation provides fallback behavior.

    This is a documentation test that verifies the expected behavior for unknown sources:
    - If no specific pattern matches, use citation or source value
    - If neither is available, use document_id
    - No URL should be generated (plain text display)
    """
    tenant_id = "tenant-test"
    tenant_schema = "test"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)

    response = rag_tools(request)
    content = response.content.decode()

    # Verify fallback logic
    assert "let fallbackText = citationValue" in content
    assert "if (fallbackText === '–' && meta.document_id)" in content
    assert "return { displayText: fallbackText, url: null }" in content
