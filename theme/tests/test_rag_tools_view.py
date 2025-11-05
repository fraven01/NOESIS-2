import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import RequestFactory
from django.urls import reverse
from django.utils.html import escapejs

from theme.views import rag_tools, web_search, web_search_ingest_selected


def test_rag_tools_page_is_accessible():
    tenant_id = "tenant-workbench"
    tenant_schema = "workbench"

    factory = RequestFactory()
    request = factory.get(reverse("rag-tools"))
    request.tenant = SimpleNamespace(tenant_id=tenant_id, schema_name=tenant_schema)
    request.tenant_schema = tenant_schema

    response = rag_tools(request)

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


@patch("theme.views.SimpleCrawlerIngestionAdapter")
def test_web_search_ingest_selected(mock_adapter_class):
    """Test that web_search_ingest_selected triggers ingestion for selected URLs."""
    tenant_id = "tenant-test"
    tenant_schema = "test"

    # Mock adapter
    mock_adapter = MagicMock()
    mock_adapter.trigger.return_value = MagicMock(
        decision="ingested",
        crawler_decision="accepted",
        document_id="doc-123",
    )
    mock_adapter_class.return_value = mock_adapter

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
    assert len(response_data["results"]) == 2
    assert response_data["results"][0]["decision"] == "ingested"

    # Verify adapter.trigger was called twice
    assert mock_adapter.trigger.call_count == 2
