from types import SimpleNamespace

from django.test import RequestFactory
from django.urls import reverse
from django.utils.html import escapejs

from theme.views import rag_tools


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
