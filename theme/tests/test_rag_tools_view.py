import pytest
from django.conf import settings
from django.urls import reverse


@pytest.mark.django_db
def test_rag_tools_page_is_accessible(client):
    response = client.get(reverse("rag-tools"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "RAG Manual Testing" in content
    assert "Upload Document" in content
    assert "Ingestion Control" in content
    assert "Ingestion Status" in content
    assert "Query" in content
    assert "X-Tenant-ID: testserver" in content
    default_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"
    assert f"X-Tenant-Schema: {default_schema}" in content
    assert 'const derivedTenantId = "testserver"' in content
    assert f'const derivedTenantSchema = "{default_schema}"' in content
    assert "const defaultEmbeddingProfile" in content
