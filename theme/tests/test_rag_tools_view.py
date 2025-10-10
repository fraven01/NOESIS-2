import pytest
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
    assert 'const derivedTenantId = "testserver"' in content
    assert 'const defaultEmbeddingProfile' in content
