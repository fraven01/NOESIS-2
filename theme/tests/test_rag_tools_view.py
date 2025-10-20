from django.conf import settings
from django.test.utils import override_settings
from django.urls import reverse


@override_settings(
    MIDDLEWARE=[
        mw
        for mw in settings.MIDDLEWARE
        if mw
        not in {
            "django_tenants.middleware.main.TenantMainMiddleware",
            "common.middleware.HeaderTenantRoutingMiddleware",
            "common.middleware.TenantSchemaMiddleware",
            "ai_core.middleware.PIISessionScopeMiddleware",
        }
    ]
)
def test_rag_tools_page_is_accessible(client):
    response = client.get(reverse("rag-tools"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "RAG Manual Testing" in content
    assert "Upload Document" in content
    assert "Ingestion Control" in content
    assert "Ingestion Status" in content
    assert "Query" in content
    assert "Aktive Collection" in content
    assert "query-collection-options" in content
    assert "X-Tenant-ID: testserver" in content
    default_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"
    assert f"X-Tenant-Schema: {default_schema}" in content
    assert 'const derivedTenantId = "testserver"' in content
    assert f'const derivedTenantSchema = "{default_schema}"' in content
    assert "const defaultEmbeddingProfile" in content
    assert "const allowDocClassAlias" in content
    assert "function requireCollection" in content
