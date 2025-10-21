from django.conf import settings
from django.test.utils import override_settings
from django.urls import reverse

from theme.forms import RagIngestionForm, RagQueryForm, RagStatusForm, RagUploadForm


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

    assert "RAG Operations Workbench" in content
    assert "Dokument hochladen" in content
    assert "Ingestion Run starten" in content
    assert "Ingestion Status prüfen" in content
    assert "RAG Query ausführen" in content
    assert "query-collection-options" in content

    assert isinstance(response.context["upload_form"], RagUploadForm)
    assert isinstance(response.context["ingestion_form"], RagIngestionForm)
    assert isinstance(response.context["status_form"], RagStatusForm)
    assert isinstance(response.context["query_form"], RagQueryForm)

    default_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"
    assert f"{default_schema}" in content
    assert "testserver" in content
