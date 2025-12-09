from uuid import uuid4

import pytest
from django.test import RequestFactory
from django.urls import reverse
from django.utils import timezone
from django_tenants.utils import get_public_schema_name, schema_context

from customers.tests.factories import TenantFactory
from cases.models import Case
from documents.contracts import DocumentMeta, DocumentRef, FileBlob, NormalizedDocument
from documents.models import DocumentCollection
from documents.repository import InMemoryDocumentsRepository
from documents.services.document_space_service import DocumentSpaceService

from theme.views import document_space


@pytest.mark.django_db
def test_document_space_requires_tenant():
    request = RequestFactory().get(reverse("document-space"))

    with schema_context(get_public_schema_name()):
        response = document_space(request)

    assert response.status_code == 403
    assert "Tenant could not be resolved" in response.content.decode()


@pytest.mark.django_db
def test_document_space_lists_collections_and_documents(monkeypatch):
    tenant = TenantFactory(schema_name="explore")
    with schema_context(tenant.schema_name):
        case = Case.objects.create(
            tenant=tenant, external_id="CASE-123", title="Framework Einführung"
        )
        collection = DocumentCollection.objects.create(
            tenant=tenant,
            case=case,
            name="Framework Docs",
            key="framework-docs",
            collection_id=uuid4(),
            type="framework",
        )

    repo = InMemoryDocumentsRepository()
    document_id = uuid4()
    workflow_id = "framework-analysis"

    checksum_value = "a" * 64
    normalized = NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant.schema_name,
            workflow_id=workflow_id,
            document_id=document_id,
            collection_id=collection.collection_id,
            document_collection_id=collection.id,
            version="v1",
        ),
        meta=DocumentMeta(
            tenant_id=tenant.schema_name,
            workflow_id=workflow_id,
            title="Digitale KBV",
            language="de",
            tags=["framework", "kbv"],
            document_collection_id=collection.id,
            origin_uri="https://example.com/kbv.pdf",
        ),
        blob=FileBlob(
            type="file",
            uri="object://tenant/framework/kbv.pdf",
            sha256=checksum_value,
            size=1024,
        ),
        checksum=checksum_value,
        created_at=timezone.now(),
        source="upload",
    )
    repo.upsert(normalized)

    monkeypatch.setattr("theme.views._get_documents_repository", lambda: repo)

    class _CollectionServiceStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def ensure_manual_collection(self, tenant: object, **_: object) -> str:
            self.calls.append(str(tenant))
            return str(collection.collection_id)

    service_stub = DocumentSpaceService(collection_service=_CollectionServiceStub())
    monkeypatch.setattr("theme.views.DOCUMENT_SPACE_SERVICE", service_stub)

    request = RequestFactory().get(
        reverse("document-space"), {"collection": str(collection.id)}
    )
    request.tenant = tenant
    request.tenant_schema = tenant.schema_name

    response = document_space(request)

    assert response.status_code == 200
    content = response.content.decode()
    assert "Document Space Explorer" in content
    assert "Framework Docs" in content
    assert "Digitale KBV" in content
    assert str(document_id) in content
    assert "Workflow Filter" in content
    assert "Download" in content
