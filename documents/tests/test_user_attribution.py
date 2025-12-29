from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from django_tenants.utils import schema_context

from ai_core.adapters.db_documents_repository import DbDocumentsRepository
from customers.models import Tenant
from documents.contracts import DocumentMeta, DocumentRef, InlineBlob, NormalizedDocument
from documents.domain_service import DocumentDomainService
from documents.models import Document, DocumentCollectionMembership
from users.models import User


pytestmark = pytest.mark.django_db


def _sample_document(*, tenant_schema: str, document_id=None) -> NormalizedDocument:
    doc_id = document_id or uuid4()
    checksum = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    return NormalizedDocument(
        ref=DocumentRef(
            tenant_id=tenant_schema,
            document_id=doc_id,
            workflow_id="test",
        ),
        meta=DocumentMeta(
            tenant_id=tenant_schema,
            workflow_id="test",
            title="Test Doc",
        ),
        blob=InlineBlob(
            type="inline",
            media_type="text/plain",
            base64="aGVsbG8=",
            sha256=checksum,
            size=5,
        ),
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="crawler",
    )


def test_ingest_document_sets_created_by_and_membership_actor(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    service = DocumentDomainService(
        ingestion_dispatcher=lambda *args: None,
        allow_missing_vector_store_for_tests=True,
    )

    with schema_context(tenant.schema_name):
        user = User.objects.create_user(username="owner")
        collection = service.ensure_collection(tenant=tenant, key="alpha", name="Alpha")

        result = service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="owner-hash",
            collections=(collection,),
            audit_meta={
                "created_by_user_id": str(user.id),
                "last_hop_service_id": "upload-worker",
            },
        )

        document = Document.objects.get(id=result.document.id)
        assert document.created_by_id == user.id
        assert document.updated_by_id == user.id

        membership = DocumentCollectionMembership.objects.get(
            document=document, collection=collection
        )
        assert membership.added_by_user_id == user.id
        assert membership.added_by_service_id is None

        other_user = User.objects.create_user(username="editor")
        service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="owner-hash",
            collections=(collection,),
            audit_meta={"created_by_user_id": str(other_user.id)},
        )
        document.refresh_from_db()
        assert document.created_by_id == user.id
        assert document.updated_by_id == other_user.id


def test_ingest_document_sets_membership_actor_from_service(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    service = DocumentDomainService(
        ingestion_dispatcher=lambda *args: None,
        allow_missing_vector_store_for_tests=True,
    )

    with schema_context(tenant.schema_name):
        collection = service.ensure_collection(tenant=tenant, key="beta", name="Beta")
        result = service.ingest_document(
            tenant=tenant,
            source="crawler",
            content_hash="service-hash",
            collections=(collection,),
            audit_meta={"last_hop_service_id": "crawler-worker"},
        )

        membership = DocumentCollectionMembership.objects.get(
            document=result.document, collection=collection
        )
        assert membership.added_by_user_id is None
        assert membership.added_by_service_id == "crawler-worker"


def test_db_repository_applies_audit_meta_created_by(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    repository = DbDocumentsRepository()

    with schema_context(tenant.schema_name):
        owner = User.objects.create_user(username="owner-repo")
        editor = User.objects.create_user(username="editor-repo")

        doc = _sample_document(tenant_schema=tenant.schema_name)

        repository.upsert(
            doc,
            audit_meta={
                "created_by_user_id": str(owner.id),
                "last_hop_service_id": "crawler-worker",
            },
        )

        stored = Document.objects.get(id=doc.ref.document_id)
        assert stored.created_by_id == owner.id
        assert stored.updated_by_id == owner.id

        repository.upsert(
            doc,
            audit_meta={"created_by_user_id": str(editor.id)},
        )

        stored.refresh_from_db()
        assert stored.created_by_id == owner.id
        assert stored.updated_by_id == editor.id
