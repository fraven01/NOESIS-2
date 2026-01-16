"""Tests for Document Versioning (SOTA 5.3) logic in domain service."""

import pytest
from documents.domain_service import DocumentDomainService
from documents.models import Document, DocumentVersion


@pytest.mark.django_db
class TestDocumentVersioning:
    def test_version_creation_on_first_ingest(self):
        """Verify initial version is created with sequence 1."""
        from django.db import connection
        from django_tenants.utils import get_public_schema_name
        from testsupport.tenants import ensure_tenant
        from users.models import User

        schema_name = connection.schema_name
        if schema_name == get_public_schema_name():
            schema_name = "autotest"

        tenant = ensure_tenant(schema_name, migrate=False)
        user, _ = User.objects.get_or_create(
            username="testuser", defaults={"email": "test@example.com"}
        )
        service = DocumentDomainService(ingestion_dispatcher=lambda *args: None)

        content_hash = "hash_v1"
        source = "test_source"

        result = service.ingest_document(
            tenant=tenant,
            source=source,
            content_hash=content_hash,
            audit_meta={"created_by_user_id": user.id},
            metadata={"title": "Test Doc"},
        )

        doc = result.document
        assert doc.metadata["document_version_id"] is not None
        assert doc.metadata["version"] == "v1"
        assert doc.metadata["is_latest"] is True

        # DB check
        versions = list(DocumentVersion.objects.filter(document=doc))
        assert len(versions) == 1
        v1 = versions[0]
        assert v1.sequence == 1
        assert v1.version_label == "v1"
        assert v1.is_latest is True
        assert str(v1.id) == doc.metadata["document_version_id"]

    def test_version_increment_on_update(self):
        """Verify re-ingesting increments sequence and manages is_latest."""
        from django.db import connection
        from django_tenants.utils import get_public_schema_name
        from testsupport.tenants import ensure_tenant
        from users.models import User

        schema_name = connection.schema_name
        if schema_name == get_public_schema_name():
            schema_name = "autotest"

        tenant = ensure_tenant(schema_name, migrate=False)
        user, _ = User.objects.get_or_create(
            username="testuser", defaults={"email": "test@example.com"}
        )
        service = DocumentDomainService(ingestion_dispatcher=lambda *args: None)

        source = "test_source"
        content_hash_1 = "hash_v1"
        content_hash_2 = "hash_v2"

        # 1. First Ingest
        res1 = service.ingest_document(
            tenant=tenant,
            source=source,
            content_hash=content_hash_1,
            audit_meta={"created_by_user_id": user.id},
        )
        doc_id = res1.document.id

        # 2. Second Ingest (Update)
        res2 = service.ingest_document(
            tenant=tenant,
            source=source,
            content_hash=content_hash_2,
            document_id=doc_id,  # EXPLICITLY Updating this doc
            audit_meta={"created_by_user_id": user.id},
        )

        assert res2.document.id == doc_id

        versions = DocumentVersion.objects.filter(document_id=doc_id).order_by(
            "sequence"
        )
        assert len(versions) == 2

        v1 = versions[0]
        v2 = versions[1]

        assert v1.sequence == 1
        assert v1.version_label == "v1"
        assert v1.is_latest is False  # Demoted

        assert v2.sequence == 2
        assert v2.version_label == "v2"
        assert v2.is_latest is True  # New latest

        # Doc metadata should point to v2
        doc = Document.objects.get(id=doc_id)
        assert doc.metadata["document_version_id"] == str(v2.id)
        assert doc.metadata["version"] == "v2"

    def test_idempotency_same_hash(self):
        """Verify sending SAME hash creates new version (interpreted as re-ingestion/update)."""
        from django.db import connection
        from django_tenants.utils import get_public_schema_name
        from testsupport.tenants import ensure_tenant
        from users.models import User

        schema_name = connection.schema_name
        if schema_name == get_public_schema_name():
            schema_name = "autotest"

        tenant = ensure_tenant(schema_name, migrate=False)
        user, _ = User.objects.get_or_create(
            username="testuser", defaults={"email": "test@example.com"}
        )
        service = DocumentDomainService(ingestion_dispatcher=lambda *args: None)

        res1 = service.ingest_document(
            tenant=tenant,
            source="src",
            content_hash="hash",
            audit_meta={"created_by_user_id": user.id},
        )

        # Ingest SAME hash again
        res2 = service.ingest_document(
            tenant=tenant,
            source="src",
            content_hash="hash",
            document_id=res1.document.id,
            audit_meta={"created_by_user_id": user.id},
        )

        assert DocumentVersion.objects.filter(document=res1.document).count() == 2
        assert res2.document.metadata["version"] == "v2"
