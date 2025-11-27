from __future__ import annotations

import uuid
from typing import Any

import pytest
from django.db import transaction
from django_tenants.utils import schema_context

from customers.models import Tenant
from testsupport.tenant_fixtures import create_test_tenant
from documents.domain_service import DocumentDomainService
from documents.models import Document, DocumentCollection, DocumentCollectionMembership


pytestmark = pytest.mark.django_db(transaction=True)


class _VectorStoreStub:
    def __init__(self) -> None:
        self.ensure_calls: list[dict[str, Any]] = []
        self.document_deletes: list[dict[str, Any]] = []
        self.collection_deletes: list[dict[str, Any]] = []

    def ensure_collection(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        embedding_profile: str | None = None,
        scope: str | None = None,
    ) -> None:
        self.ensure_calls.append(
            {
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "embedding_profile": embedding_profile,
                "scope": scope,
            }
        )

    def hard_delete_documents(
        self, *, tenant_id: str, document_ids: list[uuid.UUID]
    ) -> None:
        self.document_deletes.append(
            {"tenant_id": tenant_id, "document_ids": tuple(document_ids)}
        )

    def dispatch_delete(self, payload: dict[str, Any]) -> None:
        self.document_deletes.append(payload)

    def delete_collection(self, *, tenant_id: str, collection_id: str) -> None:
        self.collection_deletes.append(
            {"tenant_id": tenant_id, "collection_id": collection_id}
        )


@pytest.fixture(autouse=True)
def _silence_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("documents.domain_service.record_span", lambda *_, **__: None)


@pytest.fixture(autouse=True)
def _run_on_commit_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("django.db.transaction.on_commit", lambda fn, using=None: fn())


@pytest.fixture(autouse=True, scope="module")
def _preserve_module_tenant_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the shared module tenant alive across tests.

    The autouse cleanup fixture in ``conftest.py`` drops all tracked schemas
    after each test. Since this module reuses a module-scoped tenant, we need
    to ensure its schema is preserved for the duration of the suite.
    """

    from testsupport import tenant_fixtures

    def _cleanup(*, preserve=None):
        preserved = set(preserve or ())
        preserved.add("autotest-domain-service")
        return tenant_fixtures.cleanup_test_tenants(preserve=preserved)

    monkeypatch.setattr(tenant_fixtures, "cleanup_test_tenants", _cleanup)
    monkeypatch.setattr("conftest.cleanup_test_tenants", _cleanup)


@pytest.fixture
def vector_store() -> _VectorStoreStub:
    return _VectorStoreStub()


@pytest.fixture(scope="module")
def tenant(django_db_blocker) -> Tenant:
    with django_db_blocker.unblock():
        return create_test_tenant(schema_name="autotest-domain-service", migrate=True)


@pytest.mark.django_db
def test_ensure_collection_syncs_django_and_vector(
    tenant: Tenant, vector_store: _VectorStoreStub
):
    service = DocumentDomainService(vector_store=vector_store)

    with schema_context(tenant.schema_name):
        collection = service.ensure_collection(
            tenant=tenant,
            key="alpha",
            name="Alpha",
            embedding_profile="profile-a",
            scope="tenant-scope",
            metadata={"channel": "crawler"},
        )

        assert DocumentCollection.objects.filter(tenant=tenant, key="alpha").count() == 1

    assert vector_store.ensure_calls == [
        {
            "tenant_id": str(tenant.id),
            "collection_id": str(collection.collection_id),
            "embedding_profile": "profile-a",
            "scope": "tenant-scope",
        }
    ]


@pytest.mark.django_db
def test_ensure_collection_is_idempotent(tenant: Tenant, vector_store: _VectorStoreStub):
    service = DocumentDomainService(vector_store=vector_store)

    with schema_context(tenant.schema_name):
        first = service.ensure_collection(tenant=tenant, key="shared", name="Shared")
        second = service.ensure_collection(tenant=tenant, key="shared", name="Shared")

        assert first.id == second.id
        assert first.collection_id == second.collection_id
        assert (
            DocumentCollection.objects.filter(tenant=tenant, key="shared").count() == 1
        )

    assert len(vector_store.ensure_calls) == 2
    assert vector_store.ensure_calls[-1]["collection_id"] == str(first.collection_id)


@pytest.mark.django_db
def test_delete_document_removes_record_and_vector_entry(
    tenant: Tenant, vector_store: _VectorStoreStub
):
    service = DocumentDomainService(
        vector_store=vector_store, deletion_dispatcher=vector_store.dispatch_delete
    )

    with schema_context(tenant.schema_name):
        document = Document.objects.create(
            tenant=tenant,
            source="upload",
            hash="deadbeef",
            metadata={"filename": "report.pdf"},
        )

        doc_id = document.id
        service.delete_document(document)

        assert not Document.objects.filter(id=doc_id).exists()

    assert vector_store.document_deletes == [
        {
            "type": "document_delete",
            "document_id": str(doc_id),
            "document_ids": (str(doc_id),),
            "tenant_id": str(tenant.id),
            "reason": None,
        }
    ]


@pytest.mark.django_db
def test_document_flow_round_trip(tenant: Tenant, vector_store: _VectorStoreStub):
    dispatched: list[tuple[uuid.UUID, tuple[uuid.UUID, ...], str | None, str | None]] = []

    def _capture_dispatch(
        document_id: uuid.UUID,
        collection_ids: tuple[uuid.UUID, ...],
        embedding_profile: str | None,
        scope: str | None,
    ) -> None:
        dispatched.append((document_id, collection_ids, embedding_profile, scope))

    service = DocumentDomainService(
        vector_store=vector_store,
        ingestion_dispatcher=_capture_dispatch,
        deletion_dispatcher=vector_store.dispatch_delete,
    )

    with schema_context(tenant.schema_name), transaction.atomic():
        collection = service.ensure_collection(
            tenant=tenant,
            key="delta",
            name="Delta",
            embedding_profile="profile-d",
            scope="round-trip",
        )
        collection_id = collection.id
        collection_uuid = collection.collection_id

        ingest_result = service.ingest_document(
            tenant=tenant,
            source="crawler",
            content_hash="cafebabe",
            metadata={"filename": "delta.txt"},
            collections=(collection,),
            embedding_profile="profile-d",
            scope="round-trip",
        )
        document = ingest_result.document

        doc_id = document.id
        assert doc_id is not None
        assert Document.objects.filter(id=document.id).exists()
        assert DocumentCollectionMembership.objects.filter(
            document=document, collection=collection
        ).exists()

        service.delete_collection(collection, reason="cleanup")

    assert len(dispatched) == 1
    dispatched_document, dispatched_collections, dispatched_profile, dispatched_scope = dispatched[0]
    assert dispatched_document == doc_id
    assert dispatched_collections == (collection_id,)
    assert dispatched_profile == "profile-d"
    assert dispatched_scope == "round-trip"
    assert vector_store.ensure_calls
    assert vector_store.document_deletes[0]["tenant_id"] == str(tenant.id)
    assert vector_store.document_deletes[0]["document_ids"] == (doc_id,)
    assert vector_store.collection_deletes == [
        {"tenant_id": str(tenant.id), "collection_id": str(collection_uuid)}
    ]
    with schema_context(tenant.schema_name):
        assert not DocumentCollection.objects.filter(id=collection_id).exists()
        assert not DocumentCollectionMembership.objects.filter(document=document).exists()
        # Document remains for audit purposes until explicit removal via the domain service
        assert Document.objects.filter(id=doc_id).exists()

    with schema_context(tenant.schema_name):
        service.delete_document(document)

    assert vector_store.document_deletes[-1] == {
        "type": "document_delete",
        "document_id": str(doc_id),
        "document_ids": (str(doc_id),),
        "tenant_id": str(tenant.id),
        "reason": None,
    }
    with schema_context(tenant.schema_name):
        assert not Document.objects.filter(id=doc_id).exists()


def test_ingest_document_requires_dispatcher(tenant: Tenant):
    service = DocumentDomainService(vector_store=None)

    with schema_context(tenant.schema_name):
        with pytest.raises(ValueError):
            service.ingest_document(
                tenant=tenant,
                source="upload",
                content_hash="missing-dispatcher",
            )


def test_ingest_document_allows_test_flag_for_missing_dispatcher(tenant: Tenant):
    service = DocumentDomainService(
        vector_store=None, allow_missing_ingestion_dispatcher_for_tests=True
    )

    with schema_context(tenant.schema_name):
        result = service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="allow-missing-dispatcher",
        )

        assert Document.objects.filter(id=result.document.id).exists()


def test_delete_document_requires_dispatcher(tenant: Tenant, vector_store: _VectorStoreStub):
    service = DocumentDomainService(vector_store=vector_store)

    with schema_context(tenant.schema_name):
        document = Document.objects.create(
            tenant=tenant,
            source="upload",
            hash="missing-dispatcher",
            metadata={},
        )

        with pytest.raises(ValueError):
            service.delete_document(document)

        assert Document.objects.filter(id=document.id).exists()


@pytest.mark.django_db
def test_delete_collection_preserves_shared_documents(
    tenant: Tenant, vector_store: _VectorStoreStub
):
    service = DocumentDomainService(vector_store=vector_store)

    with schema_context(tenant.schema_name):
        primary_collection = service.ensure_collection(
            tenant=tenant, key="shared-primary", name="Shared Primary"
        )
        sibling_collection = service.ensure_collection(
            tenant=tenant, key="shared-sibling", name="Shared Sibling"
        )

        document = Document.objects.create(
            tenant=tenant,
            source="upload",
            hash="shared-hash",
            metadata={"filename": "shared.pdf"},
        )

        DocumentCollectionMembership.objects.bulk_create(
            [
                DocumentCollectionMembership(
                    document=document, collection=primary_collection
                ),
                DocumentCollectionMembership(
                    document=document, collection=sibling_collection
                ),
            ]
        )

        service.delete_collection(primary_collection)

        assert not DocumentCollection.objects.filter(id=primary_collection.id).exists()
        assert Document.objects.filter(id=document.id).exists()
        assert DocumentCollectionMembership.objects.filter(
            document=document, collection=sibling_collection
        ).exists()

    assert not vector_store.document_deletes
    assert vector_store.collection_deletes == [
        {
            "tenant_id": str(tenant.id),
            "collection_id": str(primary_collection.collection_id),
        }
    ]
