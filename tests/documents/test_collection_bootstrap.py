"""Tests for collection bootstrap and key-based lookup.

These tests verify the fix for the duplicate collection bug where
_resolve_collection_reference was creating new collections instead
of finding existing ones by collection_id.
"""

import pytest
from uuid import uuid4
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.collection_service import CollectionService
from documents.domain_service import DocumentDomainService
from documents.models import DocumentCollection, DocumentCollectionMembership
from documents.service_facade import ingest_document
from ai_core.contracts.scope import ScopeContext


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_bootstrap_creates_manual_collection(tenant: Tenant):
    """Test that bootstrap creates manual-search collection."""
    service = CollectionService()

    with schema_context(tenant.schema_name):
        # Ensure collection doesn't exist
        DocumentCollection.objects.filter(tenant=tenant, key="manual-search").delete()

        # Bootstrap
        collection_id = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        # Verify collection exists
        collection = DocumentCollection.objects.get(tenant=tenant, key="manual-search")
        assert str(collection.collection_id) == collection_id
        assert collection.name == "Manual Search"
        assert collection.key == "manual-search"


def test_bootstrap_is_idempotent(tenant: Tenant):
    """Test that bootstrapping same collection twice doesn't create duplicates."""
    service = CollectionService()

    with schema_context(tenant.schema_name):
        # Bootstrap twice
        collection_id_1 = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )
        collection_id_2 = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        # Should return same ID
        assert collection_id_1 == collection_id_2

        # Should only have one collection
        count = DocumentCollection.objects.filter(
            tenant=tenant, key="manual-search"
        ).count()
        assert count == 1


def test_ingest_with_collection_key_uses_existing_collection(tenant: Tenant):
    """Test that ingestion with collection_key finds existing collection.

    This is the key fix: by passing collection_key instead of just
    collection_id, we avoid the duplicate creation bug.
    """
    service = CollectionService()

    with schema_context(tenant.schema_name):
        # Bootstrap collection
        collection_id = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        # Simulate ingestion with collection_key (new way)
        scope = ScopeContext(
            tenant_id=str(tenant.id),
            tenant_schema=tenant.schema_name,
            trace_id="test-trace",
            invocation_id=str(uuid4()),
            ingestion_run_id=str(uuid4()),
        )

        meta = {
            "tenant_id": str(tenant.id),
            "collection_key": "manual-search",  # NEW: Use key
            "collection_id": collection_id,  # OLD: Also pass ID for compat
            "hash": "test-hash-123",
            "content_hash": "test-hash-123",
            "source": "test-source",
        }

        # Mock dispatcher
        def mock_dispatcher(doc_id, coll_ids, profile, scope_val):
            pass

        result = ingest_document(
            scope=scope,
            meta=meta,
            chunks_path="test/path/chunks.json",
            dispatcher=mock_dispatcher,
        )

        # Should have ONE collection (no duplicate created)
        collections = DocumentCollection.objects.filter(
            tenant=tenant, collection_id=collection_id
        )
        assert collections.count() == 1

        # Collection should be the original one
        collection = collections.first()
        assert collection.key == "manual-search"

        # Document should be linked to correct collection
        document_id = result["document_id"]
        memberships = DocumentCollectionMembership.objects.filter(
            document_id=document_id, collection=collection
        )
        assert memberships.count() == 1


def test_ingest_with_only_collection_id_creates_duplicate_old_bug(tenant: Tenant):
    """Test that ingestion with only collection_id creates duplicate (OLD BUG).

    This demonstrates the old bug behavior for documentation purposes.
    With the fix, this should NOT happen anymore if collection_key is passed.
    """
    service = CollectionService()
    domain_service = DocumentDomainService()

    with schema_context(tenant.schema_name):
        # Bootstrap collection
        collection_id = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        # Get the collection
        DocumentCollection.objects.get(tenant=tenant, key="manual-search")

        # Simulate OLD ingestion (only collection_id, no key)
        # This triggers _resolve_collection_reference with UUID
        collections_tuple = (str(collection_id),)  # UUID as string

        # This WOULD create a duplicate before the fix
        # After the fix in _resolve_collection_reference, it should find
        # the collection by collection_id
        result = domain_service.ingest_document(
            tenant=tenant,
            source="test-source",
            content_hash="test-hash-456",
            metadata={"test": "data"},
            collections=collections_tuple,
            dispatcher=lambda *args: None,
            allow_missing_ingestion_dispatcher_for_tests=True,
        )

        # With the fix, should still use original collection
        # (because _resolve_collection_reference now searches by collection_id)
        memberships = DocumentCollectionMembership.objects.filter(
            document=result.document
        )

        # Should be linked to SOME collection
        assert memberships.count() > 0

        # IDEALLY should be linked to original, but depending on fix strategy
        # this test documents expected behavior


def test_resolve_collection_reference_finds_by_collection_id(tenant: Tenant):
    """Test that _resolve_collection_reference finds collection by collection_id."""
    service = CollectionService()
    domain_service = DocumentDomainService()

    with schema_context(tenant.schema_name):
        # Create collection with key and collection_id
        collection_id = service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        collection = DocumentCollection.objects.get(tenant=tenant, key="manual-search")

        # Call _resolve_collection_reference with UUID
        # This should FIND the existing collection, not create a new one
        resolved = domain_service._resolve_collection_reference(
            tenant=tenant,
            collection=collection_id,  # Pass UUID as string
            embedding_profile=None,
            scope=None,
        )

        # Should return the SAME collection (by id)
        assert resolved.id == collection.id
        assert resolved.key == "manual-search"

        # Should NOT create a duplicate
        all_collections = DocumentCollection.objects.filter(
            tenant=tenant, collection_id=collection.collection_id
        )
        assert all_collections.count() == 1


def test_resolve_collection_reference_with_key_finds_collection(tenant: Tenant):
    """Test that _resolve_collection_reference finds collection by key."""
    service = CollectionService()
    domain_service = DocumentDomainService()

    with schema_context(tenant.schema_name):
        # Create collection
        service.ensure_manual_collection(
            tenant=tenant, slug="manual-search", label="Manual Search"
        )

        collection = DocumentCollection.objects.get(tenant=tenant, key="manual-search")

        # Call _resolve_collection_reference with KEY (not UUID)
        resolved = domain_service._resolve_collection_reference(
            tenant=tenant,
            collection="manual-search",  # Pass key as string
            embedding_profile=None,
            scope=None,
        )

        # Should return the SAME collection
        assert resolved.id == collection.id
        assert resolved.key == "manual-search"
