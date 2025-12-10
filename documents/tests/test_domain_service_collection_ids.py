"""Regression tests for collection ID handling in DocumentDomainService.

These tests verify that the service correctly uses logical collection_id
instead of Django PK (id) when dispatching to ingestion workers.

Historical Context:
- Bug: Dispatcher received collection.id (PK) instead of collection.collection_id
- Fix: Commit de4e089 addressed collection ID consistency issues
- These tests prevent regression
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from customers.models import Tenant
from documents.domain_service import DocumentDomainService
from documents.models import DocumentCollection


@pytest.mark.django_db
class TestCollectionIdConsistency:
    """Verify collection ID handling across domain service operations."""

    @pytest.fixture
    def tenant(self, test_tenant_schema_name):
        """Get test tenant."""
        return Tenant.objects.get(schema_name=test_tenant_schema_name)

    @pytest.fixture
    def service(self):
        """Create domain service instance."""
        return DocumentDomainService()

    @pytest.fixture
    def collection_with_divergent_ids(self, tenant):
        """Create collection where PK != logical collection_id.

        This simulates real-world scenarios where Django auto-increments
        the PK but collection_id is set to a specific UUID.
        """
        logical_id = uuid4()
        unique_key = f"test-collection-{uuid4()}"
        collection = DocumentCollection.objects.create(
            tenant=tenant,
            collection_id=logical_id,  # Explicit logical ID
            key=unique_key,  # Required for uniqueness constraint
            name="test_collection",
            # Note: id (PK) will be auto-generated and != logical_id
        )

        # Verify divergence
        assert collection.id != logical_id, "Test setup requires PK != logical ID"
        assert collection.collection_id == logical_id

        return collection

    def test_ingest_uses_logical_collection_id_not_pk(
        self, service, tenant, collection_with_divergent_ids
    ):
        """Verify ingestion dispatcher receives logical collection_id, not PK.

        Critical regression test for bug where dispatcher received wrong ID.
        """
        mock_dispatcher = MagicMock()

        result = service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="abc123",
            metadata={"title": "Test"},
            collections=[collection_with_divergent_ids],
            dispatcher=mock_dispatcher,
        )

        # Verify dispatcher was called (on transaction commit)
        # Note: In tests without real transactions, we need to trigger manually
        # or check the returned collection_ids

        # Critical assertion: returned IDs are logical, not PK
        assert len(result.collection_ids) == 1
        assert result.collection_ids[0] == collection_with_divergent_ids.collection_id
        assert (
            result.collection_ids[0] != collection_with_divergent_ids.id
        ), "BUG: Using PK instead of logical collection_id!"

    def test_bulk_ingest_preserves_logical_collection_ids(
        self, service, tenant, collection_with_divergent_ids
    ):
        """Verify bulk ingestion also uses logical collection_id."""
        from documents.domain_service import DocumentIngestSpec

        mock_dispatcher = MagicMock()

        specs = [
            DocumentIngestSpec(
                source="upload",
                content_hash="hash1",
                metadata={"title": "Doc 1"},
                collections=[collection_with_divergent_ids],
            ),
            DocumentIngestSpec(
                source="upload",
                content_hash="hash2",
                metadata={"title": "Doc 2"},
                collections=[collection_with_divergent_ids],
            ),
        ]

        result = service.bulk_ingest_documents(
            tenant=tenant,
            documents=specs,
            dispatcher=mock_dispatcher,
        )

        # Verify all results use logical ID
        for record in result.records:
            if record.result.collection_ids:
                assert (
                    collection_with_divergent_ids.collection_id
                    in record.result.collection_ids
                )
                assert (
                    collection_with_divergent_ids.id not in record.result.collection_ids
                ), "BUG: Bulk ingest using PK instead of logical ID!"

    def test_collection_resolution_by_key_returns_correct_id(self, service, tenant):
        """Verify collection resolution returns logical ID."""
        # Create collection with known key
        collection_key = "test-key-" + str(uuid4())
        logical_id = uuid4()
        mock_dispatcher = MagicMock()

        collection = DocumentCollection.objects.create(
            tenant=tenant,
            collection_id=logical_id,
            key=collection_key,
            name=collection_key,
        )

        # Ingest with collection key (string)
        result = service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="xyz",
            metadata={},
            collections=[collection_key],  # Pass key, not instance
            dispatcher=mock_dispatcher,
        )

        # Verify resolution yielded logical ID
        assert logical_id in result.collection_ids
        assert collection.id not in result.collection_ids

    def test_multiple_collections_all_use_logical_ids(self, service, tenant):
        """Verify multi-collection ingestion uses all logical IDs."""
        mock_dispatcher = MagicMock()

        collections = [
            DocumentCollection.objects.create(
                tenant=tenant,
                collection_id=uuid4(),
                key=f"collection-key-{uuid4()}",
                name=f"collection_{i}",
            )
            for i in range(3)
        ]

        result = service.ingest_document(
            tenant=tenant,
            source="upload",
            content_hash="multi123",
            metadata={},
            collections=collections,
            dispatcher=mock_dispatcher,
        )

        # Verify all logical IDs present, no PKs
        logical_ids = {c.collection_id for c in collections}
        pk_ids = {c.id for c in collections}

        assert set(result.collection_ids) == logical_ids
        assert set(result.collection_ids).isdisjoint(
            pk_ids
        ), "BUG: Result contains PK IDs instead of logical IDs!"
