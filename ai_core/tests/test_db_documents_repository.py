import pytest
from datetime import datetime, timezone
from uuid import uuid4
from ai_core.adapters.db_documents_repository import DbDocumentsRepository
from documents.contracts import (
    NormalizedDocument,
    DocumentRef,
    DocumentMeta,
    InlineBlob,
    Asset,
    AssetRef,
)
from documents.models import DocumentCollection
from customers.models import Tenant


@pytest.mark.django_db
class TestDbDocumentsRepository:

    @pytest.fixture
    def repository(self):
        return DbDocumentsRepository()

    @pytest.fixture
    def tenant(self, test_tenant_schema_name):
        return Tenant.objects.get(schema_name=test_tenant_schema_name)

    @pytest.fixture
    def collection(self, tenant):
        return DocumentCollection.objects.create(
            tenant=tenant, collection_id=uuid4(), name="default"
        )

    def test_upsert_fails_on_missing_collection(self, repository, tenant):
        """Test that upsert raises ValueError if the referenced collection does not exist."""
        doc_id = uuid4()
        missing_collection_id = uuid4()

        doc = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant.schema_name,
                document_id=doc_id,
                workflow_id="test",
                collection_id=missing_collection_id,
                version="v1",
            ),
            meta=DocumentMeta(
                tenant_id=tenant.schema_name,
                workflow_id="test",
                document_collection_id=missing_collection_id,
                title="Test Doc",
            ),
            blob=InlineBlob(
                type="inline",
                media_type="text/plain",
                base64="aGVsbG8=",
                sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                size=5,
            ),
            checksum="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(
            ValueError, match=f"Collection not found: {missing_collection_id}"
        ):
            repository.upsert(doc)

    def test_list_latest_by_collection_deduplication(
        self, repository, tenant, collection
    ):
        """Test that list_latest_by_collection returns only the latest version of a document."""
        doc_id = uuid4()

        # Create version 1 (older)
        doc_v1 = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant.schema_name,
                document_id=doc_id,
                workflow_id="test",
                collection_id=collection.collection_id,
                version="v1",
            ),
            meta=DocumentMeta(
                tenant_id=tenant.schema_name,
                workflow_id="test",
                document_collection_id=collection.collection_id,
                title="Test Doc V1",
            ),
            blob=InlineBlob(
                type="inline",
                media_type="text/plain",
                base64="aGVsbG8=",
                sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                size=5,
            ),
            checksum="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            created_at=datetime.fromisoformat("2023-01-01T10:00:00+00:00"),
        )
        repository.upsert(doc_v1)

        # Create version 2 (newer)
        doc_v2 = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant.schema_name,
                document_id=doc_id,
                workflow_id="test",
                collection_id=collection.collection_id,
                version="v2",
            ),
            meta=DocumentMeta(
                tenant_id=tenant.schema_name,
                workflow_id="test",
                document_collection_id=collection.collection_id,
                title="Test Doc V2",
            ),
            blob=InlineBlob(
                type="inline",
                media_type="text/plain",
                base64="d29ybGQ=",
                sha256="486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7",
                size=5,
            ),
            checksum="486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7",
            created_at=datetime.fromisoformat("2023-01-02T10:00:00+00:00"),
        )
        repository.upsert(doc_v2)

        # List should prevent duplicate document_id
        refs, _ = repository.list_latest_by_collection(
            tenant.schema_name, collection.collection_id
        )

        assert len(refs) == 1
        assert refs[0].version == "v2"

    def test_asset_persistence(self, repository, tenant, collection):
        """Test adding, getting, listing and deleting assets."""
        doc_id = uuid4()

        # Setup parent document
        doc = NormalizedDocument(
            ref=DocumentRef(
                tenant_id=tenant.schema_name,
                document_id=doc_id,
                workflow_id="test",
                collection_id=collection.collection_id,
                version="v1",
            ),
            meta=DocumentMeta(
                tenant_id=tenant.schema_name,
                workflow_id="test",
                document_collection_id=collection.collection_id,
                title="Test Doc",
            ),
            blob=InlineBlob(
                type="inline",
                media_type="text/plain",
                base64="aGVsbG8=",
                sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                size=5,
            ),
            checksum="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            created_at=datetime.now(timezone.utc),
        )
        repository.upsert(doc)

        # Create Asset
        asset_id = uuid4()
        asset = Asset(
            ref=AssetRef(
                tenant_id=tenant.schema_name,
                workflow_id="test",
                asset_id=asset_id,
                document_id=doc_id,
                collection_id=collection.collection_id,
            ),
            media_type="text/plain",
            blob=InlineBlob(
                type="inline",
                media_type="text/plain",
                base64="aGVsbG8=",
                sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                size=5,
            ),
            checksum="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            caption_method="none",
            created_at=datetime.now(timezone.utc),
        )

        # ADD
        saved = repository.add_asset(asset)
        assert saved.ref.asset_id == asset_id

        # GET
        fetched = repository.get_asset(tenant.schema_name, asset_id)
        assert fetched is not None
        assert fetched.ref.asset_id == asset_id
        assert fetched.blob.sha256 == asset.blob.sha256

        # LIST
        refs, cursor = repository.list_assets_by_document(tenant.schema_name, doc_id)
        assert len(refs) == 1
        assert refs[0].asset_id == asset_id

        # DELETE
        deleted = repository.delete_asset(tenant.schema_name, asset_id, hard=True)
        assert deleted is True

        fetched_after = repository.get_asset(tenant.schema_name, asset_id)
        assert fetched_after is None
