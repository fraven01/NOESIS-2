"""Unit tests for ``AssetIngestionPipeline`` behaviour."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4, uuid5

from documents.asset_ingestion import AssetIngestionPipeline, CaptionResult


class DummyStorage:
    """In-memory storage stub for `BlobStorageAdapter`."""

    def __init__(self):
        self.put_calls = []

    def put(self, payload: bytes):
        checksum = hashlib.sha256(payload).hexdigest()
        uri = f"store://{len(payload)}"
        self.put_calls.append(payload)
        return uri, checksum, len(payload)

    def get(self, uri: str):
        raise KeyError("not used for content uploads")


class DummyRepository:
    """Simple repository stub to capture interactions."""

    def __init__(self, existing_asset=None):
        self.existing_asset = existing_asset
        self.added = []
        self.get_args = []

    def get_asset(self, tenant_id, asset_id, workflow_id):
        self.get_args.append((tenant_id, asset_id, workflow_id))
        return self.existing_asset

    def add_asset(self, asset_obj, workflow_id):
        self.added.append((asset_obj, workflow_id))
        return SimpleNamespace(stored=True)


class ContractsStub:
    """Contracts stub that records provided identifiers."""

    def __init__(self):
        self.created_refs = []
        self.created_assets = []

    def asset_ref(
        self,
        tenant_id,
        workflow_id,
        asset_id,
        document_id,
        collection_id,
    ):
        ref = SimpleNamespace(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            asset_id=asset_id,
            document_id=document_id,
            collection_id=collection_id,
        )
        self.created_refs.append(ref)
        return ref

    def asset(
        self,
        ref,
        media_type,
        blob,
        origin_uri,
        page_index,
        bbox,
        context_before,
        context_after,
        ocr_text,
        text_description,
        caption_method,
        caption_model,
        caption_confidence,
        caption_source,
        parent_ref,
        perceptual_hash,
        asset_kind,
        created_at,
        checksum,
    ):
        asset_obj = SimpleNamespace(
            ref=ref,
            media_type=media_type,
            blob=blob,
            origin_uri=origin_uri,
            page_index=page_index,
            bbox=bbox,
            context_before=context_before,
            context_after=context_after,
            ocr_text=ocr_text,
            text_description=text_description,
            caption_method=caption_method,
            caption_model=caption_model,
            caption_confidence=caption_confidence,
            caption_source=caption_source,
            parent_ref=parent_ref,
            perceptual_hash=perceptual_hash,
            asset_kind=asset_kind,
            created_at=created_at,
            checksum=checksum,
        )
        self.created_assets.append(asset_obj)
        return asset_obj


class CaptionResolverStub:
    """Deterministic caption resolver for tests."""

    def __init__(self):
        self.called_with = None

    def resolve(self, metadata):
        self.called_with = metadata
        return CaptionResult(
            text="resolved", source="test", method="manual", confidence=0.5
        )


class ParsedAssetStub:
    """Minimal parsed asset for pipeline tests."""

    def __init__(self, content: bytes, metadata=None):
        self.content = content
        self.file_uri = None
        self.media_type = "application/octet-stream"
        self.metadata = metadata or {}
        self.bbox = (0, 0, 1, 1)
        self.page_index = 0
        self.context_before = "before"
        self.context_after = "after"


def _create_pipeline(existing_asset=None):
    storage = DummyStorage()
    repository = DummyRepository(existing_asset=existing_asset)
    contracts = ContractsStub()
    catalog = CaptionResolverStub()
    pipeline = AssetIngestionPipeline(
        repository=repository,
        storage=storage,
        contracts=contracts,
        caption_resolver=catalog,
    )
    return pipeline, repository, storage, contracts, catalog


def test_persist_asset_stores_new_asset():
    """Happy path stores blob, builds asset, and persists it."""
    pipeline, repository, storage, contracts, resolver = _create_pipeline()
    parsed_asset = ParsedAssetStub(content=b"payload", metadata={"locator": "img-01"})
    document_id = uuid4()

    stored = pipeline.persist_asset(
        index=0,
        parsed_asset=parsed_asset,
        tenant_id="tenant-test",
        workflow_id="upload",
        document_id=document_id,
        collection_id="col-a",
        created_at="now",
    )

    assert resolver.called_with == parsed_asset.metadata
    assert repository.added, "Repository should receive asset for persistence"
    assert stored is not None
    assert contracts.created_refs, "Contracts.asset_ref must be invoked"
    generated_asset_id = contracts.created_refs[-1].asset_id
    expected_asset_id = uuid5(document_id, "asset:img-01")
    assert generated_asset_id == expected_asset_id
    assert repository.get_args, "Pipeline should check for duplicate assets"
    assert storage.put_calls, "Blob storage should be invoked for binary content"


def test_persist_asset_returns_existing_on_duplicate():
    """Deduplication path returns cached asset when checksum matches."""
    parsed_asset = ParsedAssetStub(content=b"payload", metadata={"locator": "dup"})
    document_id = uuid4()
    checksum = hashlib.sha256(parsed_asset.content).hexdigest()
    existing_asset = SimpleNamespace(checksum=checksum, reused=True)

    pipeline, repository, storage, _, _ = _create_pipeline(
        existing_asset=existing_asset
    )

    result = pipeline.persist_asset(
        index=1,
        parsed_asset=parsed_asset,
        tenant_id="tenant-test",
        workflow_id="upload",
        document_id=document_id,
        collection_id="col-b",
        created_at="now",
    )

    assert result is existing_asset
    assert repository.added == [], "Existing duplicates should skip persistence"
    assert repository.get_args, "Pipeline still queries repository for deduplication"
