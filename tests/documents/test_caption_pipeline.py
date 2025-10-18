import base64
import hashlib
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

import pytest

from documents.captioning import (
    AssetExtractionPipeline,
    DeterministicCaptioner,
    MultimodalCaptioner,
)
from documents.contracts import (
    Asset,
    AssetRef,
    DocumentMeta,
    DocumentRef,
    FileBlob,
    InlineBlob,
    NormalizedDocument,
)
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage


class _RecordingCaptioner(MultimodalCaptioner):
    def __init__(self, response_text: str = "generated caption") -> None:
        self.response_text = response_text
        self.contexts: List[Optional[str]] = []

    def caption(self, image: bytes, context: Optional[str] = None):
        self.contexts.append(context)
        return {
            "text_description": self.response_text,
            "confidence": 0.75,
            "model": "recording-stub",
            "tokens": len(image) // 4,
        }


class _EmptyCaptioner(MultimodalCaptioner):
    def caption(self, image: bytes, context: Optional[str] = None):
        return {
            "text_description": "",
            "confidence": 0.0,
            "model": "empty-stub",
            "tokens": len(image) // 4,
        }


class _MissingConfidenceCaptioner(MultimodalCaptioner):
    def caption(self, image: bytes, context: Optional[str] = None):
        return {
            "text_description": "generated",  # missing confidence on purpose
            "model": "stub",  # valid model ensures confidence check triggers
        }


def _make_inline_blob(payload: bytes, media_type: str = "image/png") -> InlineBlob:
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=encoded,
        sha256=digest,
        size=len(payload),
    )


def _make_document(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: Optional[UUID] = None,
    collection_id: Optional[UUID] = None,
    assets: Optional[list[Asset]] = None,
    checksum: str = "d" * 64,
) -> NormalizedDocument:
    doc_id = document_id or uuid4()
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=doc_id,
        collection_id=collection_id,
    )
    meta = DocumentMeta(tenant_id=tenant_id, workflow_id=workflow_id, title="Demo")
    blob = FileBlob(type="file", uri="memory://doc", sha256=checksum, size=1)
    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
        assets=list(assets or []),
    )


def _make_asset(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID,
    collection_id: Optional[UUID] = None,
    blob: FileBlob | InlineBlob,
    caption_method: str = "none",
    text_description: Optional[str] = None,
    context_before: Optional[str] = None,
    context_after: Optional[str] = None,
    ocr_text: Optional[str] = None,
) -> Asset:
    ref = AssetRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        asset_id=uuid4(),
        document_id=document_id,
        collection_id=collection_id,
    )
    return Asset(
        ref=ref,
        media_type="image/png",
        blob=blob,
        caption_method=caption_method,
        text_description=text_description,
        context_before=context_before,
        context_after=context_after,
        ocr_text=ocr_text,
        created_at=datetime.now(timezone.utc),
        checksum=blob.sha256,
    )


def test_pipeline_generates_caption_for_missing_description():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner(model_name="stub-det")
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    payload = b"sample-image"
    inline = _make_inline_blob(payload)
    doc_id = uuid4()
    doc = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        assets=[
            _make_asset(
                tenant_id="tenant-a",
                document_id=doc_id,
                blob=inline,
            )
        ],
    )

    stored = pipeline.process_document(doc)
    fetched = repo.get(
        "tenant-a", stored.ref.document_id, workflow_id=stored.ref.workflow_id
    )

    assert fetched is not None
    assert fetched.assets[0].text_description
    assert fetched.assets[0].caption_method == "vlm_caption"
    assert fetched.assets[0].caption_model == "stub-det"
    assert 0 <= fetched.assets[0].caption_confidence <= 1


def test_pipeline_skips_assets_with_existing_description():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    inline = _make_inline_blob(b"another")
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=inline,
        caption_method="manual",
        text_description="already provided",
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    stored = pipeline.process_document(doc)

    stored_asset = stored.assets[0]
    assert stored_asset.text_description == "already provided"
    assert stored_asset.caption_method == "manual"


def test_pipeline_uses_ocr_fallback_when_caption_empty():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = _EmptyCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    inline = _make_inline_blob(b"ocr-image")
    ocr_text = "OCR description " * 200
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=inline,
        ocr_text=ocr_text,
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    stored = pipeline.process_document(doc)
    stored_asset = stored.assets[0]

    assert stored_asset.caption_method == "ocr_only"
    assert stored_asset.caption_model is None
    assert stored_asset.caption_confidence is None
    assert len(stored_asset.text_description.encode("utf-8")) <= 2048


def test_pipeline_falls_back_when_caption_metadata_missing():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = _MissingConfidenceCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    inline = _make_inline_blob(b"meta-bytes")
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=inline,
        ocr_text="ocr-fallback-text",
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    stored = pipeline.process_document(doc)
    stored_asset = stored.assets[0]

    assert stored_asset.caption_method == "ocr_only"
    assert stored_asset.text_description == "ocr-fallback-text"
    assert stored_asset.caption_model is None
    assert stored_asset.caption_confidence is None


def test_pipeline_strict_mode_raises_on_caption_metadata_missing():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = _MissingConfidenceCaptioner()
    pipeline = AssetExtractionPipeline(
        repo,
        storage,
        captioner,
        strict_caption_validation=True,
    )

    inline = _make_inline_blob(b"meta-bytes")
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=inline,
        ocr_text="ocr-fallback-text",
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    with pytest.raises(ValueError, match="caption_confidence_missing"):
        pipeline.process_document(doc)

    assert (
        repo.get("tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id)
        is None
    )


def test_pipeline_raises_for_missing_blob():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    file_blob = FileBlob(type="file", uri="memory://missing", sha256="a" * 64, size=10)
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=file_blob,
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    with pytest.raises(ValueError, match="blob_missing"):
        pipeline.process_document(doc)

    assert (
        repo.get("tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id)
        is None
    )


def test_pipeline_process_assets_persists_new_assets():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner(model_name="stub-det")
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    inline = _make_inline_blob(b"asset-bytes")
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc.ref.document_id,
        blob=inline,
    )

    stored_assets = pipeline.process_assets("tenant-a", doc.ref.document_id, [asset])
    assert stored_assets[0].caption_method == "vlm_caption"
    assert stored_assets[0].caption_model == "stub-det"

    refs, _ = repo.list_assets_by_document(
        "tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id
    )
    assert len(refs) == 1


def test_pipeline_limits_context_before_passing_to_captioner():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = _RecordingCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    before = "Ä" * 600
    after = "ß" * 600
    inline = _make_inline_blob(b"context-bytes")
    doc_id = uuid4()
    asset = _make_asset(
        tenant_id="tenant-a",
        document_id=doc_id,
        blob=inline,
        context_before=before,
        context_after=after,
    )
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    pipeline.process_document(doc)

    assert captioner.contexts
    context = captioner.contexts[0]
    assert context is not None
    parts = context.split(pipeline.context_separator)
    assert all(
        0 < len(part.encode("utf-8")) <= pipeline.context_limit for part in parts
    )


def test_process_collection_updates_missing_captions():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner(model_name="stub-det")
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    collection_id = uuid4()
    inline = _make_inline_blob(b"collection-bytes")
    doc_id = uuid4()
    doc = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        collection_id=collection_id,
        assets=[
            _make_asset(
                tenant_id="tenant-a",
                document_id=doc_id,
                collection_id=collection_id,
                blob=inline,
            )
        ],
    )
    repo.upsert(doc)

    updated, next_cursor = pipeline.process_collection(
        "tenant-a", collection_id, limit=10
    )

    assert next_cursor is None
    assert updated
    assert updated[0].caption_method == "vlm_caption"

    stored = repo.get(
        "tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id
    )
    assert stored is not None
    assert stored.assets[0].caption_method == "vlm_caption"


def test_process_collection_skips_when_no_caption_needed():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    captioner = DeterministicCaptioner()
    pipeline = AssetExtractionPipeline(repo, storage, captioner)

    collection_id = uuid4()
    doc_id = uuid4()
    doc = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        collection_id=collection_id,
        assets=[
            _make_asset(
                tenant_id="tenant-a",
                document_id=doc_id,
                collection_id=collection_id,
                blob=_make_inline_blob(b"bytes"),
                text_description="already",
                caption_method="manual",
            )
        ],
    )
    repo.upsert(doc)

    updated, next_cursor = pipeline.process_collection(
        "tenant-a", collection_id, limit=5
    )

    assert updated == []
    assert next_cursor is None

