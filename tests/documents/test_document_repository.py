from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

import base64
import hashlib

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


def _make_document(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID | None = None,
    collection_id: UUID | None = None,
    version: str | None = None,
    created_at: datetime | None = None,
    checksum: str = "a" * 64,
    blob: object | None = None,
    assets: list[Asset] | None = None,
) -> NormalizedDocument:
    doc_id = document_id or uuid4()
    created = created_at or datetime.now(timezone.utc)
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=doc_id,
        collection_id=collection_id,
        version=version,
    )
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title="Sample",
        tags=["alpha"],
    )
    doc_blob = blob or FileBlob(
        type="file", uri="s3://bucket/doc", sha256=checksum, size=10
    )
    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=doc_blob,
        checksum=checksum,
        created_at=created,
        source="upload",
        assets=list(assets or []),
    )


def _make_asset(
    *,
    tenant_id: str,
    workflow_id: str = "workflow-1",
    document_id: UUID,
    asset_id: UUID | None = None,
    created_at: datetime | None = None,
    checksum: str = "b" * 64,
    blob: object | None = None,
) -> Asset:
    asset_uuid = asset_id or uuid4()
    created = created_at or datetime.now(timezone.utc)
    ref = AssetRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        asset_id=asset_uuid,
        document_id=document_id,
    )
    asset_blob = blob or FileBlob(
        type="file", uri="s3://bucket/asset", sha256=checksum, size=5
    )
    return Asset(
        ref=ref,
        media_type="image/png",
        blob=asset_blob,
        caption_method="none",
        created_at=created,
        checksum=checksum,
    )


def _make_inline_blob(payload: bytes, media_type: str) -> InlineBlob:
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=encoded,
        sha256=digest,
        size=len(payload),
    )


def test_upsert_and_get_roundtrip():
    repo = InMemoryDocumentsRepository()
    doc_id = uuid4()
    asset = _make_asset(tenant_id="tenant-a", document_id=doc_id)
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    stored = repo.upsert(doc)
    fetched = repo.get("tenant-a", doc_id, workflow_id=doc.ref.workflow_id)

    assert stored == fetched
    assert fetched is not None
    assert len(fetched.assets) == 1
    assert fetched.assets[0].ref.asset_id == asset.ref.asset_id


def test_upsert_idempotency_preserves_latest_payload():
    repo = InMemoryDocumentsRepository()
    doc_id = uuid4()
    first = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        version="v1",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    second = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        version="v1",
        checksum="c" * 64,
    )

    repo.upsert(first)
    stored = repo.upsert(second)

    assert stored.checksum == "c" * 64
    assert (
        repo.get("tenant-a", doc_id, "v1", workflow_id=second.ref.workflow_id).checksum
        == "c" * 64
    )


def test_get_prefer_latest_returns_most_recent_version():
    repo = InMemoryDocumentsRepository()
    doc_id = uuid4()
    first = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        version="v1",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    second = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        version="v2",
        created_at=datetime.now(timezone.utc),
        checksum="d" * 64,
    )
    repo.upsert(first)
    repo.upsert(second)

    latest = repo.get("tenant-a", doc_id, prefer_latest=True)
    assert latest is not None
    assert latest.ref.version == "v2"
    assert latest.checksum == "d" * 64


def test_prefer_latest_uses_lexicographic_version_tiebreaker():
    repo = InMemoryDocumentsRepository()
    doc_id = uuid4()
    moment = datetime.now(timezone.utc)
    collection_id = uuid4()
    first = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        collection_id=collection_id,
        version="v10",
        created_at=moment,
    )
    second = _make_document(
        tenant_id="tenant-a",
        document_id=doc_id,
        collection_id=collection_id,
        version="v2",
        created_at=moment,
        checksum="e" * 64,
    )

    repo.upsert(first)
    repo.upsert(second)

    latest = repo.get("tenant-a", doc_id, prefer_latest=True)
    assert latest is not None
    assert latest.ref.version == "v2"
    assert latest.checksum == "e" * 64

    refs, _ = repo.list_latest_by_collection("tenant-a", collection_id, limit=5)
    assert [ref.version for ref in refs] == ["v2"]


def test_list_by_collection_orders_by_created_at_desc():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    collection_id = uuid4()
    older = _make_document(
        tenant_id=tenant_id,
        document_id=uuid4(),
        collection_id=collection_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    newer = _make_document(
        tenant_id=tenant_id,
        document_id=uuid4(),
        collection_id=collection_id,
        created_at=datetime.now(timezone.utc),
    )
    other_collection = _make_document(
        tenant_id=tenant_id,
        document_id=uuid4(),
        collection_id=uuid4(),
    )
    other_tenant = _make_document(
        tenant_id="tenant-b",
        document_id=uuid4(),
        collection_id=collection_id,
    )

    for doc in (older, newer, other_collection, other_tenant):
        repo.upsert(doc)

    refs, cursor = repo.list_by_collection(tenant_id, collection_id, limit=10)
    assert cursor is None
    assert [ref.document_id for ref in refs] == [
        newer.ref.document_id,
        older.ref.document_id,
    ]


def test_list_by_collection_latest_only_flag():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    collection_id = uuid4()
    doc_id = uuid4()
    first = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        collection_id=collection_id,
        version="v1",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    second = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        collection_id=collection_id,
        version="v2",
        created_at=datetime.now(timezone.utc),
    )
    repo.upsert(first)
    repo.upsert(second)

    refs, cursor = repo.list_by_collection(
        tenant_id, collection_id, limit=5, latest_only=True
    )
    assert cursor is None
    assert [ref.version for ref in refs] == ["v2"]


def test_list_by_collection_paginates_with_cursor():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    collection_id = uuid4()
    docs = [
        _make_document(
            tenant_id=tenant_id,
            document_id=uuid4(),
            collection_id=collection_id,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=i),
        )
        for i in range(3)
    ]
    for doc in docs:
        repo.upsert(doc)

    first_page, cursor = repo.list_by_collection(tenant_id, collection_id, limit=1)
    assert cursor is not None
    assert [ref.document_id for ref in first_page] == [docs[0].ref.document_id]

    second_page, cursor_two = repo.list_by_collection(
        tenant_id, collection_id, limit=1, cursor=cursor
    )
    assert [ref.document_id for ref in second_page] == [docs[1].ref.document_id]
    assert cursor_two is not None

    third_page, final_cursor = repo.list_by_collection(
        tenant_id, collection_id, limit=1, cursor=cursor_two
    )
    assert [ref.document_id for ref in third_page] == [docs[2].ref.document_id]
    assert final_cursor is None


def test_list_by_collection_empty_cursor_returns_none():
    repo = InMemoryDocumentsRepository()
    cursor = InMemoryDocumentsRepository._encode_cursor(
        ["2024-01-01T00:00:00+00:00", str(uuid4()), "workflow-test", ""]
    )

    refs, next_cursor = repo.list_by_collection(
        "tenant-a", uuid4(), limit=1, cursor=cursor
    )
    assert refs == []
    assert next_cursor is None


def test_list_latest_by_collection_returns_latest_per_document():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    collection_id = uuid4()
    doc_id = uuid4()
    older = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        collection_id=collection_id,
        version="v1",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    newer = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        collection_id=collection_id,
        version="v2",
        created_at=datetime.now(timezone.utc),
    )
    repo.upsert(older)
    repo.upsert(newer)

    refs, cursor = repo.list_latest_by_collection(tenant_id, collection_id, limit=5)
    assert cursor is None
    assert [ref.version for ref in refs] == ["v2"]


def test_delete_document_soft_and_hard():
    repo = InMemoryDocumentsRepository()
    doc_id = uuid4()
    asset = _make_asset(tenant_id="tenant-a", document_id=doc_id)
    doc = _make_document(tenant_id="tenant-a", document_id=doc_id, assets=[asset])

    repo.upsert(doc)

    # Soft delete marks document as retired
    assert repo.delete("tenant-a", doc_id, workflow_id=doc.ref.workflow_id) is True
    deleted_doc = repo.get("tenant-a", doc_id, workflow_id=doc.ref.workflow_id)
    assert deleted_doc is not None, "Soft delete should return retired document"
    assert (
        deleted_doc.lifecycle_state == "retired"
    ), "Document should be marked as retired"
    assert len(deleted_doc.assets) == 0, "Assets should be cleared on soft delete"

    # Asset should also be retired/unavailable
    assert (
        repo.get_asset(
            "tenant-a", asset.ref.asset_id, workflow_id=asset.ref.workflow_id
        )
        is None
    )

    # Hard delete actually removes the document
    assert (
        repo.delete("tenant-a", doc_id, workflow_id=doc.ref.workflow_id, hard=True)
        is True
    )
    assert repo.get("tenant-a", doc_id, workflow_id=doc.ref.workflow_id) is None


def test_delete_document_returns_false_for_missing():
    repo = InMemoryDocumentsRepository()
    assert repo.delete("tenant-a", uuid4()) is False


def test_add_asset_requires_existing_document():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    asset = _make_asset(tenant_id="tenant-a", document_id=doc.ref.document_id)
    stored_asset = repo.add_asset(asset)
    assert stored_asset.ref.asset_id == asset.ref.asset_id

    fetched_doc = repo.get(
        "tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id
    )
    assert fetched_doc is not None
    assert {a.ref.asset_id for a in fetched_doc.assets} == {asset.ref.asset_id}

    with pytest.raises(ValueError):
        repo.add_asset(
            _make_asset(
                tenant_id="tenant-b",
                document_id=doc.ref.document_id,
            )
        )


def test_add_asset_rejects_workflow_mismatch():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a", workflow_id="workflow-a")
    repo.upsert(doc)

    mismatched_asset = _make_asset(
        tenant_id="tenant-a",
        workflow_id="workflow-b",
        document_id=doc.ref.document_id,
    )

    with pytest.raises(ValueError) as exc:
        repo.add_asset(mismatched_asset)

    assert str(exc.value) == "asset_workflow_mismatch"


def test_list_assets_by_document_sorted_and_limited():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    older = _make_asset(
        tenant_id="tenant-a",
        document_id=doc.ref.document_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    newer = _make_asset(
        tenant_id="tenant-a",
        document_id=doc.ref.document_id,
        created_at=datetime.now(timezone.utc),
    )
    repo.add_asset(older)
    repo.add_asset(newer)

    refs, cursor = repo.list_assets_by_document(
        "tenant-a",
        doc.ref.document_id,
        limit=2,
        workflow_id=doc.ref.workflow_id,
    )
    assert cursor is None
    assert [ref.asset_id for ref in refs] == [newer.ref.asset_id, older.ref.asset_id]


def test_list_assets_by_document_cursor_round_trip():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    first = _make_asset(
        tenant_id="tenant-a",
        document_id=doc.ref.document_id,
        created_at=datetime.now(timezone.utc),
    )
    second = _make_asset(
        tenant_id="tenant-a",
        document_id=doc.ref.document_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    repo.add_asset(second)
    repo.add_asset(first)

    page_one, cursor = repo.list_assets_by_document(
        "tenant-a",
        doc.ref.document_id,
        limit=1,
        workflow_id=doc.ref.workflow_id,
    )
    assert [ref.asset_id for ref in page_one] == [first.ref.asset_id]
    assert cursor is not None

    page_two, cursor_two = repo.list_assets_by_document(
        "tenant-a",
        doc.ref.document_id,
        limit=1,
        cursor=cursor,
        workflow_id=doc.ref.workflow_id,
    )
    assert [ref.asset_id for ref in page_two] == [second.ref.asset_id]
    assert cursor_two is None


def test_delete_asset_soft_and_hard():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    asset = _make_asset(tenant_id="tenant-a", document_id=doc.ref.document_id)
    repo.add_asset(asset)

    assert repo.delete_asset("tenant-a", asset.ref.asset_id) is True
    assert (
        repo.get_asset(
            "tenant-a", asset.ref.asset_id, workflow_id=asset.ref.workflow_id
        )
        is None
    )
    assert repo.delete_asset("tenant-a", asset.ref.asset_id) is False

    assert (
        repo.delete_asset(
            "tenant-a", asset.ref.asset_id, workflow_id=asset.ref.workflow_id, hard=True
        )
        is True
    )


def test_assets_follow_document_deletion_modes():
    repo = InMemoryDocumentsRepository()
    doc = _make_document(tenant_id="tenant-a")
    repo.upsert(doc)

    asset = _make_asset(tenant_id="tenant-a", document_id=doc.ref.document_id)
    repo.add_asset(asset)

    repo.delete("tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id)
    assert (
        repo.get_asset(
            "tenant-a", asset.ref.asset_id, workflow_id=asset.ref.workflow_id
        )
        is None
    )

    repo.delete(
        "tenant-a", doc.ref.document_id, workflow_id=doc.ref.workflow_id, hard=True
    )
    assert (
        repo.get_asset(
            "tenant-a", asset.ref.asset_id, workflow_id=asset.ref.workflow_id
        )
        is None
    )


def test_upsert_materializes_inline_document_and_assets():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    tenant_id = "tenant-a"
    doc_id = uuid4()

    document_payload = b"document bytes"
    document_blob = _make_inline_blob(document_payload, "application/pdf")
    document_checksum = hashlib.sha256(document_payload).hexdigest()

    asset_payload = b"asset-bytes"
    asset_blob = _make_inline_blob(asset_payload, "image/png")
    asset_checksum = hashlib.sha256(asset_payload).hexdigest()

    asset = _make_asset(
        tenant_id=tenant_id,
        document_id=doc_id,
        blob=asset_blob,
        checksum=asset_checksum,
    )
    doc = _make_document(
        tenant_id=tenant_id,
        document_id=doc_id,
        checksum=document_checksum,
        blob=document_blob,
        assets=[asset],
    )

    stored = repo.upsert(doc)
    assert isinstance(stored.blob, FileBlob)
    assert storage.get(stored.blob.uri) == document_payload
    assert stored.blob.sha256 == document_checksum
    assert stored.checksum == document_checksum

    assert stored.assets
    stored_asset = stored.assets[0]
    assert isinstance(stored_asset.blob, FileBlob)
    assert storage.get(stored_asset.blob.uri) == asset_payload
    assert stored_asset.blob.sha256 == asset_checksum
    assert stored_asset.checksum == asset_checksum


def test_add_asset_materializes_inline_blob_and_updates_document_view():
    storage = InMemoryStorage()
    repo = InMemoryDocumentsRepository(storage=storage)
    tenant_id = "tenant-a"
    doc = _make_document(tenant_id=tenant_id)
    repo.upsert(doc)

    asset_payload = b"inline-asset"
    inline_blob = _make_inline_blob(asset_payload, "image/png")
    checksum = hashlib.sha256(asset_payload).hexdigest()
    asset = _make_asset(
        tenant_id=tenant_id,
        document_id=doc.ref.document_id,
        blob=inline_blob,
        checksum=checksum,
    )

    stored_asset = repo.add_asset(asset)
    assert isinstance(stored_asset.blob, FileBlob)
    assert storage.get(stored_asset.blob.uri) == asset_payload
    assert stored_asset.checksum == checksum

    fetched_asset = repo.get_asset(
        tenant_id,
        stored_asset.ref.asset_id,
        workflow_id=stored_asset.ref.workflow_id,
    )
    assert isinstance(fetched_asset, Asset)
    assert isinstance(fetched_asset.blob, FileBlob)
    assert storage.get(fetched_asset.blob.uri) == asset_payload

    fetched_doc = repo.get(
        tenant_id, doc.ref.document_id, workflow_id=doc.ref.workflow_id
    )
    assert fetched_doc is not None
    assert fetched_doc.assets
    assert isinstance(fetched_doc.assets[0].blob, FileBlob)
    assert storage.get(fetched_doc.assets[0].blob.uri) == asset_payload


def test_workflow_scoped_document_operations():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    collection_id = uuid4()
    doc_id = uuid4()
    baseline = datetime.now(timezone.utc)
    first = _make_document(
        tenant_id=tenant_id,
        workflow_id="workflow-a",
        document_id=doc_id,
        collection_id=collection_id,
        created_at=baseline - timedelta(minutes=5),
    )
    second = _make_document(
        tenant_id=tenant_id,
        workflow_id="workflow-b",
        document_id=doc_id,
        collection_id=collection_id,
        created_at=baseline,
    )

    repo.upsert(first)
    repo.upsert(second)

    scoped_a = repo.get(tenant_id, doc_id, workflow_id="workflow-a")
    assert scoped_a is not None and scoped_a.ref.workflow_id == "workflow-a"

    scoped_b = repo.get(tenant_id, doc_id, workflow_id="workflow-b")
    assert scoped_b is not None and scoped_b.ref.workflow_id == "workflow-b"

    latest = repo.get(tenant_id, doc_id, prefer_latest=True)
    assert latest is not None and latest.ref.workflow_id == "workflow-b"

    filtered_refs, _ = repo.list_by_collection(
        tenant_id,
        collection_id,
        workflow_id="workflow-a",
    )
    assert [ref.workflow_id for ref in filtered_refs] == ["workflow-a"]

    repo.delete(tenant_id, doc_id, workflow_id="workflow-a")
    deleted_a = repo.get(tenant_id, doc_id, workflow_id="workflow-a")
    assert deleted_a is not None and deleted_a.lifecycle_state == "retired"
    assert repo.get(tenant_id, doc_id, workflow_id="workflow-b") is not None


def test_assets_are_isolated_per_workflow():
    repo = InMemoryDocumentsRepository()
    tenant_id = "tenant-a"
    doc_id = uuid4()

    doc_a = _make_document(
        tenant_id=tenant_id,
        workflow_id="workflow-a",
        document_id=doc_id,
    )
    doc_b = _make_document(
        tenant_id=tenant_id,
        workflow_id="workflow-b",
        document_id=doc_id,
    )
    repo.upsert(doc_a)
    repo.upsert(doc_b)

    shared_asset_id = uuid4()
    asset_a = _make_asset(
        tenant_id=tenant_id,
        workflow_id="workflow-a",
        document_id=doc_id,
        asset_id=shared_asset_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    asset_b = _make_asset(
        tenant_id=tenant_id,
        workflow_id="workflow-b",
        document_id=doc_id,
        asset_id=shared_asset_id,
    )
    repo.add_asset(asset_a)
    repo.add_asset(asset_b)

    scoped_a = repo.get_asset(tenant_id, shared_asset_id, workflow_id="workflow-a")
    assert scoped_a is not None and scoped_a.ref.workflow_id == "workflow-a"

    scoped_b = repo.get_asset(tenant_id, shared_asset_id, workflow_id="workflow-b")
    assert scoped_b is not None and scoped_b.ref.workflow_id == "workflow-b"

    default_asset = repo.get_asset(tenant_id, shared_asset_id)
    assert default_asset is not None and default_asset.ref.workflow_id == "workflow-b"

    refs, _ = repo.list_assets_by_document(
        tenant_id,
        doc_id,
        workflow_id="workflow-a",
    )
    assert [ref.workflow_id for ref in refs] == ["workflow-a"]

    repo.delete_asset(tenant_id, shared_asset_id, workflow_id="workflow-a")
    assert repo.get_asset(tenant_id, shared_asset_id, workflow_id="workflow-a") is None
    assert (
        repo.get_asset(tenant_id, shared_asset_id, workflow_id="workflow-b") is not None
    )
