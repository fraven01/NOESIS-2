import base64
from typing import Any
from uuid import UUID

import pytest
from django.conf import settings

from ai_core import ingestion_status
from ai_core.infra import object_store
from documents.contracts import InlineBlob, NormalizedDocument
from documents.repository import (
    DocumentsRepository,
)

from tests.plugins.rag_db import *  # noqa: F401,F403


class ObjectStoreDocumentsRepository(DocumentsRepository):
    """Test repository writing documents to the legacy object store."""

    def __init__(self) -> None:
        self.saved: list[NormalizedDocument] = []

    def upsert(
        self, doc: NormalizedDocument, workflow_id: str | None = None
    ) -> NormalizedDocument:
        self.saved.append(doc)
        tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
        workflow_segment = object_store.sanitize_identifier("upload")
        storage_prefix = f"{tenant_segment}/{workflow_segment}/uploads"
        filename = f"{doc.ref.document_id}_upload.bin"
        upload_path = f"{storage_prefix}/{filename}"
        payload = _extract_payload(doc.blob)
        object_store.put_bytes(upload_path, payload)
        object.__setattr__(doc.blob, "uri", upload_path)

        metadata = _build_metadata_snapshot(doc)
        object_store.write_json(
            f"{storage_prefix}/{doc.ref.document_id}.meta.json", metadata
        )
        return doc

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: str | None = None,
        *,
        prefer_latest: bool = False,
        workflow_id: str | None = None,
    ) -> NormalizedDocument | None:
        """Return a matching saved document or None.

        - If ``version`` is provided, return that exact version when present.
        - Otherwise return the most recent by ``created_at`` then ``version``.
        - When ``workflow_id`` is set, restrict to that workflow.
        """
        candidates = [
            d
            for d in self.saved
            if d.ref.tenant_id == tenant_id
            and d.ref.document_id == document_id
            and (workflow_id is None or d.ref.workflow_id == workflow_id)
        ]
        if not candidates:
            return None
        if version is not None:
            for d in candidates:
                if d.ref.version == version:
                    return d
            return None

        # Choose latest by created_at, then by version string (None -> "")
        def _order_key(d: NormalizedDocument):
            return (d.created_at, d.ref.version or "")

        return max(candidates, key=_order_key)


def _extract_payload(blob: Any) -> bytes:
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()
    base64_value = getattr(blob, "base64", None)
    if isinstance(base64_value, str):
        return base64.b64decode(base64_value)
    raise TypeError("unsupported_blob_type")


def _build_metadata_snapshot(doc: NormalizedDocument) -> dict[str, Any]:
    # The upload metadata persisted for tests is intentionally minimal.
    # Only expose keys that are consumed by the endpoints and assertions.
    payload: dict[str, Any] = {}

    external_ref = doc.meta.external_ref or {}
    external_id = external_ref.get("external_id")
    if external_id:
        payload["external_id"] = external_id

    if doc.ref.collection_id is not None:
        payload["collection_id"] = str(doc.ref.collection_id)

    if doc.ref.version is not None:
        payload["version"] = doc.ref.version

    return payload


@pytest.fixture
def documents_repository_stub(monkeypatch) -> ObjectStoreDocumentsRepository:
    from ai_core import services, views

    repository = ObjectStoreDocumentsRepository()
    monkeypatch.setattr(
        settings,
        "DOCUMENTS_REPOSITORY_CLASS",
        "ai_core.adapters.object_store_repository.ObjectStoreDocumentsRepository",
        raising=False,
    )
    monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", repository, raising=False)
    monkeypatch.setattr(services, "_DOCUMENTS_REPOSITORY", None, raising=False)
    try:
        yield repository
    finally:
        monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", None, raising=False)


@pytest.fixture(autouse=True)
def _auto_documents_repository(documents_repository_stub):
    yield


@pytest.fixture(autouse=True)
def ingestion_status_store(monkeypatch):
    from ai_core.tests.doubles import MemoryDocumentLifecycleStore

    store = MemoryDocumentLifecycleStore()
    monkeypatch.setattr(ingestion_status, "_LIFECYCLE_STORE", store, raising=False)
    yield store


@pytest.fixture(autouse=True)
def disable_async_graphs(monkeypatch):
    """Disable async graph execution in tests to run graphs synchronously."""
    from ai_core import services

    monkeypatch.setattr(services, "_should_enqueue_graph", lambda graph_name: False)
    yield
