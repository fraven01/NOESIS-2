import base64
from typing import Any

import pytest

from ai_core.infra import object_store
from documents.contracts import InlineBlob, NormalizedDocument
from documents.repository import DocumentsRepository

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
        workflow_segment = object_store.sanitize_identifier(doc.ref.workflow_id)
        storage_prefix = f"{tenant_segment}/{workflow_segment}/uploads"
        filename = f"{doc.ref.document_id.hex}_upload.bin"
        payload = _extract_payload(doc.blob)
        object_store.write_bytes(f"{storage_prefix}/{filename}", payload)
        metadata = _build_metadata_snapshot(doc)
        object_store.write_json(
            f"{storage_prefix}/{doc.ref.document_id.hex}.meta.json", metadata
        )
        return doc


def _extract_payload(blob: Any) -> bytes:
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()
    base64_value = getattr(blob, "base64", None)
    if isinstance(base64_value, str):
        return base64.b64decode(base64_value)
    raise TypeError("unsupported_blob_type")


def _build_metadata_snapshot(doc: NormalizedDocument) -> dict[str, Any]:
    metadata = doc.meta.model_dump(mode="json", exclude_none=True)
    metadata.pop("tenant_id", None)
    metadata.pop("workflow_id", None)

    external_ref = metadata.pop("external_ref", None) or {}
    external_id = external_ref.get("external_id")
    for key, value in external_ref.items():
        if key == "external_id":
            continue
        metadata.setdefault(key, value)
    if external_id:
        metadata["external_id"] = external_id

    if doc.ref.collection_id is not None:
        metadata["collection_id"] = str(doc.ref.collection_id)
    if doc.ref.version is not None:
        metadata["version"] = doc.ref.version

    return metadata


@pytest.fixture
def documents_repository_stub(monkeypatch) -> ObjectStoreDocumentsRepository:
    from ai_core import services, views

    repository = ObjectStoreDocumentsRepository()
    monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", repository, raising=False)
    monkeypatch.setattr(services, "_DOCUMENTS_REPOSITORY", None, raising=False)
    try:
        yield repository
    finally:
        monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", None, raising=False)


@pytest.fixture(autouse=True)
def _auto_documents_repository(documents_repository_stub):
    yield
