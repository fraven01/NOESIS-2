"""Unit tests for public functions in :mod:`documents.api`."""

from __future__ import annotations

import base64
import hashlib

import pytest
from pydantic import ValidationError

from common.object_store import FilesystemObjectStore
from documents.api import normalize_from_raw
from documents.contracts import DocumentBlobDescriptorV1, NormalizedDocumentInputV1


def _metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "provider": "crawler",
        "origin_uri": "https://example.com",
        "content_type": "text/plain",
    }
    base.update(overrides)
    return base


def test_normalize_from_raw_accepts_payload_bytes() -> None:
    payload = "Grüße aus Köln".encode("utf-16-le")

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(payload_encoding="utf-16-le"),
        blob=DocumentBlobDescriptorV1(payload_bytes=payload),
    )

    result = normalize_from_raw(contract=contract)

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload
    assert result.checksum == hashlib.sha256(payload).hexdigest()


def test_normalize_from_raw_accepts_payload_base64() -> None:
    payload = "Plain content".encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(),
        blob=DocumentBlobDescriptorV1(payload_base64=encoded),
    )

    result = normalize_from_raw(contract=contract)

    assert result.primary_text == "Plain content"
    assert result.payload_bytes == payload


def test_normalize_from_raw_rejects_string_payload_bytes() -> None:
    with pytest.raises(ValidationError):
        NormalizedDocumentInputV1(
            tenant_id="tenant-x",
            metadata=_metadata(origin_uri="https://example.com/base64"),
            blob=DocumentBlobDescriptorV1(payload_bytes="VGVzdCBjb250ZW50"),
        )


def test_normalize_from_raw_accepts_payload_path(tmp_path) -> None:
    payload = b"Binary via path"
    relative_path = "tenant-x/case-default/crawler/raw/doc.bin"

    store = FilesystemObjectStore(lambda: tmp_path)
    target = store.BASE_PATH / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(),
        blob=DocumentBlobDescriptorV1(object_store_path=relative_path),
    )

    result = normalize_from_raw(contract=contract, object_store=store)

    assert result.primary_text == "Binary via path"
    assert result.payload_bytes == payload


def test_normalize_from_raw_rejects_payload_path_outside_store(tmp_path) -> None:
    store = FilesystemObjectStore(lambda: tmp_path / "store")
    (tmp_path / "secret.bin").write_bytes(b"shh")

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(),
        blob=DocumentBlobDescriptorV1(object_store_path="../secret.bin"),
    )

    with pytest.raises(ValueError):
        normalize_from_raw(contract=contract, object_store=store)


def test_normalize_from_raw_rejects_absolute_payload_path(tmp_path) -> None:
    absolute_path = tmp_path / "secret.bin"
    absolute_path.write_text("hidden", encoding="utf-8")
    store = FilesystemObjectStore(lambda: tmp_path / "store")

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(),
        blob=DocumentBlobDescriptorV1(object_store_path=str(absolute_path)),
    )

    with pytest.raises(ValueError):
        normalize_from_raw(contract=contract, object_store=store)


def test_normalize_from_raw_uses_charset_from_content_type_metadata() -> None:
    payload = "Grüße aus Köln".encode("iso-8859-1")

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(content_type="text/html; charset=iso-8859-1"),
        blob=DocumentBlobDescriptorV1(payload_bytes=payload),
    )

    result = normalize_from_raw(contract=contract)

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload


def test_normalize_from_raw_requires_payload() -> None:
    with pytest.raises(ValidationError):
        NormalizedDocumentInputV1(
            tenant_id="tenant-x",
            metadata=_metadata(),
            blob=DocumentBlobDescriptorV1(),
        )


def test_normalize_from_raw_uses_metadata_source_and_workflow() -> None:
    payload = b"Metadata overrides"

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(
            provider="upload",
            origin_uri="https://example.com/from-upload",
            workflow_id="upload-flow",
            source="upload",
        ),
        workflow_id="upload-flow",
        source="upload",
        blob=DocumentBlobDescriptorV1(payload_bytes=payload),
    )

    result = normalize_from_raw(contract=contract)

    assert result.document.ref.workflow_id == "upload-flow"
    assert result.document.source == "upload"
    assert result.metadata["workflow_id"] == "upload-flow"
    assert result.metadata["source"] == "upload"


def test_normalize_from_raw_prefers_explicit_parameters() -> None:
    payload = b"Parameter overrides"

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(
            provider="integration",
            origin_uri="https://example.com/from-integration",
            workflow_id="meta-flow",
            source="crawler",
        ),
        workflow_id="param-flow",
        source="integration",
        blob=DocumentBlobDescriptorV1(payload_bytes=payload),
    )

    result = normalize_from_raw(contract=contract)

    assert result.document.ref.workflow_id == "param-flow"
    assert result.document.source == "integration"


def test_normalized_document_input_requires_tenant_identifier() -> None:
    with pytest.raises(ValidationError):
        NormalizedDocumentInputV1(
            tenant_id="",  # type: ignore[arg-type]
            metadata=_metadata(),
            blob=DocumentBlobDescriptorV1(payload_bytes=b"content"),
        )


def test_document_blob_descriptor_rejects_conflicting_sources() -> None:
    with pytest.raises(ValidationError) as excinfo:
        NormalizedDocumentInputV1(
            tenant_id="tenant-x",
            metadata=_metadata(),
            blob=DocumentBlobDescriptorV1(
                payload_bytes=b"binary",
                inline_text="conflict",
            ),
        )

    assert "blob_source_ambiguous" in str(excinfo.value)


def test_normalized_document_input_normalizes_media_type_from_metadata() -> None:
    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata=_metadata(content_type="TEXT/HTML; charset=utf-8"),
        blob=DocumentBlobDescriptorV1(payload_bytes=b"<html />"),
    )

    assert contract.media_type == "text/html"


def test_normalize_from_raw_applies_provider_default_for_crawler() -> None:
    payload = b"Default semantics"

    contract = NormalizedDocumentInputV1(
        tenant_id="tenant-x",
        metadata={"origin_uri": "https://example.com/default"},
        blob=DocumentBlobDescriptorV1(payload_bytes=payload),
    )

    result = normalize_from_raw(contract=contract)

    assert result.document.source == "crawler"
    assert result.metadata["source"] == "crawler"
    assert result.metadata["provider"] == "web"
    external_ref = result.document.meta.external_ref or {}
    assert external_ref.get("provider") == "web"
