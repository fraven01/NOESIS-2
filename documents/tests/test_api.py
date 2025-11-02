"""Unit tests for public functions in :mod:`documents.api`."""

from __future__ import annotations

import base64
import hashlib

import pytest

from common.object_store import FilesystemObjectStore
from documents.api import normalize_from_raw
from documents.contracts import NormalizedDocumentInputV1


def test_normalize_from_raw_accepts_payload_bytes() -> None:
    payload = "Grüße aus Köln".encode("utf-16-le")

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "payload_encoding": "utf-16-le",
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload
    assert result.checksum == hashlib.sha256(payload).hexdigest()


def test_normalize_from_raw_accepts_payload_base64() -> None:
    payload = "Plain content".encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_base64": encoded,
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.primary_text == "Plain content"
    assert result.payload_bytes == payload


def test_normalize_from_raw_accepts_payload_path(tmp_path) -> None:
    payload = b"Binary via path"
    relative_path = "tenant-x/case-default/crawler/raw/doc.bin"

    store = FilesystemObjectStore(lambda: tmp_path)
    target = store.BASE_PATH / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_path": relative_path,
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
        object_store=store,
    )

    assert result.primary_text == "Binary via path"
    assert result.payload_bytes == payload


def test_normalize_from_raw_rejects_payload_path_outside_store(tmp_path) -> None:
    store = FilesystemObjectStore(lambda: tmp_path / "store")
    (tmp_path / "secret.bin").write_bytes(b"shh")

    with pytest.raises(ValueError):
        document_input = NormalizedDocumentInputV1(
            raw_reference={
                "payload_path": "../secret.bin",
                "metadata": {
                    "provider": "crawler",
                    "origin_uri": "https://example.com",
                },
            },
            tenant_id="tenant-x",
        )
        normalize_from_raw(
            document_input=document_input,
            object_store=store,
        )


def test_normalize_from_raw_rejects_absolute_payload_path(tmp_path) -> None:
    absolute_path = tmp_path / "secret.bin"
    absolute_path.write_text("hidden", encoding="utf-8")
    store = FilesystemObjectStore(lambda: tmp_path / "store")

    with pytest.raises(ValueError):
        document_input = NormalizedDocumentInputV1(
            raw_reference={
                "payload_path": str(absolute_path),
                "metadata": {
                    "provider": "crawler",
                    "origin_uri": "https://example.com",
                },
            },
            tenant_id="tenant-x",
        )
        normalize_from_raw(
            document_input=document_input,
            object_store=store,
        )


def test_normalize_from_raw_uses_charset_from_content_type_metadata() -> None:
    payload = "Grüße aus Köln".encode("iso-8859-1")

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "metadata": {
                "provider": "crawler",
                "origin_uri": "https://example.com",
                "content_type": "text/html; charset=iso-8859-1",
            },
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload


def test_normalize_from_raw_requires_payload() -> None:
    with pytest.raises(ValueError):
        NormalizedDocumentInputV1(raw_reference={}, tenant_id="tenant-x")


def test_normalize_from_raw_uses_metadata_source_and_workflow() -> None:
    payload = b"Metadata overrides"

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "metadata": {
                "provider": "upload",
                "origin_uri": "https://example.com/from-upload",
                "workflow_id": "upload-flow",
                "source": "upload",
            },
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.document.ref.workflow_id == "upload-flow"
    assert result.document.source == "upload"
    assert result.metadata["workflow_id"] == "upload-flow"
    assert result.metadata["source"] == "upload"


def test_normalize_from_raw_prefers_explicit_parameters() -> None:
    payload = b"Parameter overrides"

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "metadata": {
                "provider": "integration",
                "origin_uri": "https://example.com/from-integration",
                "workflow_id": "meta-flow",
                "source": "crawler",
            },
        },
        tenant_id="tenant-x",
        workflow_id="param-flow",
        source="integration",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.document.ref.workflow_id == "param-flow"
    assert result.document.source == "integration"
    assert result.metadata["workflow_id"] == "param-flow"
    assert result.metadata["source"] == "integration"


def test_normalize_from_raw_applies_provider_default_for_crawler() -> None:
    payload = b"Default semantics"

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "metadata": {"origin_uri": "https://example.com/default"},
        },
        tenant_id="tenant-x",
    )

    result = normalize_from_raw(
        document_input=document_input,
    )

    assert result.document.source == "crawler"
    assert result.metadata["source"] == "crawler"
    assert result.metadata["provider"] == "web"
    external_ref = result.document.meta.external_ref or {}
    assert external_ref.get("provider") == "web"


def test_normalized_document_input_v1_defaults() -> None:
    payload = b"Defaults"

    document_input = NormalizedDocumentInputV1(
        raw_reference={"payload_bytes": payload, "metadata": {}},
        tenant_id="tenant-x",
    )

    assert document_input.workflow_id == "crawler"
    assert document_input.source == "crawler"
    assert document_input.provider == "web"
    assert document_input.external_id.startswith("web:")


def test_normalized_document_input_v1_validates_source() -> None:
    payload = b"invalid source"

    with pytest.raises(ValueError):
        NormalizedDocumentInputV1(
            raw_reference={
                "payload_bytes": payload,
                "metadata": {"source": "unknown", "provider": "custom"},
            },
            tenant_id="tenant-x",
        )


def test_normalized_document_input_v1_allows_explicit_overrides() -> None:
    payload = b"overrides"

    document_input = NormalizedDocumentInputV1(
        raw_reference={
            "payload_bytes": payload,
            "metadata": {
                "provider": "crawler",
                "external_id": "provider:123",
                "workflow_id": "from-meta",
            },
        },
        tenant_id="tenant-x",
        workflow_id="explicit-flow",
        source="upload",
        provider="explicit",
        external_id="explicit:321",
    )

    assert document_input.workflow_id == "explicit-flow"
    assert document_input.source == "upload"
    assert document_input.provider == "explicit"
    assert document_input.external_id == "explicit:321"
