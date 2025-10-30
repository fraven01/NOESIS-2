"""Unit tests for public functions in :mod:`documents.api`."""

from __future__ import annotations

import base64
import hashlib

import pytest

from ai_core.infra import object_store
from documents.api import normalize_from_raw


def test_normalize_from_raw_accepts_payload_bytes() -> None:
    payload = "Grüße aus Köln".encode("utf-16-le")

    result = normalize_from_raw(
        raw_reference={
            "payload_bytes": payload,
            "payload_encoding": "utf-16-le",
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload
    assert result.checksum == hashlib.sha256(payload).hexdigest()


def test_normalize_from_raw_accepts_payload_base64() -> None:
    payload = "Plain content".encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")

    result = normalize_from_raw(
        raw_reference={
            "payload_base64": encoded,
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    assert result.primary_text == "Plain content"
    assert result.payload_bytes == payload


def test_normalize_from_raw_accepts_payload_path(tmp_path, monkeypatch) -> None:
    payload = b"Binary via path"
    relative_path = "tenant-x/case-default/crawler/raw/doc.bin"

    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    target = object_store.BASE_PATH / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    result = normalize_from_raw(
        raw_reference={
            "payload_path": relative_path,
            "metadata": {"provider": "crawler", "origin_uri": "https://example.com"},
        },
        tenant_id="tenant-x",
    )

    assert result.primary_text == "Binary via path"
    assert result.payload_bytes == payload


def test_normalize_from_raw_rejects_payload_path_outside_store(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path / "store")
    (tmp_path / "secret.bin").write_bytes(b"shh")

    with pytest.raises(ValueError):
        normalize_from_raw(
            raw_reference={
                "payload_path": "../secret.bin",
                "metadata": {
                    "provider": "crawler",
                    "origin_uri": "https://example.com",
                },
            },
            tenant_id="tenant-x",
        )


def test_normalize_from_raw_rejects_absolute_payload_path(
    tmp_path, monkeypatch
) -> None:
    absolute_path = tmp_path / "secret.bin"
    absolute_path.write_text("hidden", encoding="utf-8")
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path / "store")

    with pytest.raises(ValueError):
        normalize_from_raw(
            raw_reference={
                "payload_path": str(absolute_path),
                "metadata": {
                    "provider": "crawler",
                    "origin_uri": "https://example.com",
                },
            },
            tenant_id="tenant-x",
        )


def test_normalize_from_raw_uses_charset_from_content_type_metadata() -> None:
    payload = "Grüße aus Köln".encode("iso-8859-1")

    result = normalize_from_raw(
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

    assert result.primary_text == "Grüße aus Köln"
    assert result.payload_bytes == payload


def test_normalize_from_raw_requires_payload() -> None:
    with pytest.raises(ValueError):
        normalize_from_raw(raw_reference={}, tenant_id="tenant-x")
