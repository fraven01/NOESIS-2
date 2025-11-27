from __future__ import annotations

import uuid

import pytest

from ai_core.rag.collections import (
    MANUAL_COLLECTION_LABEL,
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
    ensure_manual_collection_model,
    manual_collection_uuid,
)


def test_manual_collection_uuid_is_deterministic():
    tenant_id = "Tenant-Example"
    first = manual_collection_uuid(tenant_id)
    second = manual_collection_uuid(tenant_id.lower())

    assert isinstance(first, uuid.UUID)
    assert first == second


def test_ensure_manual_collection_routes_through_domain_service(
    monkeypatch: pytest.MonkeyPatch,
):
    tenant_id = "tenant-seed"
    collection_uuid = manual_collection_uuid(tenant_id)
    dummy_client = object()

    captured: dict[str, object] = {}

    class _DummyCollection:
        def __init__(self, collection_id):
            self.collection_id = collection_id

    class _DummyService:
        def __init__(self, *, vector_store):
            captured["vector_store"] = vector_store

        def ensure_collection(self, **kwargs):  # type: ignore[override]
            captured["ensure_kwargs"] = kwargs
            return _DummyCollection(kwargs["collection_id"])

    monkeypatch.setattr("ai_core.rag.collections.DocumentDomainService", _DummyService)
    monkeypatch.setattr("ai_core.rag.collections.get_default_client", lambda: dummy_client)
    monkeypatch.setattr("ai_core.rag.collections._resolve_tenant", lambda value: "tenant")

    collection_id = ensure_manual_collection(tenant_id)

    assert collection_id == str(collection_uuid)
    assert captured["vector_store"] is dummy_client
    ensure_kwargs = captured.get("ensure_kwargs")
    assert ensure_kwargs
    assert ensure_kwargs["collection_id"] == collection_uuid
    assert ensure_kwargs["key"] == MANUAL_COLLECTION_SLUG
    assert ensure_kwargs["name"] == MANUAL_COLLECTION_LABEL


def test_ensure_manual_collection_model_creates_document_collection(
    monkeypatch: pytest.MonkeyPatch,
):
    tenant = object()
    collection_uuid = uuid.uuid4()
    captured: dict[str, object] = {}

    class _DummyManager:
        def get_or_create(self, *, tenant, key, defaults):  # type: ignore[override]
            captured["tenant"] = tenant
            captured["key"] = key
            captured["defaults"] = defaults
            return "document-collection", True

    class _DummyDocumentCollection:
        objects = _DummyManager()

    monkeypatch.setattr(
        "ai_core.rag.collections._get_document_collection_model",
        lambda: _DummyDocumentCollection,
    )
    monkeypatch.setattr(
        "ai_core.rag.collections._resolve_tenant", lambda value: tenant
    )

    result = ensure_manual_collection_model(
        tenant_id="tenant",
        collection_uuid=collection_uuid,
        slug="custom-slug",
        label="Custom Label",
    )

    assert result == "document-collection"
    assert captured["tenant"] is tenant
    assert captured["key"] == "custom-slug"
    assert captured["defaults"] == {
        "id": collection_uuid,
        "collection_id": collection_uuid,
        "name": "Custom Label",
        "type": "",
        "visibility": "",
        "metadata": {},
    }
