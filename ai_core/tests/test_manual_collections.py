from __future__ import annotations

import pytest

from ai_core.rag.collections import (
    ensure_manual_collection,
    ensure_manual_collection_model,
    manual_collection_uuid,
)
from customers.models import Tenant
from uuid import UUID, uuid4


def test_manual_collection_uuid_is_deterministic():
    tenant_id = uuid4()
    tenant = Tenant(id=tenant_id, schema_name="tenant-example")
    first = manual_collection_uuid(tenant)
    second = manual_collection_uuid(tenant)

    assert isinstance(first, UUID)
    assert first == second


def test_ensure_manual_collection_routes_through_collection_service(
    monkeypatch: pytest.MonkeyPatch,
):
    tenant = Tenant(id=uuid4(), schema_name="tenant-seed")
    dummy_client = object()

    captured: dict[str, object] = {}

    class _DummyService:
        def __init__(self, *, vector_client=None):
            captured["vector_client"] = vector_client

        def ensure_manual_collection(self, **kwargs):  # type: ignore[override]
            captured["ensure_kwargs"] = kwargs
            return "collection-id"

    monkeypatch.setattr("ai_core.rag.collections.CollectionService", _DummyService)

    with pytest.deprecated_call():
        collection_id = ensure_manual_collection(
            tenant,
            slug="custom-slug",
            label="Custom Label",
            client=dummy_client,
        )

    assert collection_id == "collection-id"
    assert captured["vector_client"] is dummy_client
    assert captured["ensure_kwargs"] == {
        "tenant": tenant,
        "slug": "custom-slug",
        "label": "Custom Label",
    }


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
    monkeypatch.setattr("ai_core.rag.collections._resolve_tenant", lambda value: tenant)

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
