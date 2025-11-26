from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from psycopg2 import sql

from ai_core.rag.collections import (
    MANUAL_COLLECTION_LABEL,
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
    ensure_manual_collection_model,
    manual_collection_uuid,
)


class _CursorContext:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self._cursor

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyCursor:
    def __init__(self):
        self.executed: list[tuple[object, tuple]] = []

    def execute(self, statement, params):
        self.executed.append((statement, params))


class _DummyConnection:
    def __init__(self):
        self.cursor_obj = _DummyCursor()
        self.committed = False

    def cursor(self):
        return _CursorContext(self.cursor_obj)

    def commit(self):
        self.committed = True


class _DummyClient:
    def __init__(self):
        self.conn = _DummyConnection()

    @contextmanager
    def connection(self):
        yield self.conn

    def _table(self, name: str):
        return sql.Identifier("rag", name)


def test_manual_collection_uuid_is_deterministic():
    tenant_id = "Tenant-Example"
    first = manual_collection_uuid(tenant_id)
    second = manual_collection_uuid(tenant_id.lower())

    assert isinstance(first, uuid.UUID)
    assert first == second


def test_ensure_manual_collection_inserts_record(monkeypatch: pytest.MonkeyPatch):
    dummy_client = _DummyClient()
    monkeypatch.setattr(
        "ai_core.rag.collections.get_default_client", lambda: dummy_client
    )
    monkeypatch.setattr(
        "ai_core.rag.collections.ensure_manual_collection_model",
        lambda **_: None,
    )

    tenant_id = "tenant-seed"
    expected_collection_id = str(manual_collection_uuid(tenant_id))

    collection_id = ensure_manual_collection(tenant_id)

    assert collection_id == expected_collection_id
    assert dummy_client.conn.committed is True
    assert dummy_client.conn.cursor_obj.executed, "SQL statement should be executed"

    _, params = dummy_client.conn.cursor_obj.executed[0]
    expected_tenant_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"tenant:{tenant_id.lower()}")
    assert params[0] == str(expected_tenant_uuid)
    assert params[1] == expected_collection_id
    assert params[2] == MANUAL_COLLECTION_SLUG
    assert params[3] == MANUAL_COLLECTION_LABEL


def test_ensure_manual_collection_dual_writes(monkeypatch: pytest.MonkeyPatch):
    dummy_client = _DummyClient()
    monkeypatch.setattr(
        "ai_core.rag.collections.get_default_client", lambda: dummy_client
    )

    tenant_id = "tenant-seed"
    collection_uuid = manual_collection_uuid(tenant_id)

    calls: dict[str, object] = {}

    def _capture_model_write(**kwargs):
        calls.update(kwargs)
        return "collection-model"

    monkeypatch.setattr(
        "ai_core.rag.collections.ensure_manual_collection_model",
        _capture_model_write,
    )

    collection_id = ensure_manual_collection(tenant_id)

    assert collection_id == str(collection_uuid)
    assert calls["tenant_id"] == tenant_id
    assert calls["collection_uuid"] == collection_uuid
    assert calls["slug"] == MANUAL_COLLECTION_SLUG
    assert calls["label"] == MANUAL_COLLECTION_LABEL


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
