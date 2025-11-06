from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from psycopg2 import sql

from ai_core.rag.collections import (
    MANUAL_COLLECTION_LABEL,
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
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
