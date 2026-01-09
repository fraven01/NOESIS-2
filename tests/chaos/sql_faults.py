"""Chaos tests for Cloud SQL fault injection scenarios."""

from __future__ import annotations

import json
import os
from typing import Iterable

import pytest
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.utils import OperationalError

from ai_core.graphs.technical import info_intake
from ai_core.infra import object_store, rate_limit
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)

pytestmark = pytest.mark.chaos


def _sql_down() -> bool:
    """Return ``True`` when the SQL_DOWN chaos flag is enabled."""

    return os.getenv("SQL_DOWN", "0").lower() in {"1", "true", "yes"}


_SQL_FAULT_STATE: dict[str, bool] = {}


@pytest.fixture(autouse=True)
def sql_faults(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterable[None]:
    """Inject Cloud SQL failures on the first cursor acquisition when armed."""

    original_cursor = BaseDatabaseWrapper.cursor
    node_id = request.node.nodeid

    def _chaos_cursor(self, *args, **kwargs):  # type: ignore[override]
        if _sql_down():
            if not _SQL_FAULT_STATE.get(node_id):
                _SQL_FAULT_STATE[node_id] = True
                raise OperationalError("sql chaos: connection dropped")
        else:
            _SQL_FAULT_STATE.pop(node_id, None)
        return original_cursor(self, *args, **kwargs)

    monkeypatch.setattr(BaseDatabaseWrapper, "cursor", _chaos_cursor, raising=False)
    try:
        yield
    finally:
        _SQL_FAULT_STATE.pop(node_id, None)


@pytest.mark.django_db
def test_health_read_path_fallback(
    client,
    chaos_env,
    monkeypatch,
    test_tenant_schema_name,
):
    """The ``/health/`` endpoint surfaces SQL outages without crashing."""

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    chaos_env.set_sql_down(True)

    headers = {
        X_TENANT_SCHEMA_HEADER: test_tenant_schema_name,
        X_TENANT_ID_HEADER: test_tenant_schema_name,
        X_CASE_ID_HEADER: "chaos-health",
        X_TRACE_ID_HEADER: "trace-chaos-health",
    }

    response = client.get("/health/", **headers)

    assert response.status_code == 503
    payload = response.json()
    assert payload.get("status") != "ok"
    services = payload.get("services", {})
    for required in ("web", "worker", "redis"):
        assert isinstance(services.get(required), str)
    sql_status = services.get("sql") or services.get("database")
    assert isinstance(sql_status, str)
    assert "Traceback" not in response.content.decode()


@pytest.mark.django_db
def test_write_path_idempotency_retry(
    client,
    chaos_env,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
):
    """POST retries remain idempotent when the first attempt hits SQL errors."""

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    call_counter = {"count": 0}
    original_run = info_intake.run

    def _wrapped_run(state, meta):  # type: ignore[override]
        call_counter["count"] += 1
        return original_run(state, meta)

    monkeypatch.setattr(info_intake, "run", _wrapped_run)

    chaos_env.set_sql_down(True)

    tenant = test_tenant_schema_name
    idempotency_key = "chaos-sql-write-001"
    headers = {
        X_TENANT_ID_HEADER: tenant,
        X_TENANT_SCHEMA_HEADER: tenant,
        X_CASE_ID_HEADER: "chaos-case",
        X_TRACE_ID_HEADER: "trace-chaos-write",
        IDEMPOTENCY_KEY_HEADER: idempotency_key,
    }
    payload = {"hello": "world"}

    first = client.post(
        "/ai/intake/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert first.status_code == 503
    assert "Traceback" not in first.content.decode()

    chaos_env.set_sql_down(False)

    retry = client.post(
        "/ai/intake/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert retry.status_code == 200
    body = retry.json()
    assert body.get("tenant") == tenant
    assert body.get("idempotent") is True
    assert call_counter["count"] == 1

    third = client.post(
        "/ai/intake/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert third.status_code == 200
    assert third.json() == body
