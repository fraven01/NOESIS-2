from datetime import datetime, timezone as dt_timezone
from types import SimpleNamespace

import pytest
from django.utils import timezone

from ai_core.infra import rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


@pytest.mark.django_db
def test_rag_ingestion_run_queues_task(client, monkeypatch, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    captured = {}

    def fake_delay(
        tenant_id,
        case_id,
        document_ids,
        *,
        run_id,
        trace_id=None,
        idempotency_key=None,
        tenant_schema=None,
    ):
        captured.update(
            {
                "tenant_id": tenant_id,
                "case_id": case_id,
                "document_ids": document_ids,
                "run_id": run_id,
                "trace_id": trace_id,
                "idempotency_key": idempotency_key,
                "tenant_schema": tenant_schema,
            }
        )

    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=fake_delay)
    )

    fixed_now = datetime(2024, 1, 1, 12, 0, tzinfo=dt_timezone.utc)
    monkeypatch.setattr(timezone, "now", lambda: fixed_now)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={"document_ids": ["abc123"], "priority": "high"},
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-123",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["queued_at"] == fixed_now.isoformat()
    assert body["trace_id"]
    assert body["idempotent"] is False
    assert body["ingestion_run_id"]

    assert captured == {
        "tenant_id": test_tenant_schema_name,
        "case_id": "case-123",
        "document_ids": ["abc123"],
        "trace_id": body["trace_id"],
        "run_id": body["ingestion_run_id"],
        "idempotency_key": None,
        "tenant_schema": test_tenant_schema_name,
    }


@pytest.mark.django_db
def test_rag_ingestion_run_with_empty_document_ids_returns_400(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={"document_ids": []},
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-123",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "document_ids must be a non-empty list."
    assert body["code"] == "invalid_document_ids"
