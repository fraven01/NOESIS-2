from datetime import datetime, timezone as dt_timezone
import json
from pathlib import Path
import uuid
from types import SimpleNamespace

import pytest
from django.utils import timezone

from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_COLLECTION_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_CASE_ID_KEY,
)


@pytest.mark.django_db
def test_rag_ingestion_run_queues_task(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    # create_case("case-123")  # Removed for caseless test
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    captured = {}
    document_id = uuid.uuid4()

    def fake_delay(
        tenant_id,
        case_id,
        document_ids,
        embedding_profile,
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
                "embedding_profile": embedding_profile,
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
        data={
            "document_ids": [str(document_id)],
            "priority": "high",
            "embedding_profile": "standard",
        },
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
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
        "case_id": "upload",
        "document_ids": [str(document_id)],
        "embedding_profile": "standard",
        "trace_id": body["trace_id"],
        "run_id": body["ingestion_run_id"],
        "idempotency_key": None,
        "tenant_schema": test_tenant_schema_name,
    }


@pytest.mark.django_db
def test_rag_ingestion_run_persists_collection_header_scope(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    # create_case("case-collection-run")
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    tenant_segment = object_store.sanitize_identifier(test_tenant_schema_name)
    case_segment = object_store.sanitize_identifier("upload")
    document_id = str(uuid.uuid4())

    meta_path = (
        Path(tmp_path)
        / tenant_segment
        / case_segment
        / "uploads"
        / f"{document_id}.meta.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"source": "ingestion"}))

    collection_scope = str(uuid.uuid4())

    def _assert_scope_persisted(
        tenant_id,
        case_id,
        document_ids,
        embedding_profile,
        *,
        run_id,
        trace_id=None,
        idempotency_key=None,
        tenant_schema=None,
    ):
        assert list(document_ids) == [document_id]
        persisted = json.loads(meta_path.read_text())
        assert persisted["collection_id"] == collection_scope

    monkeypatch.setattr(
        "ai_core.views.run_ingestion",
        SimpleNamespace(delay=_assert_scope_persisted),
    )

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={
            "document_ids": [document_id],
            "embedding_profile": "standard",
        },
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
            META_COLLECTION_ID_KEY: collection_scope,
        },
    )

    assert response.status_code == 202
    stored_metadata = json.loads(meta_path.read_text())
    assert stored_metadata["collection_id"] == collection_scope
    assert stored_metadata["source"] == "ingestion"


@pytest.mark.django_db
def test_rag_ingestion_run_with_empty_document_ids_returns_400(
    client, monkeypatch, test_tenant_schema_name
):
    # create_case("case-123")
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={"document_ids": [], "embedding_profile": "standard"},
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "document_ids" in body["detail"]


@pytest.mark.django_db
def test_rag_ingestion_run_without_profile_returns_400(
    client, monkeypatch, test_tenant_schema_name
):
    # create_case("case-123")
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    document_id = str(uuid.uuid4())

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={"document_ids": [document_id]},
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "embedding_profile" in body["detail"]


@pytest.mark.django_db
def test_rag_ingestion_run_with_invalid_priority_returns_400(
    client, monkeypatch, test_tenant_schema_name
):
    # create_case("case-123")
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    document_id = str(uuid.uuid4())

    response = client.post(
        "/ai/rag/ingestion/run/",
        data={
            "document_ids": [document_id],
            "priority": "urgent",
            "embedding_profile": "standard",
        },
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "priority" in body["detail"]


@pytest.mark.django_db
def test_rag_ingestion_run_with_invalid_document_id_returns_400(
    client, monkeypatch, test_tenant_schema_name
):
    # create_case("case-uuid-check")
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps(
            {
                "document_ids": ["not-a-uuid"],
                "embedding_profile": "standard",
            }
        ),
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "upload",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "document_ids" in body["detail"]
