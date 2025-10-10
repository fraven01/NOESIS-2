import pytest

from ai_core.ingestion_status import (
    mark_ingestion_run_completed,
    mark_ingestion_run_running,
    record_ingestion_run_queued,
)
from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


@pytest.mark.django_db
def test_rag_ingestion_status_returns_latest_run(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    tenant = test_tenant_schema_name
    case = "case-status"
    monkeypatch.setattr(rate_limit, "check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    run_id = "run-123"
    record_ingestion_run_queued(
        tenant,
        case,
        run_id,
        ["doc-1", "doc-2"],
        queued_at="2024-01-01T00:00:00Z",
        trace_id="trace-abc",
        embedding_profile="standard",
        source="manual",
        invalid_document_ids=["missing-doc"],
    )
    mark_ingestion_run_running(
        tenant,
        case,
        run_id,
        started_at="2024-01-01T00:01:00Z",
    )
    mark_ingestion_run_completed(
        tenant,
        case,
        run_id,
        finished_at="2024-01-01T00:02:00Z",
        duration_ms=120.5,
        inserted_documents=2,
        replaced_documents=0,
        skipped_documents=0,
        inserted_chunks=6,
        invalid_document_ids=["missing-doc"],
        document_ids=["doc-1", "doc-2"],
        error=None,
    )

    response = client.get(
        "/ai/rag/ingestion/status/",
        **{
            META_TENANT_SCHEMA_KEY: tenant,
            META_TENANT_ID_KEY: tenant,
            META_CASE_ID_KEY: case,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["status"] == "succeeded"
    assert payload["inserted_documents"] == 2
    assert payload["invalid_document_ids"] == ["missing-doc"]
    assert payload["document_ids"] == ["doc-1", "doc-2"]


@pytest.mark.django_db
def test_rag_ingestion_status_returns_404_when_empty(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    response = client.get(
        "/ai/rag/ingestion/status/",
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-none",
        },
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "ingestion_status_not_found"
