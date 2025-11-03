import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from types import SimpleNamespace

from ai_core.ingestion import process_document
from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_ingestion_idempotency_skips_unchanged_documents(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
):
    tenant = test_tenant_schema_name
    case = "case-idempotent"
    external_id = "demo-hello-1759389009"

    store_path = tmp_path / "object-store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)
    monkeypatch.setattr(rate_limit, "check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    def upload_document(content: str) -> tuple[str, str]:
        upload = SimpleUploadedFile(
            "hello.txt", content.encode("utf-8"), content_type="text/plain"
        )
        payload = {
            "file": upload,
            "metadata": json.dumps({"external_id": external_id}),
        }
        response = client.post(
            "/ai/rag/documents/upload/",
            data=payload,
            **{
                META_TENANT_SCHEMA_KEY: tenant,
                META_TENANT_ID_KEY: tenant,
                META_CASE_ID_KEY: case,
            },
        )
        assert response.status_code == 202
        body = response.json()
        document_id = body["document_id"]
        trace_id = body["trace_id"]
        tenant_segment = object_store.sanitize_identifier(tenant)
        case_segment = object_store.sanitize_identifier(case)
        metadata_path = Path(
            store_path,
            tenant_segment,
            case_segment,
            "uploads",
            f"{document_id}.meta.json",
        )
        stored_metadata = json.loads(metadata_path.read_text())
        assert stored_metadata["external_id"] == external_id
        return document_id, trace_id

    first_doc, first_trace = upload_document("Hello RAG ingestion!")
    first_result = process_document(
        tenant,
        case,
        first_doc,
        "standard",
        tenant_schema=tenant,
        trace_id=first_trace,
    )

    assert first_result["external_id"] == external_id
    assert first_result["inserted"] == 1
    assert first_result["skipped"] == 0
    assert first_result["replaced"] == 0
    assert first_result["action"] == "inserted"
    assert first_result["written"] == 1
    assert first_result["embedding_profile"] == "standard"

    second_doc, second_trace = upload_document("Hello RAG ingestion!")
    second_result = process_document(
        tenant,
        case,
        second_doc,
        "standard",
        tenant_schema=tenant,
        trace_id=second_trace,
    )

    assert second_result["external_id"] == external_id
    assert second_result["inserted"] == 0
    # Ingestion may emit a dedicated near-duplicate action, which still counts as one skipped document.
    assert (
        second_result["skipped"]
        + int(second_result["action"] == "near_duplicate_skipped")
        == 1
    )
    assert second_result["replaced"] == 0
    assert second_result["action"] in {"skipped", "near_duplicate_skipped"}
    assert second_result["written"] == 0

    third_doc, third_trace = upload_document("Hello RAG ingestion version two!")
    third_result = process_document(
        tenant,
        case,
        third_doc,
        "standard",
        tenant_schema=tenant,
        trace_id=third_trace,
    )

    assert third_result["external_id"] == external_id
    assert third_result["skipped"] == 0
    assert third_result["action"] in {"inserted", "replaced"}
    assert third_result["inserted"] == 1 or third_result["replaced"] == 1
    assert third_result["written"] == 1


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_ingestion_concurrent_same_external_id_is_idempotent(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
):
    tenant = test_tenant_schema_name
    case = "case-race"
    external_id = "race-hello-external-id"
    content = "Concurrent hello!"

    store_path = tmp_path / "object-store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)
    monkeypatch.setattr(rate_limit, "check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    def upload_document() -> tuple[str, str]:
        upload = SimpleUploadedFile(
            "hello.txt", content.encode("utf-8"), content_type="text/plain"
        )
        payload = {
            "file": upload,
            "metadata": json.dumps({"external_id": external_id}),
        }
        response = client.post(
            "/ai/rag/documents/upload/",
            data=payload,
            **{
                META_TENANT_SCHEMA_KEY: tenant,
                META_TENANT_ID_KEY: tenant,
                META_CASE_ID_KEY: case,
            },
        )
        assert response.status_code == 202
        body = response.json()
        document_id = body["document_id"]
        trace_id = body["trace_id"]
        tenant_segment = object_store.sanitize_identifier(tenant)
        case_segment = object_store.sanitize_identifier(case)
        metadata_path = Path(
            store_path,
            tenant_segment,
            case_segment,
            "uploads",
            f"{document_id}.meta.json",
        )
        stored_metadata = json.loads(metadata_path.read_text())
        assert stored_metadata["external_id"] == external_id
        return document_id, trace_id

    doc_a, trace_a = upload_document()
    doc_b, trace_b = upload_document()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            process_document,
            tenant,
            case,
            doc_a,
            "standard",
            tenant_schema=tenant,
            trace_id=trace_a,
        )
        future_b = executor.submit(
            process_document,
            tenant,
            case,
            doc_b,
            "standard",
            tenant_schema=tenant,
            trace_id=trace_b,
        )
        res_a = future_a.result()
        res_b = future_b.result()

    assert res_a["external_id"] == external_id
    assert res_b["external_id"] == external_id

    actions = {res_a["action"], res_b["action"]}
    assert actions == {"inserted", "skipped"}

    written_values = {res_a["written"], res_b["written"]}
    assert written_values == {0, 1}

    assert res_a["written"] + res_b["written"] == 1
