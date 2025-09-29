import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from ai_core.ingestion import process_document
from ai_core.infra import object_store, rate_limit
from ai_core.views import make_fallback_external_id


@pytest.mark.django_db
def test_upload_ingest_query_end2end(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    rag_database,
):
    tenant = test_tenant_schema_name
    case = "case-e2e"

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    # Upload
    upload = SimpleUploadedFile(
        "note.txt", b"hello ZEBRAGURKE world", content_type="text/plain"
    )
    payload = {
        "file": upload,
        "metadata": json.dumps({"label": "e2e", "external_id": "doc-e2e"}),
    }

    resp = client.post(
        "/ai/rag/documents/upload/",
        data=payload,
        **{
            "X-Tenant-Schema": tenant,
            "X-Tenant-Id": tenant,
            "X-Case-Id": case,
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["external_id"] == "doc-e2e"
    doc_id = body["document_id"]

    # Ingestion (direkt Task ausführen; alternativ run_ingestion.delay(...) und warten)
    result = process_document(tenant, case, doc_id)
    assert result["written"] >= 1

    # Query
    resp = client.post(
        "/ai/v1/rag-demo/",
        data=json.dumps({"query": "zebragurke", "top_k": 3}),
        content_type="application/json",
        **{
            "X-Tenant-Schema": tenant,
            "X-Tenant-Id": tenant,
            "X-Case-Id": case,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    # kein Demo-Fallback
    assert "warnings" not in data

    assert data["matches"], "expected vector matches"
    assert all(
        match.get("metadata", {}).get("external_id") == "doc-e2e"
        for match in data["matches"]
    )


@pytest.mark.django_db
def test_ingestion_run_reports_missing_documents(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    rag_database,
):
    tenant = test_tenant_schema_name
    case = "case-e2e"

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    upload = SimpleUploadedFile(
        "note.txt", b"hello ZEBRAGURKE world", content_type="text/plain"
    )
    payload = {
        "file": upload,
        "metadata": json.dumps({"label": "e2e", "external_id": "doc-e2e"}),
    }

    resp = client.post(
        "/ai/rag/documents/upload/",
        data=payload,
        **{
            "X-Tenant-Schema": tenant,
            "X-Tenant-Id": tenant,
            "X-Case-Id": case,
        },
    )
    assert resp.status_code == 202
    doc_id = resp.json()["document_id"]

    calls: list[tuple[tuple, dict]] = []

    class DummyTask:
        def delay(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("ai_core.views.run_ingestion", DummyTask())

    run_payload = {
        "document_ids": [doc_id, "missing-document"],
        "priority": "normal",
    }

    resp = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps(run_payload),
        content_type="application/json",
        **{
            "X-Tenant-Schema": tenant,
            "X-Tenant-Id": tenant,
            "X-Case-Id": case,
        },
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["invalid_ids"] == ["missing-document"]

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert list(args[2]) == [doc_id]


def test_fallback_external_id_consistency():
    result = make_fallback_external_id("note.txt", 11, b"hello world")
    assert result == "730b11756bd5a6af33f1ee8c07433a1042d6626af49ba4296d1170f0fdd71eff"
