import json
import uuid
from pathlib import Path

import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.client import BOUNDARY, MULTIPART_CONTENT, encode_multipart

from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_COLLECTION_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from types import SimpleNamespace


@pytest.mark.django_db
def test_rag_upload_persists_file_and_metadata(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    documents_repository_stub,
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    captured: dict[str, object] = {}

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
                "document_ids": list(document_ids),
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

    upload = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
    metadata = {"source": "unit-test"}

    payload = encode_multipart(
        BOUNDARY, {"file": upload, "metadata": json.dumps(metadata)}
    )
    response = client.generic(
        "POST",
        "/ai/rag/documents/upload/",
        payload,
        content_type=MULTIPART_CONTENT,
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-123",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["trace_id"]
    assert body["idempotent"] is False
    assert body["ingestion_status"] == "queued"
    assert body["ingestion_run_id"]

    assert captured["tenant_id"] == test_tenant_schema_name
    assert captured["case_id"] == "case-123"
    assert captured["document_ids"] == [body["document_id"]]
    assert captured["embedding_profile"] == getattr(
        settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
    )
    assert captured["run_id"] == body["ingestion_run_id"]
    assert captured["trace_id"] == body["trace_id"]
    assert captured["tenant_schema"] == test_tenant_schema_name

    document_id = body["document_id"]
    tenant_segment = object_store.sanitize_identifier(test_tenant_schema_name)
    case_segment = object_store.sanitize_identifier("case-123")
    uploads_dir = Path(tmp_path, tenant_segment, case_segment, "uploads")

    stored_files = list(uploads_dir.glob(f"{document_id}_*"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == b"hello world"

    metadata_path = uploads_dir / f"{document_id}.meta.json"
    assert metadata_path.exists()
    stored_metadata = json.loads(metadata_path.read_text())
    assert stored_metadata["external_id"] == body["external_id"]
    assert "source" not in stored_metadata

    assert len(documents_repository_stub.saved) == 1
    saved_document = documents_repository_stub.saved[0]
    assert str(saved_document.ref.document_id) == document_id
    assert (
        saved_document.meta.external_ref
        and saved_document.meta.external_ref["external_id"] == body["external_id"]
    )
    assert saved_document.blob.media_type == "text/plain"

    status_path = (
        Path(tmp_path) / tenant_segment / case_segment / "ingestion" / "run_status.json"
    )
    assert status_path.exists()
    status_payload = json.loads(status_path.read_text())
    assert status_payload["run_id"] == body["ingestion_run_id"]
    assert status_payload["status"] == "queued"
    assert status_payload["document_ids"] == [body["document_id"]]


@pytest.mark.django_db
def test_rag_upload_bridges_collection_header_to_metadata(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    upload = SimpleUploadedFile(
        "notes.txt", b"header scoped", content_type="text/plain"
    )
    payload = encode_multipart(BOUNDARY, {"file": upload})

    collection_scope = str(uuid.uuid4())
    response = client.generic(
        "POST",
        "/ai/rag/documents/upload/",
        payload,
        content_type=MULTIPART_CONTENT,
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-collection",
            META_COLLECTION_ID_KEY: collection_scope,
        },
    )

    assert response.status_code == 202
    document_id = response.json()["document_id"]

    tenant_segment = object_store.sanitize_identifier(test_tenant_schema_name)
    case_segment = object_store.sanitize_identifier("case-collection")
    metadata_path = Path(
        tmp_path, tenant_segment, case_segment, "uploads", f"{document_id}.meta.json"
    )

    assert metadata_path.exists()
    stored_metadata = json.loads(metadata_path.read_text())
    assert stored_metadata["collection_id"] == collection_scope
    assert stored_metadata["external_id"]
    assert set(stored_metadata) == {"collection_id", "external_id"}


@pytest.mark.django_db
def test_rag_upload_external_id_fallback(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    def _upload_once() -> dict:
        upload = SimpleUploadedFile(
            "notes.txt", b"hello world", content_type="text/plain"
        )
        metadata = {"label": "fallback"}
        payload = encode_multipart(
            BOUNDARY, {"file": upload, "metadata": json.dumps(metadata)}
        )
        response = client.generic(
            "POST",
            "/ai/rag/documents/upload/",
            payload,
            content_type=MULTIPART_CONTENT,
            **{
                META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
                META_TENANT_ID_KEY: test_tenant_schema_name,
                META_CASE_ID_KEY: "case-123",
            },
        )
        assert response.status_code == 202
        return response.json()

    first = _upload_once()
    second = _upload_once()

    assert first["external_id"]
    assert first["external_id"] == second["external_id"]
    assert first["document_id"] != second["document_id"]


@pytest.mark.django_db
def test_rag_upload_without_file_returns_400(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    payload = encode_multipart(BOUNDARY, {"metadata": json.dumps({"foo": "bar"})})
    response = client.generic(
        "POST",
        "/ai/rag/documents/upload/",
        payload,
        content_type=MULTIPART_CONTENT,
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-123",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "File form part is required for document uploads."
    assert body["code"] == "missing_file"


@pytest.mark.django_db
def test_rag_upload_with_invalid_metadata_returns_400(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    upload = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")

    payload = encode_multipart(BOUNDARY, {"file": upload, "metadata": "{not-json"})
    response = client.generic(
        "POST",
        "/ai/rag/documents/upload/",
        payload,
        content_type=MULTIPART_CONTENT,
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-123",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Metadata must be valid JSON."
    assert body["code"] == "invalid_metadata"
