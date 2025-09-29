import json
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.client import BOUNDARY, MULTIPART_CONTENT, encode_multipart

from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


@pytest.mark.django_db
def test_rag_upload_persists_file_and_metadata(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

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

    document_id = body["document_id"]
    tenant_segment = object_store.sanitize_identifier(test_tenant_schema_name)
    case_segment = object_store.sanitize_identifier("case-123")
    uploads_dir = Path(tmp_path, tenant_segment, case_segment, "uploads")

    stored_files = list(uploads_dir.glob(f"{document_id}_*"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == b"hello world"

    metadata_path = uploads_dir / f"{document_id}.meta.json"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text()) == metadata


@pytest.mark.django_db
def test_rag_upload_external_id_fallback(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

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
