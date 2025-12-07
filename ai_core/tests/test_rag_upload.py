import json
import uuid

import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.client import BOUNDARY, MULTIPART_CONTENT, encode_multipart

from ai_core.infra import object_store, rate_limit
from ai_core.rag.collections import manual_collection_uuid
from common.constants import (
    META_COLLECTION_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from customers.models import Tenant
from documents.models import DocumentCollection
from types import SimpleNamespace


@pytest.mark.django_db
def test_rag_upload_persists_file_and_metadata(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    documents_repository_stub,
    ingestion_status_store,
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    # Case creation removed for caseless strategy
    tenant_obj = Tenant.objects.get(schema_name=test_tenant_schema_name)
    DocumentCollection.objects.filter(tenant=tenant_obj).delete()

    # with tenant_context(tenant_obj):
    #     Case.objects.create(tenant=tenant_obj, external_id="case-123")

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
            # META_CASE_ID_KEY: "case-123",  # Removed for caseless test
        },
    )

    if response.status_code != 202:
        with open("ai_core/tests/debug_output.txt", "w") as f:
            f.write(f"Response Status: {response.status_code}\n")
            f.write(f"Response Content: {response.content.decode()}\n")
    assert response.status_code == 202
    body = response.json()
    manual_collection_id = str(manual_collection_uuid(tenant_obj))
    assert body["trace_id"]
    assert body["workflow_id"]  # Generated UUID in caseless mode
    assert body["ingestion_run_id"]
    assert body["collection_id"] == manual_collection_id

    assert captured["tenant_id"] == test_tenant_schema_name
    assert captured["case_id"] == ""  # Empty for caseless
    assert captured["document_ids"] == [body["document_id"]]
    assert captured["embedding_profile"] == getattr(
        settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
    )
    assert captured["run_id"] == body["ingestion_run_id"]
    assert captured["trace_id"] == body["trace_id"]
    assert captured["tenant_schema"] == test_tenant_schema_name

    document_id = body["document_id"]
    assert len(documents_repository_stub.saved) == 1
    saved_document = documents_repository_stub.saved[0]
    assert str(saved_document.ref.document_id) == document_id
    assert str(saved_document.ref.collection_id) == manual_collection_id
    assert saved_document.meta.external_ref
    assert saved_document.meta.external_ref.get("external_id")
    assert getattr(saved_document.blob, "type", "") == "file"

    status_payload = ingestion_status_store.get_ingestion_run(
        tenant_id=test_tenant_schema_name,
        case="",  # Empty case
    )

    assert status_payload is not None
    assert status_payload["run_id"] == body["ingestion_run_id"]
    assert status_payload["status"] == "queued"
    assert status_payload["document_ids"] == [body["document_id"]]
    assert status_payload["tenant_id"] == test_tenant_schema_name

    collection_entry = DocumentCollection.objects.get(
        tenant__schema_name=test_tenant_schema_name,
        key="manual-search",
    )

    assert str(collection_entry.collection_id) == manual_collection_id


@pytest.mark.django_db
def test_rag_upload_bridges_collection_header_to_metadata(
    client, monkeypatch, tmp_path, test_tenant_schema_name, documents_repository_stub
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
            # META_CASE_ID_KEY: "case-collection",
            META_COLLECTION_ID_KEY: collection_scope,
        },
    )

    assert response.status_code == 202
    body = response.json()
    document_id = body["document_id"]
    assert body["workflow_id"]  # Generated UUID
    assert body["collection_id"] == collection_scope

    assert len(documents_repository_stub.saved) == 1
    saved_document = documents_repository_stub.saved[0]
    assert str(saved_document.ref.document_id) == document_id
    assert str(saved_document.ref.collection_id) == collection_scope
    assert saved_document.meta.external_ref.get("external_id")


@pytest.mark.django_db
def test_rag_upload_external_id_fallback(
    client, monkeypatch, tmp_path, test_tenant_schema_name, documents_repository_stub
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    def _upload_once() -> tuple[str, str]:
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
                # META_CASE_ID_KEY: "case-123",
            },
        )

        assert response.status_code == 202
        body = response.json()
        document_id = body["document_id"]
        saved_document = documents_repository_stub.saved[-1]
        external_id = saved_document.meta.external_ref.get("external_id")
        return document_id, external_id

    first_doc, first_external = _upload_once()
    second_doc, second_external = _upload_once()

    assert first_external
    assert first_external == second_external
    assert first_doc == second_doc


@pytest.mark.django_db
def test_rag_upload_guardrail_skip_returns_403(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    documents_repository_stub,
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    class _StubGraph:
        def __init__(self, *, persistence_handler=None):
            self.persistence_handler = persistence_handler

        def run(self, payload, run_until=None):
            assert payload["tenant_id"] == test_tenant_schema_name
            return {
                "decision": "skip_guardrail",
                "reason": "blocked",
                "transitions": {
                    "accept_upload": {
                        "decision": "accepted",
                        "diagnostics": {},
                    },
                    "delta_and_guardrails": {
                        "decision": "skip_guardrail",
                        "diagnostics": {"policy_events": ("upload_blocked",)},
                    },
                },
            }

    monkeypatch.setattr("ai_core.services.UploadIngestionGraph", _StubGraph)

    upload = SimpleUploadedFile("blocked.txt", b"deny-me", content_type="text/plain")
    payload = encode_multipart(BOUNDARY, {"file": upload})
    response = client.generic(
        "POST",
        "/ai/rag/documents/upload/",
        payload,
        content_type=MULTIPART_CONTENT,
        **{
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_TENANT_ID_KEY: test_tenant_schema_name,
        },
    )

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "upload_blocked"
    assert "guardrail" in body["detail"].lower()
    assert documents_repository_stub.saved == []


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
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Metadata must be valid JSON."
    assert body["code"] == "invalid_metadata"
