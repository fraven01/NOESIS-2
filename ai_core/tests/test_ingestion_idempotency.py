import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

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

    def upload_document(content: str) -> str:
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
        assert body["external_id"] == external_id
        return body["document_id"]

    first_doc = upload_document("Hello RAG ingestion!")
    first_result = process_document(tenant, case, first_doc, tenant_schema=tenant)

    assert first_result["external_id"] == external_id
    assert first_result["inserted"] == 1
    assert first_result["skipped"] == 0
    assert first_result["replaced"] == 0
    assert first_result["action"] == "inserted"
    assert first_result["written"] == 1

    second_doc = upload_document("Hello RAG ingestion!")
    second_result = process_document(tenant, case, second_doc, tenant_schema=tenant)

    assert second_result["external_id"] == external_id
    assert second_result["inserted"] == 0
    assert second_result["skipped"] == 1
    assert second_result["replaced"] == 0
    assert second_result["action"] == "skipped"
    assert second_result["written"] == 0

    third_doc = upload_document("Hello RAG ingestion version two!")
    third_result = process_document(tenant, case, third_doc, tenant_schema=tenant)

    assert third_result["external_id"] == external_id
    assert third_result["skipped"] == 0
    assert third_result["action"] in {"inserted", "replaced"}
    assert third_result["inserted"] == 1 or third_result["replaced"] == 1
    assert third_result["written"] == 1
