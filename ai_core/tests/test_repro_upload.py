from uuid import uuid4
import pytest
from unittest.mock import patch

from ai_core.services import handle_document_upload
from django.core.files.uploadedfile import SimpleUploadedFile


@pytest.mark.django_db
def test_reproduce_failure():
    print("Starting reproduction...")

    # Simulate the data structure from ingestion_submit
    tenant_id = "b140e25e-04a9-5f32-b59e-90f7e6fa2419"  # from user log
    tenant_schema = "public"
    case_id = str(uuid4())
    manual_collection_id = str(uuid4())

    scope_context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,  # Matches view
        "trace_id": uuid4().hex,
        "invocation_id": uuid4().hex,
        "run_id": uuid4().hex,
    }
    business_context = {
        "case_id": case_id,
        "collection_id": manual_collection_id,
        "workflow_id": "document-upload-manual",
    }

    meta = {"scope_context": scope_context, "business_context": business_context}

    # Mock file
    file_content = b"fake pdf content"
    uploaded_file = SimpleUploadedFile(
        "test.pdf", file_content, content_type="application/pdf"
    )

    print(f"Meta: {meta}")

    with patch("ai_core.services.document_upload.with_scope_apply_async") as mock_apply:
        mock_apply.return_value.id = "task-123"
        response = handle_document_upload(
            upload=uploaded_file, metadata_raw=None, meta=meta, idempotency_key=None
        )

    if hasattr(response, "render") and getattr(response, "accepted_renderer", None):
        response.render()

    if response.status_code not in (200, 201, 202):
        print(f"Got status {response.status_code} (Valid failure, not crash)")
        # Try to print data or content safely
        data = getattr(response, "data", None)
        if data is None:
            data = (
                response.content if getattr(response, "content", None) else "No content"
            )
        print(f"Response data: {data}")

    # We accept 400 (Validation Error) as passing the regression test for the 500 Crash
    assert response.status_code in (200, 201, 202, 400)
    print(
        "Regression test passed: handle_document_upload execution completed (no 500)."
    )
