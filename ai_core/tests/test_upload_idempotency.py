import pytest
import hashlib
from uuid import uuid4
from datetime import timedelta
from django.utils import timezone
from unittest.mock import patch
from django.apps import apps
from django_tenants.utils import schema_context
from django.core.files.uploadedfile import SimpleUploadedFile

from ai_core.services.document_upload import handle_document_upload
from customers.tenant_context import TenantContext


@pytest.mark.django_db
def test_upload_overwrites_stale_pending_document(test_tenant_schema_name):
    # 1. Setup: Use the project's test tenant
    tenant_schema = test_tenant_schema_name
    tenant = TenantContext.resolve_identifier(tenant_schema, allow_pk=True)
    tenant_id = str(tenant.id)

    Document = apps.get_model("documents", "Document")
    source = "upload"  # Use the default source
    file_content = b"fake pdf content for idempotency test " + str(uuid4()).encode()
    checksum = hashlib.sha256(file_content).hexdigest()

    with schema_context(tenant_schema):
        # Create a document that is already 3 hours old and 'pending'
        doc = Document.objects.create(
            tenant=tenant,
            source=source,
            hash=checksum,
            lifecycle_state="pending",
        )
        # Manually set created_at back in time
        Document.objects.filter(id=doc.id).update(
            created_at=timezone.now() - timedelta(hours=3)
        )

    # 2. Test: Attempt to upload the same document
    scope_context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "trace_id": uuid4().hex,
        "invocation_id": uuid4().hex,
        "run_id": uuid4().hex,
    }
    # BusinessContext fields are optional per Option A
    business_context = {
        "workflow_id": "test-idempotency",
    }
    meta = {"scope_context": scope_context, "business_context": business_context}

    uploaded_file = SimpleUploadedFile(
        "test.pdf", file_content, content_type="application/pdf"
    )

    # Mock the task dispatch since we only want to test the service-level check
    with patch("ai_core.services.document_upload.with_scope_apply_async") as mock_apply:
        mock_apply.return_value.id = "task-idempotency"

        response = handle_document_upload(
            upload=uploaded_file, metadata_raw=None, meta=meta, idempotency_key=None
        )

    # 3. Assert: Should BE ACCEPTED (202) because the previous one was stale pending
    assert response.status_code == 202
    print("Test passed: Stale pending document was overwritten.")


@pytest.mark.django_db
def test_upload_rejects_fresh_pending_document(test_tenant_schema_name):
    # Setup: Use the project's test tenant
    tenant_schema = test_tenant_schema_name
    tenant = TenantContext.resolve_identifier(tenant_schema, allow_pk=True)
    tenant_id = str(tenant.id)

    Document = apps.get_model("documents", "Document")
    source = "upload"
    file_content = b"fresh pending content " + str(uuid4()).encode()
    checksum = hashlib.sha256(file_content).hexdigest()

    with schema_context(tenant_schema):
        # Create a document that is FRESH and 'pending'
        Document.objects.create(
            tenant=tenant,
            source=source,
            hash=checksum,
            lifecycle_state="pending",
        )

    # Test: Attempt to upload the same document
    scope_context = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "trace_id": uuid4().hex,
        "invocation_id": uuid4().hex,
        "run_id": uuid4().hex,
    }
    meta = {"scope_context": scope_context, "business_context": {}}

    uploaded_file = SimpleUploadedFile(
        "test.pdf", file_content, content_type="application/pdf"
    )

    response = handle_document_upload(
        upload=uploaded_file, metadata_raw=None, meta=meta, idempotency_key=None
    )

    # Assert: Should be REJECTED (409) because it's not stale yet
    assert response.status_code == 409
    print("Test passed: Fresh pending document was correctly rejected.")
