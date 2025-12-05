import os
import sys
import json
import django
from django.core.files.uploadedfile import SimpleUploadedFile

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from ai_core.services import handle_document_upload  # noqa: E402
from customers.models import Tenant  # noqa: E402


def simulate_upload():
    print("Simulating document upload...")

    # Get a tenant (e.g., dev)
    tenant = Tenant.objects.filter(schema_name="dev").first()
    if not tenant:
        print("Tenant 'dev' not found.")
        return

    print(f"Using tenant: {tenant.schema_name}")

    # Create dummy PDF file
    pdf_content = b"%PDF-1.4 dummy content"
    pdf_file = SimpleUploadedFile(
        "test_upload.pdf", pdf_content, content_type="application/octet-stream"
    )

    # Create dummy HTML file
    html_content = b"<html><head><title>Extracted Title</title></head><body><h1>Hello</h1></body></html>"
    html_file = SimpleUploadedFile(
        "test_upload.html", html_content, content_type="text/html"
    )

    # Metadata with missing title to test fallback
    metadata = {"collection_id": None}

    # Set tenant context
    from django.db import connection

    connection.set_tenant(tenant)

    # 1. Test PDF Upload (should default title to filename)
    print("\nUploading PDF...")
    try:
        result_pdf = handle_document_upload(
            upload=pdf_file,
            metadata_raw=json.dumps(metadata),
            meta={
                "tenant_id": tenant.schema_name,
                "case_id": None,
            },  # Assuming schema_name is used as tenant_id in meta
            idempotency_key=None,
        )
        print(f"PDF Upload Result: {result_pdf}")
    except Exception as e:
        print(f"PDF Upload Failed: {e}")

    # 2. Test HTML Upload (should detect mime type)
    print("\nUploading HTML...")
    try:
        result_html = handle_document_upload(
            upload=html_file,
            metadata_raw=json.dumps(metadata),
            meta={"tenant_id": tenant.schema_name, "case_id": None},
            idempotency_key=None,
        )
        print(f"HTML Upload Result: {result_html}")
    except Exception as e:
        print(f"HTML Upload Failed: {e}")


if __name__ == "__main__":
    simulate_upload()
