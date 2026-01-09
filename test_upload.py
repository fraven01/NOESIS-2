#!/usr/bin/env python
"""Test upload with enable_embedding=True."""
import os
import sys


def main() -> int:
    # Add project to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

    import django

    django.setup()

    from django.core.files.uploadedfile import SimpleUploadedFile
    from uuid import uuid4
    import json

    from ai_core.contracts import ScopeContext, BusinessContext
    from ai_core.services import handle_document_upload

    # Create a simple test PDF content (minimal valid PDF)
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000229 00000 n
0000000328 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
420
%%EOF"""

    # Create uploaded file
    uploaded_file = SimpleUploadedFile(
        name="test_document.pdf",
        content=pdf_content,
        content_type="application/pdf",
    )

    # Build meta (simulating UI request)
    tenant_id = "dev"
    scope = ScopeContext(
        tenant_id=tenant_id,
        tenant_schema="dev",
        trace_id=uuid4().hex,
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
    )
    business = BusinessContext(
        workflow_id="test-upload",
    )

    meta = {
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
        "tool_context": scope.to_tool_context(business=business).model_dump(
            mode="json", exclude_none=True
        ),
    }

    # Test with enable_embedding=True (should use default True now)
    metadata_obj = {}  # Empty - should use default enable_embedding=True
    metadata_raw = json.dumps(metadata_obj)

    print("Testing upload with enable_embedding=True (default)...")
    print(f"Metadata: {metadata_raw}")

    try:
        response = handle_document_upload(
            upload=uploaded_file,
            metadata_raw=metadata_raw,
            meta=meta,
            idempotency_key=None,
        )
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Data: {response.data}")

        if response.status_code < 400:
            print("\nOK: Upload successful.")
            print(f"Document ID: {response.data.get('document_id')}")
            print(f"Ingestion Run ID: {response.data.get('ingestion_run_id')}")
        else:
            print(f"\nERROR: Upload failed: {response.data}")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
