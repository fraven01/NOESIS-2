#!/usr/bin/env python
"""Manually trigger ingestion for a specific document to test parser fix."""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

import django

django.setup()

from uuid import UUID  # noqa: E402
from django_tenants.utils import schema_context  # noqa: E402
from ai_core.ingestion import process_document  # noqa: E402

DOC_ID = "fd168f2b-3289-47a8-8b76-0e8271374a66"
TENANT_SCHEMA = "dev"
CASE = "upload"
EMBEDDING_PROFILE = "standard"


def main():
    print(f"=== Re-running ingestion for document: {DOC_ID} ===\n")

    # First, verify the parser can now handle the document
    print("Testing parser detection...")

    from customers.models import Tenant
    from documents.models import Document
    from documents.parsers_pdf import PdfDocumentParser
    from documents.contracts import NormalizedDocument

    _tenant = Tenant.objects.get(schema_name=TENANT_SCHEMA)  # noqa: F841

    with schema_context(TENANT_SCHEMA):
        doc = Document.objects.get(id=UUID(DOC_ID))

        # Create NormalizedDocument from metadata
        metadata = doc.metadata or {}
        normalized_doc_dict = metadata.get("normalized_document", {})

        print(
            f"Blob media_type: {normalized_doc_dict.get('blob', {}).get('media_type')}"
        )
        print(
            f"External ref media_type: {normalized_doc_dict.get('meta', {}).get('external_ref', {}).get('media_type')}"
        )

        # Build a NormalizedDocument
        try:
            normalized_document = NormalizedDocument.model_validate(normalized_doc_dict)
            print("Loaded NormalizedDocument successfully")
            print(
                f"  blob.media_type: {getattr(normalized_document.blob, 'media_type', None)}"
            )

            # Test parser
            parser = PdfDocumentParser()
            can_handle = parser.can_handle(normalized_document)
            print(f"\nPdfDocumentParser.can_handle(normalized_document) = {can_handle}")

            if can_handle:
                print("\n✅ Parser fix WORKS! Parser can now detect the PDF.")
            else:
                print("\n❌ Parser still cannot handle the document!")
                return

        except Exception as e:
            print(f"Error loading document: {e}")
            return

    # Now run ingestion
    print("\n=== Running ingestion task ===")
    try:
        result = process_document.run(
            tenant=TENANT_SCHEMA,
            case=CASE,
            document_id=DOC_ID,
            embedding_profile=EMBEDDING_PROFILE,
            tenant_schema=TENANT_SCHEMA,
            trace_id="debug-manual-trigger",
        )
        print("\nIngestion result:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        if result.get("written", 0) > 0:
            print(
                "\n✅ SUCCESS! Document was ingested with chunks written to vector store!"
            )
        else:
            print("\n⚠️ Document processed but no chunks written (might be delta skip)")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
