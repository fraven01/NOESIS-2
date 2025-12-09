import pickle
import sys
import os
from uuid import uuid4
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Project Root: {PROJECT_ROOT}")

try:
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.base")
    django.setup()
    print("Django setup success")
except Exception as e:
    print(f"Django setup failed: {e}")
    sys.exit(1)

try:
    from documents.contracts import (
        NormalizedDocument,
        DocumentMeta,
        DocumentRef,
        InlineBlob,
    )

    print("Imported documents.contracts")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    from ai_core.rag.ingestion_contracts import (
        CrawlerIngestionPayload,
        IngestionAction,
        ChunkMeta,
    )

    print("Imported ai_core.rag.ingestion_contracts")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def test_pickle_serialization():
    print("Testing serialization of core ingestion objects...")

    # 1. Create a dummy NormalizedDocument
    print("Creating NormalizedDocument...")
    try:
        doc_ref = DocumentRef(
            tenant_id="dev",
            workflow_id="test-workflow",
            document_id=uuid4(),
            collection_id=uuid4(),
        )

        meta = DocumentMeta(
            tenant_id="dev",
            workflow_id="test-workflow",
            title="Test Doc",
            origin_uri="http://example.com",
        )

        # Hash of empty string
        empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        blob = InlineBlob(
            type="inline", media_type="text/plain", base64="", sha256=empty_hash, size=0
        )

        norm_doc = NormalizedDocument(
            ref=doc_ref,
            meta=meta,
            blob=blob,
            checksum=empty_hash,
            created_at=datetime.now(timezone.utc),
        )

        pickle.dumps(norm_doc)
        print("OK: NormalizedDocument is pickleable")
    except Exception as e:
        print(f"FAIL: NormalizedDocument serialization FAILED: {e}")
        if hasattr(norm_doc.meta, "parse_stats"):
            print(f"  meta.parse_stats type: {type(norm_doc.meta.parse_stats)}")

    # 3. Create CrawlerIngestionPayload
    print("Creating CrawlerIngestionPayload...")
    try:
        payload = CrawlerIngestionPayload(
            action=IngestionAction.UPSERT,
            lifecycle_state="active",
            adapter_metadata={},
            document_id=str(norm_doc.ref.document_id),
            workflow_id=norm_doc.ref.workflow_id,
            tenant_id=norm_doc.ref.tenant_id,
            case_id="default",
            chunk_meta=ChunkMeta(
                tenant_id="dev",
                case_id="default",
                source="crawler",
                hash="0" * 64,
                external_id="ext",
                content_hash="0" * 64,
            ),
        )

        pickle.dumps(payload)
        print("OK: CrawlerIngestionPayload is pickleable")
    except Exception as e:
        print(f"FAIL: CrawlerIngestionPayload serialization FAILED: {e}")
        if hasattr(payload, "adapter_metadata"):
            print(f"  adapter_metadata type: {type(payload.adapter_metadata)}")


if __name__ == "__main__":
    with open("verification_result.txt", "w") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            test_pickle_serialization()
        except Exception as e:
            print(f"\nCRITICAL SCRIPT FAILURE: {e}")
            import traceback

            traceback.print_exc()
