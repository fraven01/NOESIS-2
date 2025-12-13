import os
import django
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from ai_core.infra import object_store  # noqa: E402
from ai_core.adapters.db_documents_repository import DbDocumentsRepository  # noqa: E402
from documents.contracts import FileBlob, LocalFileBlob  # noqa: E402


def diagnose():
    print(f"object_store.BASE_PATH = {object_store.BASE_PATH}")
    print(f"object_store.BASE_PATH.resolve() = {object_store.BASE_PATH.resolve()}")
    print(f"os.getcwd() = {os.getcwd()}")

    # Check if base path exists
    full_base = object_store.BASE_PATH.resolve()
    print(f"BASE_PATH exists: {full_base.exists()}")

    # List contents if it exists
    if full_base.exists():
        print(f"Contents of {full_base}:")
        for item in full_base.iterdir():
            print(f"  - {item.name} (dir={item.is_dir()})")

    # Try to find the example.com document
    doc_id = "dd77c68a-3fd9-4e18-9013-654b4b6162fb"
    tenant_id = "dev"

    try:
        repo = DbDocumentsRepository()
        doc = repo.get(tenant_id, doc_id)
        if doc:
            print(f"\nDocument found: {doc.ref.document_id}")
            print(f"Blob type: {type(doc.blob).__name__}")
            if isinstance(doc.blob, FileBlob):
                print(f"Blob URI: {doc.blob.uri}")
                resolved_path = object_store.BASE_PATH / doc.blob.uri
                print(f"Resolved path: {resolved_path}")
                print(f"Resolved path.resolve(): {resolved_path.resolve()}")
                print(f"File exists: {resolved_path.exists()}")
            elif isinstance(doc.blob, LocalFileBlob):
                print(f"Blob path: {doc.blob.path}")
                print(f"File exists: {Path(doc.blob.path).exists()}")
            else:
                print(f"Unknown blob type: {type(doc.blob)}")
        else:
            print(f"Document {doc_id} not found in tenant {tenant_id}")
    except Exception as e:
        print(f"Error fetching document: {e}")


if __name__ == "__main__":
    diagnose()
