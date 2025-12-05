import os
import sys
import django

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from documents.models import Document  # noqa: E402
from django.db import connection, ProgrammingError  # noqa: E402
from customers.models import Tenant  # noqa: E402


def verify_documents():
    print("Verifying documents in database...")

    for tenant in Tenant.objects.all():
        print(f"\nChecking tenant: {tenant.schema_name}")
        connection.set_tenant(tenant)

        try:
            count = Document.objects.count()
            print(f"  Found {count} documents.")

            docs = Document.objects.all().order_by("-created_at")[:3]
            # Force evaluation
            docs = list(docs)
        except ProgrammingError:
            print(f"  Skipping tenant {tenant.schema_name} (documents table not found)")
            continue

        if not docs:
            print("  No documents found.")
            continue

        for doc in docs:
            print(f"  Document ID: {doc.id}")
            print(f"  Created At: {doc.created_at}")
            print(f"  Metadata: {doc.metadata}")

            title = doc.metadata.get("title")
            print(f"  Title: {title}")

            # Check for size in metadata
            size = doc.metadata.get("size")
            print(f"  Size in Metadata: {size}")

            # Check for content_type
            content_type = doc.metadata.get("content_type")
            print(f"  Content Type: {content_type}")


if __name__ == "__main__":
    verify_documents()
