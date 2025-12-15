#!/usr/bin/env python
"""Debug script to look up a document by ID across all tenants."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

import django

django.setup()

from uuid import UUID
from django.conf import settings
from django_tenants.utils import schema_context
from customers.models import Tenant
from documents.models import Document, DocumentCollection, DocumentCollectionMembership

DOC_ID = "fd168f2b-3289-47a8-8b76-0e8271374a66"


def main():
    doc_uuid = UUID(DOC_ID)
    print(f"=== Looking for document: {DOC_ID} ===\n")

    # List all tenants
    tenants = list(Tenant.objects.all())
    print(f"Found {len(tenants)} tenants:")
    for t in tenants:
        print(f"  - {t.schema_name}")
    print()

    # Search in each tenant
    for tenant in tenants:
        schema = tenant.schema_name
        print(f"--- Checking tenant: {schema} ---")
        try:
            with schema_context(schema):
                # Check for document
                doc = Document.objects.filter(id=doc_uuid).first()
                if doc:
                    print(f"  FOUND document!")
                    print(f"    ID: {doc.id}")
                    print(f"    Source: {doc.source}")
                    print(f"    Hash: {doc.hash}")
                    print(f"    Lifecycle State: {doc.lifecycle_state}")
                    print(f"    Created: {doc.created_at}")
                    print(
                        f"    Metadata keys: {list(doc.metadata.keys()) if doc.metadata else 'None'}"
                    )

                    # Check memberships
                    memberships = DocumentCollectionMembership.objects.filter(
                        document_id=doc.id
                    )
                    print(f"    Collection memberships: {memberships.count()}")
                    for m in memberships:
                        print(
                            f"      - Collection: {m.collection.name if m.collection else 'N/A'} (key: {m.collection.key if m.collection else 'N/A'})"
                        )
                else:
                    doc_count = Document.objects.count()
                    print(f"  Not found. (Total docs in schema: {doc_count})")

                    # List recent documents
                    if doc_count > 0:
                        recent = Document.objects.order_by("-created_at")[:5]
                        print(f"  Recent documents:")
                        for d in recent:
                            print(f"    - {d.id} ({d.source}) created {d.created_at}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Also check vector store
    print("=== Checking Vector Store ===")
    try:
        from ai_core.rag.vector_client import PgVectorClient
        from ai_core.rag.vector_store import resolve_vector_store

        store = resolve_vector_store("global")
        if hasattr(store, "_get_connection") or hasattr(store, "search_chunks"):
            print("Vector store resolved. Checking for chunks...")

            # Try to find chunks by document_id
            from django.db import connection
            from django.conf import settings

            # Get vector store table info
            vector_config = getattr(settings, "RAG_VECTOR_STORES", {}).get("global", {})
            print(f"Vector store config: {vector_config}")
    except Exception as e:
        print(f"Vector store check error: {e}")


if __name__ == "__main__":
    main()
