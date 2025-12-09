from django_tenants.utils import schema_context
from customers.models import Tenant
from documents.models import Document, DocumentCollection


def run():
    print("Debug Persistence Script Started")

    # 1. List all tenants
    tenants = list(Tenant.objects.all())
    print(f"Found {len(tenants)} tenants: {[t.schema_name for t in tenants]}")

    # 2. Check Document counts in each schema
    for tenant in tenants:
        schema = tenant.schema_name
        print(f"\nChecking schema: {schema}")
        try:
            with schema_context(schema):
                doc_count = Document.objects.count()
                coll_count = DocumentCollection.objects.count()
                print(f"  Documents: {doc_count}")
                print(f"  Collections: {coll_count}")

                if doc_count > 0:
                    print("  Sample Documents:")
                    for doc in Document.objects.all()[:5]:
                        print(
                            f"    - ID: {doc.id}, Source: {doc.source}, State: {doc.lifecycle_state}"
                        )

                if coll_count > 0:
                    print("  Sample Collections:")
                    for coll in DocumentCollection.objects.all()[:5]:
                        print(
                            f"    - ID: {coll.collection_id}, Key: {coll.key}, Name: {coll.name}"
                        )

        except Exception as e:
            print(f"  Error accessing schema {schema}: {e}")

    # 3. Check public schema explicitly (if not covered)
    if "public" not in [t.schema_name for t in tenants]:
        print("\nChecking schema: public (explicit)")
        try:
            with schema_context("public"):
                # Note: Document might not exist in public if tenant-specific
                try:
                    doc_count = Document.objects.count()
                    print(f"  Documents: {doc_count}")
                except Exception as e:
                    print(f"  Documents table error: {e}")
        except Exception as e:
            print(f"  Error accessing public: {e}")


if __name__ == "__main__":
    run()
