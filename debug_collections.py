import os
import django
from uuid import UUID

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")
django.setup()

from documents.collection_service import CollectionService  # noqa: E402
from documents.models import DocumentCollection  # noqa: E402

tenant_id = "dev"
manual_uuid = CollectionService.manual_collection_uuid(tenant_id)
print(f"Computed Manual UUID for '{tenant_id}': {manual_uuid}")

print("\n--- DB Collections ---")
try:
    cols = DocumentCollection.objects.all()
    for c in cols:
        print(
            f"ID: {c.collection_id} | Name: {c.name} | Key: {c.key} | Tenant: {c.tenant.schema_name}"
        )

    print("\n--- Specific Checks ---")
    id_9375 = UUID("93758ef2-6d4f-543f-a4c2-af9d213fb98f")
    id_ebbce = UUID("ebbce985-7150-414d-bb60-46f8b1a6a97f")

    try:
        c1 = DocumentCollection.objects.get(collection_id=id_9375)
        print(
            f"9375... Found: Name={c1.name}, Key={c1.key}, Tenant={c1.tenant.schema_name}"
        )
    except Exception:
        print("9375... NOT FOUND")

    try:
        c2 = DocumentCollection.objects.get(collection_id=id_ebbce)
        print(
            f"ebbce... Found: Name={c2.name}, Key={c2.key}, Tenant={c2.tenant.schema_name}"
        )
    except Exception:
        print("ebbce... NOT FOUND")

except Exception as e:
    print(f"Error querying DB: {e}")
