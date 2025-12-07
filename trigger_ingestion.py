import os
import sys
import django

sys.path.append(os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")
    sys.exit(1)

from django.test import Client
from documents.models import DocumentCollection


def trigger():
    print("Triggering ingestion via Django Client...")
    c = Client()

    # Needs matching tenant. 'dev' tenant usually maps to localhost.
    # We might need to start a tenant session or headers.
    # DocumentsRepository filters by proper tenant.

    # We need a valid collection ID.
    try:
        col = DocumentCollection.objects.first()
        col_id = (
            str(col.id) if col else "93758ef2-b0e2-4545-9383-751722026369"
        )  # fallback from logs
        print(f"Using Collection ID: {col_id}")
    except Exception:
        col_id = "00000000-0000-0000-0000-000000000000"
        print("Could not fetch collection, using zero UUID")

    payload = {
        "workflow_id": "ad-hoc",
        "urls": ["https://de.wikipedia.org/wiki/Bamberg"],
        "collection_id": col_id,
        "mode": "crawl_new",
        "depth": 1,
    }

    response = c.post(
        "/rag-tools/api/crawler/run/",
        data=payload,
        content_type="application/json",
        HTTP_X_TENANT_ID="dev",
        HTTP_X_WORKFLOW_ID="ad-hoc",
    )

    print(f"Response Status: {response.status_code}")
    print(f"Response Content: {response.content}")


if __name__ == "__main__":
    trigger()
