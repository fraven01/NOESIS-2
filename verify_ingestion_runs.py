import os
import sys
import django

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()

from ai_core.models import IngestionRun  # noqa: E402
from django.db import connection, ProgrammingError  # noqa: E402
from customers.models import Tenant  # noqa: E402


def verify_ingestion_runs():
    print("Verifying IngestionRun objects in database...")

    for tenant in Tenant.objects.all():
        print(f"\nChecking tenant: {tenant.schema_name}")
        connection.set_tenant(tenant)

        try:
            count = IngestionRun.objects.count()
            print(f"  Found {count} IngestionRuns.")

            runs = IngestionRun.objects.all().order_by("-created_at")[:3]
            for run in runs:
                print(f"  Run ID: {run.run_id}")
                print(f"  Created At: {run.created_at}")
                print(f"  Status: {run.status}")

        except ProgrammingError:
            print(f"  Skipping tenant {tenant.schema_name} (table not found)")
            continue

        if count == 0:
            print("  No IngestionRuns found.")


if __name__ == "__main__":
    verify_ingestion_runs()
