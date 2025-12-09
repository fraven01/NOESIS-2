"""Bootstrap system collections for all tenants.

This management command ensures that all system-level collections
(like "manual-search") exist for all tenants in the database.

Usage:
    python manage.py bootstrap_system_collections
    python manage.py bootstrap_system_collections --tenant=public
"""

from django.core.management.base import BaseCommand
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.collection_service import CollectionService
from common.logging import get_logger

logger = get_logger(__name__)


class Command(BaseCommand):
    """Bootstrap system collections for tenants."""

    help = "Ensure all system collections exist for tenants"

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant",
            type=str,
            help="Specific tenant schema name (default: all tenants)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without actually creating",
        )

    def handle(self, *args, **options):
        tenant_filter = options.get("tenant")
        dry_run = options.get("dry_run", False)

        if tenant_filter:
            tenants = Tenant.objects.filter(schema_name=tenant_filter)
            if not tenants.exists():
                self.stdout.write(
                    self.style.ERROR(f"Tenant '{tenant_filter}' not found")
                )
                return
        else:
            tenants = Tenant.objects.all()

        service = CollectionService()
        total_tenants = tenants.count()
        created_count = 0
        existing_count = 0
        error_count = 0

        self.stdout.write(
            self.style.SUCCESS(
                f"Bootstrapping system collections for {total_tenants} tenant(s)..."
            )
        )

        for tenant in tenants:
            self.stdout.write(f"\nProcessing tenant: {tenant.schema_name}")

            with schema_context(tenant.schema_name):
                # Bootstrap "manual-search" collection
                try:
                    if dry_run:
                        self.stdout.write(
                            self.style.WARNING(
                                f"  [DRY RUN] Would ensure 'manual-search' collection for {tenant.schema_name}"
                            )
                        )
                        continue

                    # This will create the collection if it doesn't exist,
                    # or return the existing one if it does
                    collection_id = service.ensure_manual_collection(
                        tenant=tenant,
                        slug="manual-search",
                        label="Manual Search",
                    )

                    # Check if it was just created or already existed
                    from documents.models import DocumentCollection

                    collection = DocumentCollection.objects.get(
                        tenant=tenant, collection_id=collection_id
                    )

                    # If created_at is very recent (< 1 second), it was just created
                    from django.utils import timezone
                    from datetime import timedelta

                    just_created = (timezone.now() - collection.created_at) < timedelta(
                        seconds=1
                    )

                    if just_created:
                        created_count += 1
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"  ✅ Created 'manual-search' collection (ID: {collection_id})"
                            )
                        )
                    else:
                        existing_count += 1
                        self.stdout.write(
                            self.style.WARNING(
                                f"  ⏭️  'manual-search' collection already exists (ID: {collection_id})"
                            )
                        )

                    logger.info(
                        "bootstrap_system_collections.processed",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "collection_id": str(collection_id),
                            "collection_key": collection.key,
                            "was_created": just_created,
                        },
                    )

                except Exception as exc:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(
                            f"  ❌ Failed to bootstrap collections: {str(exc)}"
                        )
                    )
                    logger.exception(
                        "bootstrap_system_collections.failed",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "error": str(exc),
                        },
                    )

        # Summary
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("Bootstrap Summary:"))
        self.stdout.write(f"  Total tenants processed: {total_tenants}")
        self.stdout.write(self.style.SUCCESS(f"  Collections created: {created_count}"))
        self.stdout.write(
            self.style.WARNING(f"  Collections already existed: {existing_count}")
        )
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f"  Errors: {error_count}"))
        self.stdout.write("=" * 80)

        if dry_run:
            self.stdout.write(
                self.style.WARNING("\nℹ️  This was a DRY RUN - no changes were made")
            )
