from django.conf import settings
from django.core.management.base import BaseCommand
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant
from documents.collection_service import CollectionService


class Command(BaseCommand):
    help = "Ensure the public tenant and primary domain exist."

    def add_arguments(self, parser):
        parser.add_argument("--domain", default="localhost")
        parser.add_argument(
            "--skip-collections",
            action="store_true",
            help="Skip bootstrapping system collections",
        )

    def handle(self, *args, **options):
        # Always operate on the public schema when creating the public tenant/domain
        with schema_context(get_public_schema_name()):
            tenant, created = Tenant.objects.get_or_create(
                schema_name=settings.PUBLIC_SCHEMA_NAME, defaults={"name": "Public"}
            )
            Domain.objects.get_or_create(
                domain=options["domain"], tenant=tenant, defaults={"is_primary": True}
            )

            # Bootstrap system collections for the public tenant
            if not options.get("skip_collections"):
                # Check if 'documents' is installed in the public schema (SHARED_APPS)
                # We check for "documents" or configured AppConfig paths starting with "documents."
                is_documents_shared = any(
                    app == "documents" or app.startswith("documents.")
                    for app in settings.SHARED_APPS
                )

                if is_documents_shared:
                    self.stdout.write("Bootstrapping system collections...")
                    try:
                        service = CollectionService()
                        collection_id = service.ensure_manual_collection(
                            tenant=tenant,
                            slug="manual-search",
                            label="Manual Search",
                        )
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"✅ System collection 'manual-search' ready (ID: {collection_id})"
                            )
                        )
                    except Exception as exc:
                        self.stdout.write(
                            self.style.WARNING(
                                f"⚠️  Could not bootstrap collections: {str(exc)}"
                            )
                        )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            "⚠️  Skipping system collection bootstrap: 'documents' app is not shared."
                        )
                    )
