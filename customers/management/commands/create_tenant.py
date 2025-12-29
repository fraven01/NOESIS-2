from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction, connection
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant


class Command(BaseCommand):
    help = "Create a tenant and associated domain"

    def add_arguments(self, parser):
        parser.add_argument("--schema", "--schema_name", dest="schema", required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--domain", "--domain-domain", dest="domain", required=True)
        parser.add_argument(
            "--tenant-type",
            choices=["ENTERPRISE", "LAW_FIRM"],
            default="ENTERPRISE",
            help="Tenant type (default: ENTERPRISE)",
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        name = options["name"]
        domain = options["domain"]
        tenant_type = options.get("tenant_type", "ENTERPRISE")

        if schema == get_public_schema_name():
            raise CommandError("Schema 'public' is reserved")
        if Tenant.objects.filter(schema_name=schema).exists():
            raise CommandError("Schema already exists")
        if Domain.objects.filter(domain=domain).exists():
            raise CommandError("Domain already exists")

        with schema_context(get_public_schema_name()):
            with transaction.atomic():
                tenant = Tenant.objects.create(
                    schema_name=schema,
                    name=name,
                    tenant_type=tenant_type,
                )
                Domain.objects.create(domain=domain, tenant=tenant, is_primary=True)

        original_auto_create = getattr(tenant, "auto_create_schema", True)
        tenant.auto_create_schema = True
        try:
            tenant.create_schema(check_if_exists=True)
        finally:
            tenant.auto_create_schema = original_auto_create

        call_command("migrate_schemas", schema=schema, tenant=True)

        # Ensure presence of customers_domain in tenant schema for introspection-based checks
        with schema_context(schema):
            with connection.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS customers_domain (LIKE public.customers_domain INCLUDING ALL)"
                )

        self.stdout.write(self.style.SUCCESS(f"Tenant '{name}' created"))
