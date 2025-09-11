from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant


class Command(BaseCommand):
    help = "Create a tenant and associated domain"

    def add_arguments(self, parser):
        # Accept synonyms to avoid conflicts with django-tenants built-in command flags
        parser.add_argument("--schema", "--schema_name", dest="schema", required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument(
            "--domain", "--domain-domain", dest="domain", required=True
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        name = options["name"]
        domain = options["domain"]

        if schema == get_public_schema_name():
            raise CommandError("Schema 'public' is reserved")
        if Tenant.objects.filter(schema_name=schema).exists():
            raise CommandError("Schema already exists")
        if Domain.objects.filter(domain=domain).exists():
            raise CommandError("Domain already exists")

        with schema_context(get_public_schema_name()):
            with transaction.atomic():
                tenant = Tenant.objects.create(schema_name=schema, name=name)
                Domain.objects.create(domain=domain, tenant=tenant, is_primary=True)

        self.stdout.write(self.style.SUCCESS(f"Tenant '{name}' created"))
