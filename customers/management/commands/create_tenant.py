from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from customers.models import Domain, Tenant


class Command(BaseCommand):
    help = "Create a tenant and associated domain"

    def add_arguments(self, parser):
        parser.add_argument("--schema", required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--domain", required=True)

    def handle(self, *args, **options):
        schema = options["schema"]
        name = options["name"]
        domain = options["domain"]

        if schema == settings.PUBLIC_SCHEMA_NAME:
            raise CommandError("Schema cannot be the public schema")

        if Tenant.objects.filter(schema_name=schema).exists():
            raise CommandError("Schema already exists")

        if Domain.objects.filter(domain=domain).exists():
            raise CommandError("Domain already exists")

        with transaction.atomic():
            tenant = Tenant.objects.create(schema_name=schema, name=name)
            Domain.objects.create(domain=domain, tenant=tenant, is_primary=True)
        self.stdout.write(self.style.SUCCESS(f"Tenant '{name}' created"))
