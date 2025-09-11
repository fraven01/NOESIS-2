from django.conf import settings
from django.core.management.base import BaseCommand
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant


class Command(BaseCommand):
    help = "Ensure the public tenant and primary domain exist."

    def add_arguments(self, parser):
        parser.add_argument("--domain", default="localhost")

    def handle(self, *args, **options):
        # Always operate on the public schema when creating the public tenant/domain
        with schema_context(get_public_schema_name()):
            tenant, _ = Tenant.objects.get_or_create(
                schema_name=settings.PUBLIC_SCHEMA_NAME, defaults={"name": "Public"}
            )
            Domain.objects.get_or_create(
                domain=options["domain"], tenant=tenant, defaults={"is_primary": True}
            )
