from django.core.management.base import BaseCommand

from customers.models import Tenant


class Command(BaseCommand):
    help = "List all tenants"

    def handle(self, *args, **options):
        for tenant in Tenant.objects.all():
            self.stdout.write(f"{tenant.schema_name}: {tenant.name}")
