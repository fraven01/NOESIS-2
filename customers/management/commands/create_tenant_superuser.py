from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.db import OperationalError
from django_tenants.utils import schema_context

from customers.models import Tenant


class Command(BaseCommand):
    help = "Create a Django superuser inside a specific tenant schema."

    def add_arguments(self, parser):
        parser.add_argument("--schema", required=True, help="Tenant schema name")
        parser.add_argument("--username", help="Username (optional)")
        parser.add_argument("--email", help="Email (optional)")
        parser.add_argument(
            "--password",
            help="Password (optional; if omitted, runs interactive createsuperuser)",
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        username = options.get("username")
        email = options.get("email") or ""
        password = options.get("password")

        if not Tenant.objects.filter(schema_name=schema).exists():
            raise CommandError(f"Tenant schema '{schema}' does not exist.")

        call_command("migrate_schemas", schema=schema)

        with schema_context(schema):
            try:
                if username and password:
                    User = get_user_model()
                    if User.objects.filter(username=username).exists():
                        self.stdout.write(
                            self.style.WARNING(
                                f"User '{username}' already exists in schema '{schema}'."
                            )
                        )
                        return
                    User.objects.create_superuser(
                        username=username, email=email, password=password
                    )
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Superuser '{username}' created in schema '{schema}'."
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            "Running interactive createsuperuser inside tenant schema."
                        )
                    )
                    # Delegate to Django's interactive command within the tenant schema
                    call_command("createsuperuser")
            except OperationalError as exc:
                raise CommandError(
                    (
                        "Database schema for tenant "
                        f"'{schema}' is not migrated. Run 'python manage.py "
                        f"migrate_schemas --schema={schema}' before creating the superuser."
                    )
                ) from exc
