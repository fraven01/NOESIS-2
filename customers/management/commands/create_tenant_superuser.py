import os

from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.db import OperationalError
from django_tenants.utils import schema_context

from customers.models import Tenant


class Command(BaseCommand):
    help = "Create or ensure a Django superuser inside a specific tenant schema (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument("--schema", required=True, help="Tenant schema name")
        parser.add_argument("--username", help="Username")
        parser.add_argument("--email", help="Email")
        parser.add_argument("--password", help="Password")
        # Accept but ignore --noinput to be compatible with CI calls
        parser.add_argument(
            "--noinput", action="store_true", help="No interactive prompts"
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        # Prefer CLI args; fall back to standard Django env vars used in CI
        username = options.get("username") or os.getenv("DJANGO_SUPERUSER_USERNAME")
        email = options.get("email") or os.getenv("DJANGO_SUPERUSER_EMAIL") or ""
        password = options.get("password") or os.getenv("DJANGO_SUPERUSER_PASSWORD")

        if not Tenant.objects.filter(schema_name=schema).exists():
            raise CommandError(f"Tenant schema '{schema}' does not exist.")

        # Ensure tenant schema is migrated
        call_command("migrate_schemas", schema=schema)

        with schema_context(schema):
            try:
                User = get_user_model()

                if not username:
                    raise CommandError(
                        "Missing username. Provide --username or DJANGO_SUPERUSER_USERNAME."
                    )

                qs = User.objects.filter(username=username)
                user = getattr(qs, "first", lambda: None)()

                if user:
                    # Ensure superuser/staff flags and optionally reset password
                    updated = False
                    if not user.is_staff:
                        user.is_staff = True
                        updated = True
                    if not user.is_superuser:
                        user.is_superuser = True
                        updated = True
                    if password:
                        user.set_password(password)
                        updated = True
                    if updated:
                        user.save()
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"Superuser '{username}' ensured (updated) in schema '{schema}'."
                            )
                        )
                    else:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Superuser '{username}' already exists in schema '{schema}'."
                            )
                        )
                    return

                if not password:
                    # Do not fall back to interactive in jobs; be explicit
                    raise CommandError(
                        "Missing password. Provide --password or DJANGO_SUPERUSER_PASSWORD."
                    )

                User.objects.create_superuser(
                    username=username, email=email, password=password
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Superuser '{username}' created in schema '{schema}'."
                    )
                )
            except OperationalError as exc:
                raise CommandError(
                    (
                        "Database schema for tenant "
                        f"'{schema}' is not migrated. Run 'python manage.py "
                        f"migrate_schemas --schema={schema}' before creating the superuser."
                    )
                ) from exc
