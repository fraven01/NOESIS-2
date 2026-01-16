"""Bootstrap dev users for a specific tenant."""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django_tenants.utils import schema_context
from customers.models import Tenant
from profiles.models import UserProfile
from profiles.services import ensure_user_profile


class Command(BaseCommand):
    help = "Ensure standard development users exist directly in a specific tenant."

    def add_arguments(self, parser):
        parser.add_argument(
            "--schema", required=True, help="Tenant schema name (e.g. 'dev')"
        )
        parser.add_argument(
            "--password", default="password", help="Default password for all dev users"
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        default_password = options["password"]

        if not Tenant.objects.filter(schema_name=schema).exists():
            # In bootstrap chain, we might run before tenant creation if ordered wrong,
            # but usually this runs after create_tenant.
            # If it doesn't exist, we can't create users in it.
            self.stdout.write(
                self.style.ERROR(
                    f"Tenant '{schema}' not found. Skipping user creation."
                )
            )
            return

        self.stdout.write(f"Bootstrapping dev users in tenant '{schema}'...")

        users_config = [
            {
                "username": "admin",
                "email": "admin@example.com",
                "first_name": "Admin",
                "last_name": "User",
                "is_staff": True,
                "is_superuser": True,
                "role": UserProfile.Roles.TENANT_ADMIN,
            },
            {
                "username": "dev",
                "email": "dev@example.com",
                "first_name": "Developer",
                "last_name": "Account",
                "is_staff": True,
                "is_superuser": False,
                # 'dev' is not a standard business role, so we treat as TENANT_ADMIN or similar for now,
                # strictly for "being a developer" inside the tenant.
                "role": UserProfile.Roles.TENANT_ADMIN,
            },
            {
                "username": "legal_bob",
                "email": "bob@example.com",
                "first_name": "Bob",
                "last_name": "Legal",
                "is_staff": False,
                "is_superuser": False,
                "role": UserProfile.Roles.LEGAL,
            },
            {
                "username": "alice_stakeholder",
                "email": "alice@example.com",
                "first_name": "Alice",
                "last_name": "Stakeholder",
                "is_staff": False,
                "is_superuser": False,
                "role": UserProfile.Roles.STAKEHOLDER,
            },
            {
                "username": "charles_external",
                "email": "charles@external.com",
                "first_name": "Charles",
                "last_name": "External",
                "is_staff": False,
                "is_superuser": False,
                "role": UserProfile.Roles.STAKEHOLDER,
                "account_type": UserProfile.AccountType.EXTERNAL,
            },
        ]

        User = get_user_model()

        with schema_context(schema):
            for config in users_config:
                username = config["username"]
                role = config["role"]
                account_type = config.get(
                    "account_type", UserProfile.AccountType.INTERNAL
                )

                user, created = User.objects.update_or_create(
                    username=username,
                    defaults={
                        "email": config["email"],
                        "first_name": config["first_name"],
                        "last_name": config["last_name"],
                        "is_staff": config["is_staff"],
                        "is_superuser": config["is_superuser"],
                        "is_active": True,
                    },
                )

                if created:
                    user.set_password(default_password)
                    user.save()
                    action = "Created"
                else:
                    # Only reset password if user was just created?
                    # For dev envs, resetting to known default is usually preferred to avoid lockouts.
                    user.set_password(default_password)
                    user.save()
                    action = "Updated"

                # Ensure Profile
                profile = ensure_user_profile(user)
                profile.role = role
                profile.account_type = account_type
                profile.is_active = True
                profile.save()

                self.stdout.write(f"  - {action} user '{username}' (Role: {role})")

        self.stdout.write(self.style.SUCCESS(f"✅ Dev users ensured in '{schema}'"))
