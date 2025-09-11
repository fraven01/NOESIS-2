from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from customers.models import Domain, Tenant


class Command(BaseCommand):
    help = "Add or ensure a domain for a tenant (optionally set as primary)."

    def add_arguments(self, parser):
        parser.add_argument("--schema", required=True, help="Tenant schema name")
        parser.add_argument("--domain", required=True, help="Domain/host to map")
        parser.add_argument(
            "--primary",
            action="store_true",
            help="Set this domain as the tenant's primary domain",
        )
        parser.add_argument(
            "--force-reassign",
            action="store_true",
            help=(
                "If the domain is assigned to another tenant, reassign it to the"
                " target tenant instead of failing"
            ),
        )

    def handle(self, *args, **options):
        schema = options["schema"]
        domain_value = options["domain"].strip()
        make_primary = options["primary"]
        force = options["force_reassign"]

        try:
            tenant = Tenant.objects.get(schema_name=schema)
        except Tenant.DoesNotExist:
            raise CommandError(f"Tenant schema '{schema}' does not exist")

        with transaction.atomic():
            domain_obj, created = Domain.objects.get_or_create(
                domain=domain_value, defaults={"tenant": tenant}
            )
            if not created and domain_obj.tenant_id != tenant.id:
                if not force:
                    raise CommandError(
                        f"Domain '{domain_value}' already assigned to a different tenant"
                    )
                # Reassign the domain to the requested tenant
                domain_obj.tenant = tenant
                domain_obj.save(update_fields=["tenant"])
            # Ensure tenant is set (covers the get_or_create non-created path with missing tenant)
            if domain_obj.tenant_id != tenant.id:
                domain_obj.tenant = tenant
                domain_obj.save(update_fields=["tenant"])

            if make_primary and not domain_obj.is_primary:
                # Unset previous primary for this tenant, then set the new primary
                Domain.objects.filter(tenant=tenant, is_primary=True).exclude(
                    pk=domain_obj.pk
                ).update(is_primary=False)
                domain_obj.is_primary = True
                domain_obj.save(update_fields=["is_primary"])

        action = "created" if created else "ensured"
        suffix = " (set as primary)" if make_primary else ""
        self.stdout.write(
            self.style.SUCCESS(
                f"Domain '{domain_value}' {action} for tenant '{schema}'{suffix}."
            )
        )
