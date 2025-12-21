"""Bootstrap default cases for tenants."""

from __future__ import annotations

import os

from django.core.management.base import BaseCommand
from django_tenants.utils import schema_context

from common.logging import get_logger
from customers.models import Tenant
from cases.services import ensure_case
from cases.models import Case

logger = get_logger(__name__)


class Command(BaseCommand):
    """Ensure default cases exist for tenants."""

    help = "Ensure default cases exist for tenants"

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant",
            type=str,
            help="Specific tenant schema name (default: all tenants)",
        )
        parser.add_argument(
            "--include-dev-case",
            action="store_true",
            help="Also create the dev default case for the dev tenant",
        )
        parser.add_argument(
            "--dev-tenant",
            type=str,
            default=os.getenv("DEV_TENANT_SCHEMA", "dev"),
            help="Tenant schema for dev defaults (default: DEV_TENANT_SCHEMA or 'dev')",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without actually creating",
        )

    def handle(self, *args, **options):
        tenant_filter = options.get("tenant")
        include_dev_case = options.get("include_dev_case", False)
        dev_tenant = options.get("dev_tenant")
        dry_run = options.get("dry_run", False)

        if tenant_filter:
            tenants = Tenant.objects.filter(schema_name=tenant_filter)
            if not tenants.exists():
                self.stdout.write(
                    self.style.ERROR(f"Tenant '{tenant_filter}' not found")
                )
                return
        else:
            tenants = Tenant.objects.all()

        total_tenants = tenants.count()
        created_cases = 0
        existing_cases = 0
        error_count = 0

        self.stdout.write(
            self.style.SUCCESS(
                f"Bootstrapping default cases for {total_tenants} tenant(s)..."
            )
        )

        for tenant in tenants:
            self.stdout.write(f"\nProcessing tenant: {tenant.schema_name}")
            with schema_context(tenant.schema_name):
                try:
                    general_exists = Case.objects.filter(
                        tenant=tenant, external_id="general"
                    ).exists()
                    if dry_run:
                        self.stdout.write(
                            self.style.WARNING(
                                f"  [DRY RUN] Would ensure 'general' case for {tenant.schema_name}"
                            )
                        )
                    else:
                        ensure_case(
                            tenant,
                            "general",
                            title="General",
                            reopen_closed=True,
                        )
                        if general_exists:
                            existing_cases += 1
                        else:
                            created_cases += 1

                    if include_dev_case and tenant.schema_name == dev_tenant:
                        dev_exists = Case.objects.filter(
                            tenant=tenant, external_id="dev-case-local"
                        ).exists()
                        if dry_run:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"  [DRY RUN] Would ensure 'dev-case-local' for {tenant.schema_name}"
                                )
                            )
                        else:
                            ensure_case(
                                tenant,
                                "dev-case-local",
                                title="Dev Local",
                                reopen_closed=True,
                            )
                            if dev_exists:
                                existing_cases += 1
                            else:
                                created_cases += 1

                    logger.info(
                        "bootstrap_default_cases.processed",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "include_dev_case": include_dev_case,
                            "dev_tenant": dev_tenant,
                        },
                    )
                except Exception as exc:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(f"  ❌ Failed to bootstrap cases: {str(exc)}")
                    )
                    logger.exception(
                        "bootstrap_default_cases.failed",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "error": str(exc),
                        },
                    )

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("Bootstrap Summary:"))
        self.stdout.write(f"  Total tenants processed: {total_tenants}")
        self.stdout.write(self.style.SUCCESS(f"  Cases created: {created_cases}"))
        self.stdout.write(
            self.style.WARNING(f"  Cases already existed: {existing_cases}")
        )
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f"  Errors: {error_count}"))
        self.stdout.write("=" * 80)

        if dry_run:
            self.stdout.write(
                self.style.WARNING("\n⚠️  This was a DRY RUN - no changes were made")
            )
