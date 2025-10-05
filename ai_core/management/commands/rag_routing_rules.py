"""Inspect and dry-run embedding routing rules."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from ai_core.rag.profile_resolver import (
    ProfileResolverError,
    resolve_embedding_profile,
)
from ai_core.rag.routing_rules import (
    RoutingConfigurationError,
    get_routing_table,
    reset_routing_rules_cache,
)
from ai_core.rag.selector_utils import normalise_selector_value


class Command(BaseCommand):
    help = (
        "Validate embedding routing rules and optionally resolve a selector "
        "against the configured table."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant",
            help="Tenant identifier for dry-run profile resolution",
        )
        parser.add_argument(
            "--process",
            help="Process label for dry-run profile resolution",
        )
        parser.add_argument(
            "--doc-class",
            help="Document class label for dry-run profile resolution",
        )
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Reload routing rules from disk before validation",
        )

    def handle(self, *args, **options):
        if options.get("refresh"):
            reset_routing_rules_cache()

        try:
            table = get_routing_table()
        except RoutingConfigurationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write("Default profile: %s" % table.default_profile)
        if table.rules:
            self.stdout.write("Overrides:")
            for rule in table.rules:
                tenant = rule.tenant or "*"
                process = rule.process or "*"
                doc_class = rule.doc_class or "*"
                self.stdout.write(
                    "  - tenant=%s, process=%s, doc_class=%s -> %s"
                    % (tenant, process, doc_class, rule.profile)
                )
        else:
            self.stdout.write("No override rules configured.")

        tenant_option = options.get("tenant")
        process_option = options.get("process")
        doc_class_option = options.get("doc_class")

        if tenant_option is None:
            if process_option is not None or doc_class_option is not None:
                raise CommandError(
                    "--tenant is required when using --process or --doc-class"
                )
            return

        tenant = str(tenant_option).strip()
        if not tenant:
            raise CommandError("--tenant must be a non-empty string")

        sanitized_process = normalise_selector_value(process_option)
        sanitized_doc_class = normalise_selector_value(doc_class_option)

        try:
            profile_id = resolve_embedding_profile(
                tenant_id=tenant,
                process=process_option,
                doc_class=doc_class_option,
            )
        except ProfileResolverError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                "Resolved selector tenant=%s, process=%s, doc_class=%s -> %s"
                % (
                    tenant,
                    sanitized_process or "*",
                    sanitized_doc_class or "*",
                    profile_id,
                )
            )
        )
