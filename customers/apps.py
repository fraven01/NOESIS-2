from django.apps import AppConfig


class CustomersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "customers"

    def ready(self):  # pragma: no cover - runtime wiring
        """Compatibility shims for django-tenants management commands."""
        try:
            from django_tenants.management.commands.create_tenant import (
                Command as DTCreateTenantCommand,
            )
            from customers.management.commands.create_tenant import (
                Command as OurCreateTenantCommand,
            )

            original_add_arguments = DTCreateTenantCommand.add_arguments

            def patched_add_arguments(self, parser):
                original_add_arguments(self, parser)
                opt_strings = {s for a in parser._actions for s in a.option_strings}
                if "--schema" not in opt_strings:
                    parser.add_argument("--schema", dest="schema_name")
                if "--domain" not in opt_strings:
                    parser.add_argument("--domain", dest="domain_domain")

            DTCreateTenantCommand.add_arguments = patched_add_arguments

            original_handle = DTCreateTenantCommand.handle

            def patched_handle(self, *args, **options):
                schema = options.get("schema") or options.get("schema_name")
                domain = options.get("domain") or options.get("domain_domain")
                name = options.get("name")
                if schema and domain and name:
                    mapped = {"schema": schema, "domain": domain, "name": name}
                    return OurCreateTenantCommand().handle(*args, **mapped)
                return original_handle(self, *args, **options)

            DTCreateTenantCommand.handle = patched_handle
        except Exception:
            pass

        try:
            from django.core import management
        except Exception:
            return

        if getattr(management, "_customers_commands_patched", False):
            return

        original_get_commands = management.get_commands

        def patched_get_commands():
            commands = dict(original_get_commands())
            commands["create_tenant"] = "customers"
            commands["create_tenant_superuser"] = "customers"
            return commands

        if hasattr(original_get_commands, "cache_clear"):
            patched_get_commands.cache_clear = original_get_commands.cache_clear  # type: ignore[attr-defined]

        management.get_commands = patched_get_commands

        if hasattr(original_get_commands, "cache_clear"):
            original_get_commands.cache_clear()

        management._customers_commands_patched = True
