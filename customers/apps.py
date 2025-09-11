from django.apps import AppConfig


class CustomersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "customers"

    def ready(self):  # pragma: no cover - runtime wiring
        """Compatibility shims for django-tenants management commands.

        Accepts our shorter flag names (schema/domain) when invoking the
        built-in django_tenants ``create_tenant`` command, so both our tests
        and CI can use a consistent interface regardless of command resolution.
        """
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
                # Add synonyms if not already defined
                opt_strings = {s for a in parser._actions for s in a.option_strings}
                if "--schema" not in opt_strings:
                    parser.add_argument("--schema", dest="schema_name")
                if "--domain" not in opt_strings:
                    parser.add_argument("--domain", dest="domain_domain")

            DTCreateTenantCommand.add_arguments = patched_add_arguments

            original_handle = DTCreateTenantCommand.handle

            def patched_handle(self, *args, **options):
                # If essential options are provided, delegate to our non-interactive
                # implementation to avoid input() prompts during tests/CI.
                schema = options.get("schema") or options.get("schema_name")
                domain = options.get("domain") or options.get("domain_domain")
                name = options.get("name")
                if schema and domain and name:
                    mapped = {"schema": schema, "domain": domain, "name": name}
                    return OurCreateTenantCommand().handle(*args, **mapped)
                return original_handle(self, *args, **options)

            DTCreateTenantCommand.handle = patched_handle
        except Exception:
            # If django_tenants isn't present or import fails, ignore.
            pass

        # Ensure our custom command takes precedence for 'create_tenant'
        try:
            from django.core import management

            management._commands = getattr(management, "_commands", {})
            management._commands["create_tenant"] = "customers"
        except Exception:
            pass
