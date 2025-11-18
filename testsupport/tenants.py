from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set

from django.core.management import call_command
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant


def ensure_tenant(schema_name: str, **fields) -> Tenant:
    """Return an existing tenant or create one with a fully migrated schema."""

    if not schema_name:
        raise ValueError("schema_name is required")

    payload = {
        "schema_name": schema_name,
        "name": fields.pop("name", f"Tenant {schema_name}"),
    }
    payload.update(fields)
    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        tenant, created = Tenant.objects.get_or_create(
            schema_name=schema_name,
            defaults=payload,
        )
        if not created:
            updates = {
                key: value
                for key, value in payload.items()
                if getattr(tenant, key) != value
            }
            if updates:
                for key, value in updates.items():
                    setattr(tenant, key, value)
                tenant.save(update_fields=list(updates.keys()))
        tenant.create_schema(check_if_exists=True)
        call_command(
            "migrate_schemas",
            tenant=True,
            schema_name=tenant.schema_name,
            interactive=False,
            verbosity=0,
        )
        domain_obj, _created = Domain.objects.get_or_create(
            domain="testserver",
            defaults={"tenant": tenant, "is_primary": True},
        )
        if domain_obj.tenant_id != tenant.id:
            domain_obj.tenant = tenant
            domain_obj.is_primary = True
            domain_obj.save(update_fields=["tenant", "is_primary"])
    return tenant


@dataclass
class TenantFactoryHelper:
    create_func: Callable[..., Tenant] = ensure_tenant
    _schemas: Set[str] = field(default_factory=set)

    def create(self, schema_name: str, **fields) -> Tenant:
        tenant = self.create_func(schema_name, **fields)
        self._schemas.add(tenant.schema_name)
        return tenant

    def cleanup(self) -> None:
        if not self._schemas:
            return
        public_schema = get_public_schema_name()
        with schema_context(public_schema):
            for schema in list(self._schemas):
                try:
                    tenant = Tenant.objects.get(schema_name=schema)
                except Tenant.DoesNotExist:
                    continue
                try:
                    tenant.delete(force_drop=True)
                except TypeError:
                    tenant.delete()
        self._schemas.clear()


__all__ = ["ensure_tenant", "TenantFactoryHelper"]
