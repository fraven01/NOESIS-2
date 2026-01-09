from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set

from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Tenant
from testsupport.tenant_fixtures import (
    _advisory_lock,
    bootstrap_tenant_schema,
    ensure_tenant_domain,
)


def ensure_tenant(schema_name: str, *, migrate: bool = True, **fields) -> Tenant:
    """Return an existing tenant or create one with an optionally migrated schema."""

    if not schema_name:
        raise ValueError("schema_name is required")

    domain = fields.pop("domain", "testserver")
    payload = {
        "schema_name": schema_name,
        "name": fields.pop("name", f"Tenant {schema_name}"),
    }
    payload.update(fields)
    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        with _advisory_lock(f"tenant:{schema_name}"):
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
    bootstrap_tenant_schema(tenant, migrate=migrate)
    if domain:
        ensure_tenant_domain(tenant, domain=domain)
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
                except Exception:
                    # Fallback if deletion fails (e.g. active connections)
                    pass
        self._schemas.clear()


__all__ = ["ensure_tenant", "TenantFactoryHelper"]
