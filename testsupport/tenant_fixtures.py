"""Shared tenant lifecycle helpers for pytest-managed environments."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from django.core.management import call_command
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant


LOGGER = logging.getLogger(__name__)

DEFAULT_TEST_DOMAIN = "testserver"
_RESERVED_SCHEMAS = frozenset({"autotest", "test"})
_TRACKED_SCHEMAS: set[str] = set()


def _normalize_schema_name(schema_name: str | None) -> str | None:
    if not schema_name:
        return None
    normalized = str(schema_name).strip()
    return normalized or None


def _track_created_schema(schema_name: str | None) -> None:
    normalized = _normalize_schema_name(schema_name)
    if normalized and normalized not in _RESERVED_SCHEMAS:
        _TRACKED_SCHEMAS.add(normalized)


def _track_removed_schema(schema_name: str | None) -> None:
    normalized = _normalize_schema_name(schema_name)
    if normalized:
        _TRACKED_SCHEMAS.discard(normalized)


def bootstrap_tenant_schema(
    tenant: Tenant,
    *,
    migrate: bool = True,
) -> Tenant:
    """Ensure a tenant schema exists and migrations ran."""

    schema_name = tenant.schema_name
    if not schema_name:
        raise ValueError("tenant schema_name must be set")

    with schema_context(get_public_schema_name()):
        tenant.create_schema(check_if_exists=True)
        if migrate:
            call_command(
                "migrate_schemas",
                tenant=True,
                schema_name=schema_name,
                interactive=False,
                verbosity=0,
            )
    return tenant


def ensure_tenant_domain(
    tenant: Tenant,
    *,
    domain: str = DEFAULT_TEST_DOMAIN,
    is_primary: bool = True,
) -> Domain:
    """Attach (or reuse) a domain so middleware resolves the tenant for testserver."""

    with schema_context(get_public_schema_name()):
        domain_obj, _ = Domain.objects.get_or_create(
            domain=domain,
            defaults={"tenant": tenant, "is_primary": is_primary},
        )
        if domain_obj.tenant_id != tenant.id:
            # Defensive check: ensure tenant is actually persisted to avoid IntegrityError
            if not Tenant.objects.filter(id=tenant.id).exists():
                tenant.save()
            
            domain_obj.tenant = tenant
            domain_obj.is_primary = is_primary
            domain_obj.save(update_fields=["tenant", "is_primary"])
    return domain_obj


def create_test_tenant(
    *,
    schema_name: str,
    name: str | None = None,
    domain: str | None = None,
    migrate: bool = True,
    **fields,
) -> Tenant:
    """Create a tenant record, provision its schema and optionally attach a domain."""

    payload = {"schema_name": schema_name, "name": name or f"Tenant {schema_name}"}
    payload.update(fields)
    with schema_context(get_public_schema_name()):
        tenant = Tenant.objects.create(**payload)
    bootstrap_tenant_schema(tenant, migrate=migrate)
    if domain:
        ensure_tenant_domain(tenant, domain=domain)
    return tenant


def cleanup_test_tenants(*, preserve: Iterable[str] | None = None) -> list[str]:
    """Drop schemas for tenants created during a test run."""

    preserved = _RESERVED_SCHEMAS.union(
        {_normalize_schema_name(entry) for entry in preserve or () if entry}
    )
    targets = sorted(schema for schema in _TRACKED_SCHEMAS if schema not in preserved)
    if not targets:
        return []

    removed: list[str] = []
    with schema_context(get_public_schema_name()):
        existing: Mapping[str, Tenant] = {
            tenant.schema_name: tenant
            for tenant in Tenant.objects.filter(schema_name__in=targets)
        }
        for schema_name in targets:
            tenant = existing.get(schema_name)
            if tenant is None:
                _track_removed_schema(schema_name)
                continue
            try:
                tenant.delete(force_drop=True)
            except TypeError:
                tenant.delete()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.warning(
                    "testsupport.tenant_cleanup_failed",
                    extra={"tenant.schema": schema_name},
                    exc_info=True,
                )
                continue
            removed.append(schema_name)
    return removed


@receiver(post_save, sender=Tenant, weak=False)
def _on_tenant_created(sender, instance: Tenant, created: bool, **_kwargs) -> None:
    if created:
        _track_created_schema(instance.schema_name)


@receiver(post_delete, sender=Tenant, weak=False)
def _on_tenant_deleted(sender, instance: Tenant, **_kwargs) -> None:
    _track_removed_schema(instance.schema_name)


__all__ = [
    "DEFAULT_TEST_DOMAIN",
    "bootstrap_tenant_schema",
    "cleanup_test_tenants",
    "create_test_tenant",
    "ensure_tenant_domain",
]
