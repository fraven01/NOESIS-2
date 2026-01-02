"""Shared tenant lifecycle helpers for pytest-managed environments."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable

from django.core.management import call_command
from django.db import connection
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant

try:  # Optional import for isolated autocommit cleanup connections.
    from psycopg2 import sql as pg_sql
    import psycopg2
except Exception:  # pragma: no cover - optional dependency path
    pg_sql = None
    psycopg2 = None


LOGGER = logging.getLogger(__name__)

DEFAULT_TEST_DOMAIN = "testserver"
_BASE_RESERVED_SCHEMAS = frozenset({"autotest", "test"})
_TRACKED_SCHEMAS: set[str] = set()
_MIGRATED_SCHEMAS: set[str] = set()


@contextmanager
def _advisory_lock(key: str):
    """Serialize creation of shared test tenants/domains across workers."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_lock(hashtext(%s))", [key])
    try:
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(hashtext(%s))", [key])


def _normalize_schema_name(schema_name: str | None) -> str | None:
    if not schema_name:
        return None
    normalized = str(schema_name).strip()
    return normalized or None


def _reserved_schemas() -> frozenset[str]:
    reserved = set(_BASE_RESERVED_SCHEMAS)
    try:
        from django.conf import settings

        public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", None)
        normalized_public = _normalize_schema_name(public_schema)
        if normalized_public:
            reserved.add(normalized_public)

        test_schema = getattr(settings, "TEST_TENANT_SCHEMA", None)
        normalized = _normalize_schema_name(test_schema)
        if normalized:
            reserved.add(normalized)
    except Exception:
        pass
    return frozenset(schema for schema in reserved if schema)


def _track_created_schema(schema_name: str | None) -> None:
    normalized = _normalize_schema_name(schema_name)
    if normalized and normalized not in _reserved_schemas():
        _TRACKED_SCHEMAS.add(normalized)


def _track_removed_schema(schema_name: str | None) -> None:
    normalized = _normalize_schema_name(schema_name)
    if normalized:
        _TRACKED_SCHEMAS.discard(normalized)
        _MIGRATED_SCHEMAS.discard(normalized)


@contextmanager
def _autocommit_cursor():
    """Yield a cursor with autocommit enabled, preferably on a fresh connection."""
    if psycopg2 is not None:
        try:
            params = connection.get_connection_params()
        except Exception:
            params = None
        if params:
            ddl_conn = None
            try:
                ddl_conn = psycopg2.connect(**params)
                ddl_conn.set_session(autocommit=True)
                with ddl_conn.cursor() as cursor:
                    yield cursor
                return
            finally:
                if ddl_conn is not None:
                    try:
                        ddl_conn.close()
                    except Exception:
                        pass

    conn = connection
    autocommit = conn.get_autocommit()
    if not autocommit:
        try:
            conn.set_autocommit(True)
        except Exception:
            pass
    try:
        with conn.cursor() as cursor:
            yield cursor
    finally:
        if not autocommit:
            try:
                conn.set_autocommit(autocommit)
            except Exception:
                pass


def _drop_schema(cursor, schema_name: str) -> None:
    if pg_sql is not None:
        cursor.execute(
            pg_sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                pg_sql.Identifier(schema_name)
            )
        )
        return
    quoted = connection.ops.quote_name(schema_name)
    cursor.execute(f"DROP SCHEMA IF EXISTS {quoted} CASCADE")


def bootstrap_tenant_schema(
    tenant: Tenant,
    *,
    migrate: bool = True,
) -> Tenant:
    """Ensure a tenant schema exists and migrations ran."""

    schema_name = tenant.schema_name
    if not schema_name:
        raise ValueError("tenant schema_name must be set")

    normalized = _normalize_schema_name(schema_name)

    # Skip if already migrated this session
    if normalized and normalized in _MIGRATED_SCHEMAS:
        LOGGER.debug(
            "testsupport.schema_already_migrated",
            extra={"tenant.schema": schema_name},
        )
        return tenant

    with schema_context(get_public_schema_name()):
        # Step 1: Ensure PostgreSQL schema exists
        LOGGER.info(
            "testsupport.creating_schema",
            extra={"tenant.schema": schema_name},
        )
        # Create the PostgreSQL schema directly if it doesn't exist
        # This is necessary when auto_create_schema=False
        with connection.cursor() as cursor:
            if pg_sql is not None:
                cursor.execute(
                    pg_sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        pg_sql.Identifier(schema_name)
                    )
                )
            else:
                quoted = connection.ops.quote_name(schema_name)
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")

        # Step 2: Run migrations if needed
        if migrate:
            LOGGER.info(
                "testsupport.running_migrations",
                extra={"tenant.schema": schema_name},
            )
            try:
                call_command(
                    "migrate_schemas",
                    tenant=True,
                    schema_name=schema_name,
                    interactive=False,
                    verbosity=1,  # Increased from 0 to see migration output
                )
                LOGGER.info(
                    "testsupport.migrations_completed",
                    extra={"tenant.schema": schema_name},
                )
            except Exception as e:
                LOGGER.error(
                    "testsupport.migration_failed",
                    extra={"tenant.schema": schema_name, "error": str(e)},
                    exc_info=True,
                )
                # Don't mark as migrated if it failed
                raise e

            # Track as migrated only after successful completion
            if normalized:
                _MIGRATED_SCHEMAS.add(normalized)
                LOGGER.debug(
                    "testsupport.schema_marked_migrated",
                    extra={"tenant.schema": schema_name},
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
        with _advisory_lock(f"domain:{domain}"):
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

    preserved = _reserved_schemas().union(
        {_normalize_schema_name(entry) for entry in preserve or () if entry}
    )
    targets = sorted(schema for schema in _TRACKED_SCHEMAS if schema not in preserved)
    if not targets:
        return []

    removed: list[str] = []
    try:
        with _autocommit_cursor() as cursor:
            try:
                public_schema = get_public_schema_name()
                if pg_sql is not None:
                    cursor.execute(
                        pg_sql.SQL("SET search_path TO {}").format(
                            pg_sql.Identifier(public_schema)
                        )
                    )
                else:
                    cursor.execute(
                        f"SET search_path TO {connection.ops.quote_name(public_schema)}"
                    )
            except Exception:
                pass
            for schema_name in targets:
                try:
                    _drop_schema(cursor, schema_name)
                    cursor.execute(
                        "DELETE FROM customers_tenant WHERE schema_name = %s",
                        [schema_name],
                    )
                except Exception:  # pragma: no cover - defensive cleanup
                    LOGGER.warning(
                        "testsupport.tenant_cleanup_failed",
                        extra={"tenant.schema": schema_name},
                        exc_info=True,
                    )
                    try:
                        cursor.connection.rollback()
                    except Exception:
                        pass
                    continue
                _track_removed_schema(schema_name)
                removed.append(schema_name)
    except Exception:  # pragma: no cover - defensive cleanup
        LOGGER.warning("testsupport.tenant_cleanup_failed", exc_info=True)
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
