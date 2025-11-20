from __future__ import annotations

from typing import Any

from django.conf import settings
from django.db import connection

from common.logging import get_logger
from common.constants import (
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from customers.models import Tenant

log = get_logger(__name__)


class TenantContext:
    """Canonical helper for resolving tenant context/identity.

    This helper standardizes the resolution of tenants across the platform,
    enforcing the rule that `schema_name` is the primary identifier (authority string),
    with a fallback to numeric PKs only for legacy fixtures/internal tools.
    """

    @staticmethod
    def resolve(identifier: str | int | None) -> Tenant | None:
        """Resolve a Tenant object from a string identifier (schema_name) or PK.

        Resolution order:
        1. Lookup by schema_name (canonical string).
        2. Lookup by PK (legacy fallback).

        Args:
            identifier: The tenant identifier (schema string or numeric PK).

        Returns:
            The Tenant object if found, else None.
        """
        if identifier is None:
            return None

        # Normalize identifier
        raw_value = str(identifier).strip()
        if not raw_value:
            return None

        # 1. Try schema_name (Canonical)
        try:
            return Tenant.objects.get(schema_name=raw_value)
        except Tenant.DoesNotExist:
            pass

        # 2. Try PK (Legacy Fallback)
        # We only try this if the string looks like an integer to avoid
        # unnecessary DB queries for obviously non-numeric schema names.
        if raw_value.isdigit():
            try:
                tenant = Tenant.objects.get(pk=raw_value)
                log.warning(
                    "tenant_resolution_legacy_pk_fallback",
                    extra={
                        "identifier": raw_value,
                        "tenant_schema": tenant.schema_name,
                    },
                )
                return tenant
            except (Tenant.DoesNotExist, ValueError):
                pass

        return None

    @staticmethod
    def from_headers(request: Any) -> Tenant | None:
        """Resolve the tenant from HTTP request headers/metadata.

        Prioritizes X-Tenant-Schema, then X-Tenant-ID.
        Also checks django-tenants request.tenant and connection.schema_name.

        Args:
            request: The HTTP request object (or similar object with headers/META).

        Returns:
            The resolved Tenant object or None.
        """
        # 1. Check explicit headers
        # We prefer X-Tenant-Schema as the specific schema identifier,
        # but X-Tenant-ID is often used as the logical ID (which should be the schema).
        headers = getattr(request, "headers", {}) or {}
        meta = getattr(request, "META", {}) or {}

        # Helper to look in headers then META
        def _get(header_key: str, meta_key: str) -> str | None:
            val = headers.get(header_key)
            if val is None:
                val = meta.get(meta_key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            return None

        # Try explicit schema header first
        schema_header = _get(X_TENANT_SCHEMA_HEADER, META_TENANT_SCHEMA_KEY)
        if schema_header:
            tenant = TenantContext.resolve(schema_header)
            if tenant:
                return tenant

        # Try generic tenant ID header
        id_header = _get(X_TENANT_ID_HEADER, META_TENANT_ID_KEY)
        if id_header:
            tenant = TenantContext.resolve(id_header)
            if tenant:
                return tenant

        # 2. Check django-tenants context
        # If the request has already been processed by TenantMiddleware, it might have .tenant
        tenant_obj = getattr(request, "tenant", None)
        if isinstance(tenant_obj, Tenant):
            return tenant_obj

        # 3. Check connection schema (last resort, context-bound)
        schema_name = getattr(connection, "schema_name", None)
        public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
        if schema_name and schema_name != public_schema:
            try:
                return Tenant.objects.get(schema_name=schema_name)
            except Tenant.DoesNotExist:
                pass

        return None
