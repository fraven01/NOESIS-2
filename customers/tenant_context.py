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


class TenantRequiredError(RuntimeError):
    """Raised when tenant resolution is required but unsuccessful."""


class TenantContext:
    """Canonical helper for resolving tenant context/identity.

    This helper standardizes the resolution of tenants across the platform,
    enforcing the rule that `schema_name` is the primary identifier (authority string),
    with a fallback to numeric PKs only for legacy fixtures/internal tools.
    """

    _CACHE_MISS = object()

    @staticmethod
    def resolve_identifier(
        identifier: str | int | None, *, allow_pk: bool = False
    ) -> Tenant | None:
        """Resolve a Tenant object from a schema identifier or optional PK.

        Args:
            identifier: The tenant identifier (schema string or numeric PK).
            allow_pk: When ``True``, also attempt lookup by primary key.

        Returns:
            The Tenant object if found, else None.
        """
        if identifier is None:
            return None

        raw_value = str(identifier).strip()
        if not raw_value:
            return None

        try:
            return Tenant.objects.get(schema_name=raw_value)
        except Tenant.DoesNotExist:
            pass

        if allow_pk and raw_value.isdigit():
            try:
                tenant = Tenant.objects.get(pk=raw_value)
                log.warning(
                    "tenant_resolution_pk_lookup",
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
    def from_request(
        request: Any, *, allow_headers: bool = True, require: bool = True
    ) -> Tenant | None:
        """Resolve the tenant for the given request.

        Resolution order (first successful wins):
        1) ``request.tenant`` (django-tenants middleware)
        2) ``connection.schema_name`` when set to a non-public schema (resolved
           via ``TenantContext.resolve_identifier(..., allow_pk=True)`` to
           support CLI/fixtures)
        3) Explicit headers (``X-Tenant-Schema`` then ``X-Tenant-ID``) when ``allow_headers``

        The resolved tenant is cached on ``request._tenant_context_cache`` to avoid
        repeated lookups. When ``require`` is ``True`` and no tenant is found, a
        ``TenantRequiredError`` is raised.
        """

        if request is None:
            if require:
                raise TenantRequiredError("Tenant could not be resolved from request")
            return None

        cached = getattr(request, "_tenant_context_cache", TenantContext._CACHE_MISS)
        if cached is not TenantContext._CACHE_MISS:
            if cached is None and require:
                raise TenantRequiredError("Tenant could not be resolved from request")
            return cached

        tenant: Tenant | None = None

        tenant_obj = getattr(request, "tenant", None)
        if isinstance(tenant_obj, Tenant):
            tenant = tenant_obj

        if tenant is None:
            schema_name = getattr(connection, "schema_name", None)
            public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
            if schema_name and schema_name != public_schema:
                tenant = TenantContext.resolve_identifier(
                    schema_name, allow_pk=True
                )

        if tenant is None and allow_headers:
            headers = getattr(request, "headers", {}) or {}
            meta = getattr(request, "META", {}) or {}

            def _get(header_key: str, meta_key: str) -> str | None:
                value = headers.get(header_key)
                if value is None:
                    value = meta.get(meta_key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                return None

            schema_header = _get(X_TENANT_SCHEMA_HEADER, META_TENANT_SCHEMA_KEY)
            if schema_header:
                tenant = TenantContext.resolve_identifier(schema_header)

            if tenant is None:
                id_header = _get(X_TENANT_ID_HEADER, META_TENANT_ID_KEY)
                if id_header:
                    tenant = TenantContext.resolve_identifier(id_header)

        setattr(request, "_tenant_context_cache", tenant)

        if tenant is None and require:
            raise TenantRequiredError("Tenant could not be resolved from request")

        return tenant
