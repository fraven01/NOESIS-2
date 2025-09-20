from collections.abc import Mapping

from django.conf import settings
from django.db import connection

from .logging import bind_log_context, clear_log_context


class TenantSchemaMiddleware:
    """Populate request.tenant_schema from the X-Tenant-Schema header."""

    header_name = "HTTP_X_TENANT_SCHEMA"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.tenant_schema = request.META.get(self.header_name)
        return self.get_response(request)


class HeaderTenantRoutingMiddleware:
    """
    In DEBUG/TESTING, allow routing the active tenant schema by header.

    This helps tests and local tools (Postman/cURL) exercise tenant-specific
    views without configuring domain records. In production (DEBUG=False), this
    middleware is inert and does not change schema routing.
    """

    header_name = "HTTP_X_TENANT_SCHEMA"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if settings.DEBUG or getattr(settings, "TESTING", False):
            from customers.models import Tenant

            schema = request.META.get(self.header_name)
            if schema and Tenant.objects.filter(schema_name=schema).exists():
                connection.set_schema(schema)
            elif getattr(settings, "TESTING", False):
                # Default to a known test schema when running tests and no header given
                if Tenant.objects.filter(schema_name="autotest").exists():
                    connection.set_schema("autotest")
        return self.get_response(request)


class RequestLogContextMiddleware:
    """Bind request metadata to the logging context for the request lifecycle."""

    _HEADER_MAP = {
        "trace_id": "HTTP_X_TRACE_ID",
        "case_id": "HTTP_X_CASE_ID",
        "tenant": "HTTP_X_TENANT_ID",
        "key_alias": "HTTP_X_KEY_ALIAS",
    }

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        clear_log_context()
        context = self._extract_context(request)
        if context:
            bind_log_context(**context)

        try:
            response = self.get_response(request)
        finally:
            clear_log_context()

        return response

    def _extract_context(self, request) -> dict[str, str]:
        context: dict[str, str] = {}
        meta = getattr(request, "META", {})

        for key, header in self._HEADER_MAP.items():
            value = meta.get(header)
            if value:
                context[key] = self._normalize(value)

        if "tenant" not in context:
            tenant = self._resolve_tenant(request)
            if tenant:
                context["tenant"] = tenant

        request_meta = getattr(request, "log_context", None)
        if isinstance(request_meta, Mapping):
            for key in ("trace_id", "case_id", "tenant", "key_alias"):
                value = request_meta.get(key)
                if value:
                    context[key] = self._normalize(value)

        return context

    def _resolve_tenant(self, request) -> str | None:
        tenant_obj = getattr(request, "tenant", None)
        if tenant_obj:
            schema = getattr(tenant_obj, "schema_name", None)
            if schema and not self._is_public_schema(schema):
                return schema

        schema = getattr(request, "tenant_schema", None)
        if schema and not self._is_public_schema(schema):
            return schema

        schema = getattr(connection, "schema_name", None)
        if schema and not self._is_public_schema(schema):
            return schema

        return None

    @staticmethod
    def _normalize(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @staticmethod
    def _is_public_schema(schema: str | None) -> bool:
        public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
        return schema == public_schema
