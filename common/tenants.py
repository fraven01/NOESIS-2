from functools import wraps

from django.db import connection
from django.http import HttpResponseForbidden

from customers.tenant_context import TenantContext


def get_current_tenant():
    """Return the tenant for the current connection's schema."""
    schema = getattr(connection, "schema_name", None)
    return TenantContext.resolve_identifier(schema, allow_pk=True)


def _is_valid_tenant_request(request):
    """Return the resolved tenant and optional error when validation fails."""

    tenant = TenantContext.from_request(
        request, allow_headers=False, require=False
    )
    if tenant is None:
        return None, "Tenant context is not active for this request"

    tenant_schema = getattr(request, "tenant_schema", None)
    if not tenant_schema:
        return None, "Tenant schema header missing"

    if tenant_schema != tenant.schema_name:
        return None, "Tenant schema does not match resolved tenant"

    return tenant, None


def tenant_schema_required(view_func):
    """Decorator ensuring the request matches the active tenant schema."""

    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        tenant, error = _is_valid_tenant_request(request)
        if tenant is None:
            return HttpResponseForbidden(error)
        return view_func(request, *args, **kwargs)

    return _wrapped


class TenantSchemaRequiredMixin:
    """Mixin validating the active tenant schema for class-based views."""

    def dispatch(self, request, *args, **kwargs):
        tenant, error = _is_valid_tenant_request(request)
        if tenant is None:
            return HttpResponseForbidden(error)
        return super().dispatch(request, *args, **kwargs)
