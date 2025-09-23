from functools import wraps

from django.db import connection
from django.http import HttpResponseForbidden

from customers.models import Tenant


def get_current_tenant():
    """Return the tenant for the current connection's schema."""
    schema = connection.schema_name
    try:
        return Tenant.objects.get(schema_name=schema)
    except Tenant.DoesNotExist:
        return None


def _is_valid_tenant_request(request):
    """Return whether the request targets the active tenant schema."""
    tenant = get_current_tenant()
    return bool(tenant and getattr(request, "tenant_schema", None) == tenant.schema_name)


def tenant_schema_required(view_func):
    """Decorator ensuring the request matches the active tenant schema."""

    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not _is_valid_tenant_request(request):
            return HttpResponseForbidden("Invalid tenant schema")
        return view_func(request, *args, **kwargs)

    return _wrapped


class TenantSchemaRequiredMixin:
    """Mixin validating the active tenant schema for class-based views."""

    def dispatch(self, request, *args, **kwargs):
        if not _is_valid_tenant_request(request):
            return HttpResponseForbidden("Invalid tenant schema")
        return super().dispatch(request, *args, **kwargs)
