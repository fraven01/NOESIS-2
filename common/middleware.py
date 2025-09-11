from django.conf import settings
from django.db import connection


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
