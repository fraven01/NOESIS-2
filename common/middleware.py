class TenantSchemaMiddleware:
    """Populate request.tenant_schema from the X-Tenant-Schema header."""

    header_name = "HTTP_X_TENANT_SCHEMA"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.tenant_schema = request.META.get(self.header_name)
        return self.get_response(request)
