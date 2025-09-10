from django.http import JsonResponse
from django.views import View

from .tenants import TenantSchemaRequiredMixin, tenant_schema_required


@tenant_schema_required
def demo_decorated_view(request):
    """Demo function-based view protected by tenant schema validation."""
    return JsonResponse({"status": "ok"})


class DemoView(TenantSchemaRequiredMixin, View):
    """Demo class-based view using the tenant schema validation mixin."""

    def get(self, request):
        return JsonResponse({"status": "ok"})
