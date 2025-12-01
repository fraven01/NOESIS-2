from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import OpenApiExample

from .tenants import TenantSchemaRequiredMixin, tenant_schema_required
from noesis2.api import curl_code_sample
from noesis2.api.schema import default_extend_schema
from noesis2.api.serializers import TenantDemoResponseSerializer


@tenant_schema_required
def demo_decorated_view(request):
    """Demo function-based view protected by tenant schema validation."""
    return JsonResponse({"status": "ok"})


class DemoView(TenantSchemaRequiredMixin, APIView):
    """Demo class-based view using the tenant schema validation mixin."""

    authentication_classes: list = []
    permission_classes: list = []

    @default_extend_schema(
        responses={200: TenantDemoResponseSerializer},
        include_trace_header=False,
        description="Tenant-aware readiness probe exposing minimal status metadata.",
        examples=[
            OpenApiExample(
                name="TenantDemoResponse",
                summary="Tenant demo response",
                description="Minimal JSON payload returned when the tenant probe succeeds.",
                value={"status": "ok"},
                response_only=True,
            )
        ],
        extensions=curl_code_sample(
            'curl -H "Host: demo.localhost" -H "X-Tenant-Schema: demo" '
            "https://api.noesis.example/tenant-demo/"
        ),
    )
    def get(self, request):
        return Response({"status": "ok"})
