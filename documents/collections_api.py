"""REST API endpoints for document collections."""

from __future__ import annotations

from drf_spectacular.utils import OpenApiParameter
from rest_framework import mixins, permissions, viewsets

from cases.authz import get_accessible_cases_queryset
from customers.tenant_context import TenantContext, TenantRequiredError
from documents.authz import user_has_tenant_wide_access
from documents.models import DocumentCollection
from documents.serializers import DocumentCollectionSerializer
from noesis2.api import default_extend_schema


class DocumentCollectionViewSet(
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = DocumentCollectionSerializer
    permission_classes = [permissions.IsAuthenticated]

    @default_extend_schema(
        summary="List collections",
        description="List document collections accessible to the current user.",
        parameters=[
            OpenApiParameter(
                name="case_id",
                description="Filter collections by case external ID.",
                required=False,
                type=str,
            ),
        ],
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    def get_queryset(self):
        try:
            tenant = TenantContext.from_request(
                self.request,
                allow_headers=True,
                require=True,
            )
        except TenantRequiredError:
            return DocumentCollection.objects.none()

        user = getattr(self.request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return DocumentCollection.objects.none()

        collections = DocumentCollection.objects.select_related("case").filter(
            tenant=tenant
        )

        if not user_has_tenant_wide_access(user=user, tenant=tenant):
            accessible_cases = get_accessible_cases_queryset(user, tenant)
            collections = collections.filter(case__in=accessible_cases)

        case_filter = self.request.query_params.get("case_id")
        if case_filter:
            collections = collections.filter(case__external_id=case_filter)

        return collections.order_by("name", "created_at")
