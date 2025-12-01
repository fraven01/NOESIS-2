"""API Views for Case Management."""

from __future__ import annotations


from django.utils import timezone
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.request import Request
from rest_framework.response import Response

from cases.models import Case
from customers.tenant_context import TenantContext, TenantRequiredError
from noesis2.api import JSON_ERROR_STATUSES, default_extend_schema
from noesis2.api.serializers import IdempotentResponseSerializer


class CaseSerializer(IdempotentResponseSerializer, serializers.ModelSerializer):
    """Serializer for Case model."""

    class Meta:
        model = Case
        fields = [
            "idempotent",
            "id",
            "external_id",
            "title",
            "status",
            "phase",
            "metadata",
            "created_at",
            "updated_at",
            "closed_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "phase",
            "created_at",
            "updated_at",
            "closed_at",
            "metadata",
        ]

    def validate_external_id(self, value: str) -> str:
        """Ensure external_id is valid."""
        return value.strip()

    def to_representation(self, instance):
        """Add idempotent flag to serialized output."""
        data = super().to_representation(instance)
        # Cases don't support idempotent replay, always False
        data["idempotent"] = False
        return data

    def create(self, validated_data):
        """Create a new case instance."""
        # Remove non-model fields
        validated_data.pop("idempotent", None)
        return super().create(validated_data)


class CreateCaseSerializer(CaseSerializer):
    """Serializer for creating a new Case."""

    class Meta(CaseSerializer.Meta):
        fields = CaseSerializer.Meta.fields


class CaseViewSet(viewsets.ModelViewSet):
    """ViewSet for managing Cases."""

    serializer_class = CaseSerializer
    lookup_field = "external_id"
    lookup_value_regex = "[^/]+"  # Allow special chars in external_id if needed

    @default_extend_schema(
        summary="List Cases",
        description="List all cases for the current tenant.",
        parameters=[
            OpenApiParameter(
                name="status",
                description="Filter by case status (open/closed)",
                required=False,
                type=str,
            ),
        ],
    )
    def list(self, request, *args, **kwargs):
        """List cases for the current tenant."""
        return super().list(request, *args, **kwargs)

    @default_extend_schema(
        summary="Get Case",
        description="Retrieve details of a specific case.",
    )
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a specific case."""
        return super().retrieve(request, *args, **kwargs)

    @default_extend_schema(
        summary="Create Case",
        description="Create a new case for the current tenant.",
        error_statuses=JSON_ERROR_STATUSES,
    )
    def create(self, request, *args, **kwargs):
        """Create a new case."""
        return super().create(request, *args, **kwargs)

    def get_queryset(self):
        """Return cases for the current tenant only."""
        try:
            tenant = TenantContext.from_request(self.request, require=True)
        except TenantRequiredError:
            return Case.objects.none()

        queryset = Case.objects.filter(tenant=tenant)

        status_param = self.request.query_params.get("status")
        if status_param:
            queryset = queryset.filter(status=status_param)

        return queryset.order_by("-created_at")

    def perform_create(self, serializer):
        """Create case associated with current tenant."""
        try:
            tenant = TenantContext.from_request(self.request, require=True)
        except TenantRequiredError as e:
            raise PermissionDenied(str(e))

        # Check for existing case with same external_id for this tenant
        external_id = serializer.validated_data.get("external_id")
        if Case.objects.filter(tenant=tenant, external_id=external_id).exists():
            raise ValidationError(
                {"external_id": f"Case with ID '{external_id}' already exists."}
            )

        serializer.save(tenant=tenant)

    @extend_schema(summary="Close Case", request=None)
    @action(detail=True, methods=["post"])
    def close(self, request: Request, external_id: str | None = None) -> Response:
        """Close a case."""
        case = self.get_object()
        if case.status == Case.Status.CLOSED:
            return Response(
                {"detail": "Case is already closed."},
                status=status.HTTP_409_CONFLICT,
            )

        case.status = Case.Status.CLOSED
        case.closed_at = timezone.now()
        case.save(update_fields=["status", "closed_at", "updated_at"])

        serializer = self.get_serializer(case)
        return Response(serializer.data)

    @extend_schema(summary="Reopen Case", request=None)
    @action(detail=True, methods=["post"])
    def reopen(self, request: Request, external_id: str | None = None) -> Response:
        """Reopen a closed case."""
        case = self.get_object()
        if case.status == Case.Status.OPEN:
            return Response(
                {"detail": "Case is already open."},
                status=status.HTTP_409_CONFLICT,
            )

        case.status = Case.Status.OPEN
        case.closed_at = None
        case.save(update_fields=["status", "closed_at", "updated_at"])

        serializer = self.get_serializer(case)
        return Response(serializer.data)
