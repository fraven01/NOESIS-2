"""Health endpoint exposing document lifecycle diagnostics."""

from __future__ import annotations

from django.db import connection
from django.db.models import Count
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import Document


@api_view(["GET"])
@permission_classes([AllowAny])
def document_lifecycle_health(_request):
    """Return basic lifecycle and database health information."""

    checks: dict[str, dict[str, object]] = {}

    # Database connectivity
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["database"] = {"status": "healthy"}
    except Exception as exc:  # pragma: no cover - defensive
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    # Document model availability
    try:
        count = Document.objects.count()
        checks["document_model"] = {
            "status": "healthy",
            "total_documents": count,
        }
    except Exception as exc:  # pragma: no cover - defensive
        checks["document_model"] = {"status": "unhealthy", "error": str(exc)}

    # Lifecycle distribution
    try:
        distribution = dict(
            Document.objects.values("lifecycle_state")
            .annotate(count=Count("id"))
            .values_list("lifecycle_state", "count")
        )
        checks["lifecycle_distribution"] = {
            "status": "healthy",
            "states": distribution,
        }
    except Exception as exc:  # pragma: no cover - defensive
        checks["lifecycle_distribution"] = {"status": "unhealthy", "error": str(exc)}

    all_healthy = all(entry.get("status") == "healthy" for entry in checks.values())

    return Response(
        {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "timestamp": timezone.now().isoformat(),
        }
    )
