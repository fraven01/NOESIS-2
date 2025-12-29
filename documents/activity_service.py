"""Document activity tracking service."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping
from uuid import UUID

from django.apps import apps
from django.http import HttpRequest
from django_tenants.utils import schema_context

from common.constants import META_TRACE_ID_KEY, X_TRACE_ID_HEADER


class ActivityTracker:
    """Centralized activity logging for documents."""

    @staticmethod
    def log(
        *,
        activity_type: str,
        document: Any | None = None,
        document_id: UUID | str | None = None,
        user: Any | None = None,
        request: HttpRequest | None = None,
        case_id: str | None = None,
        trace_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tenant_schema: str | None = None,
    ):
        """Log a document activity event."""
        if document is None and document_id is None:
            raise ValueError("document_or_document_id_required")

        ip_address = None
        user_agent = ""
        if request is not None:
            ip_address = request.META.get("REMOTE_ADDR")
            user_agent = request.META.get("HTTP_USER_AGENT", "")[:500]
            if trace_id is None:
                trace_id = request.headers.get(X_TRACE_ID_HEADER) or request.META.get(
                    META_TRACE_ID_KEY
                )

        context = schema_context(tenant_schema) if tenant_schema else nullcontext()
        DocumentActivity = apps.get_model("documents", "DocumentActivity")
        Document = apps.get_model("documents", "Document")

        with context:
            if document is None:
                document = (
                    Document.objects.filter(id=document_id)
                    .only("id", "case_id")
                    .first()
                )
                if document is None:
                    return None

            if case_id is None:
                case_id = getattr(document, "case_id", None)

            return DocumentActivity.objects.create(
                document=document,
                user=user,
                activity_type=activity_type,
                ip_address=ip_address,
                user_agent=user_agent,
                case_id=case_id,
                trace_id=trace_id,
                metadata=dict(metadata or {}),
            )
