"""Document lifecycle service abstractions for graph orchestration."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence
from uuid import UUID

from documents import api as documents_api
from documents.api import LifecycleStatusUpdate, NormalizedDocumentPayload


class DocumentLifecycleService(Protocol):
    """Protocol describing lifecycle operations used by ingestion graphs."""

    def normalize_from_raw(
        self,
        *,
        raw_reference: Mapping[str, Any],
        tenant_id: str,
        case_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> NormalizedDocumentPayload:
        """Normalize crawler payloads into canonical :class:`NormalizedDocument` objects."""

    def update_lifecycle_status(
        self,
        *,
        tenant_id: str,
        document_id: str | UUID,
        status: str,
        previous_status: Optional[str] = None,
        workflow_id: Optional[str] = None,
        reason: Optional[str] = None,
        policy_events: Optional[Mapping[str, Any] | Sequence[str]] = None,
    ) -> LifecycleStatusUpdate:
        """Persist lifecycle transitions for crawler documents."""


class DocumentsApiLifecycleService:
    """Default implementation delegating to :mod:`documents.api`."""

    def normalize_from_raw(
        self,
        *,
        raw_reference: Mapping[str, Any],
        tenant_id: str,
        case_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> NormalizedDocumentPayload:
        return documents_api.normalize_from_raw(
            raw_reference=raw_reference,
            tenant_id=tenant_id,
            case_id=case_id,
            request_id=request_id,
        )

    def update_lifecycle_status(
        self,
        *,
        tenant_id: str,
        document_id: str | UUID,
        status: str,
        previous_status: Optional[str] = None,
        workflow_id: Optional[str] = None,
        reason: Optional[str] = None,
        policy_events: Optional[Mapping[str, Any] | Sequence[str]] = None,
    ) -> LifecycleStatusUpdate:
        return documents_api.update_lifecycle_status(
            tenant_id=tenant_id,
            document_id=document_id,
            status=status,
            previous_status=previous_status,
            workflow_id=workflow_id,
            reason=reason,
            policy_events=policy_events,
        )


__all__ = ["DocumentLifecycleService", "DocumentsApiLifecycleService"]
