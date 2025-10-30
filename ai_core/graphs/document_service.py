"""Document lifecycle service abstractions for graph orchestration."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence
from uuid import UUID

from documents import api as documents_api
from documents.api import LifecycleStatusUpdate, NormalizedDocumentPayload
from documents.contracts import NormalizedDocument
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository
from common.object_store import ObjectStore, get_default_object_store


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


class DocumentPersistenceService(Protocol):
    """Protocol describing persistence operations for normalized documents."""

    def upsert_normalized(
        self,
        *,
        normalized: NormalizedDocumentPayload,
    ) -> NormalizedDocument:
        """Persist a normalized document and return the stored representation."""


class DocumentsApiLifecycleService:
    """Default implementation delegating to :mod:`documents.api`."""

    def __init__(
        self,
        *,
        repository: Optional[DocumentsRepository] = None,
        object_store: ObjectStore | None = None,
    ) -> None:
        self._repository = repository
        self._object_store = object_store

    def _resolve_object_store(self) -> ObjectStore:
        if self._object_store is not None:
            return self._object_store
        return get_default_object_store()

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
            object_store=self._resolve_object_store(),
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

    @property
    def repository(self) -> Optional[DocumentsRepository]:
        return self._repository

    @property
    def object_store(self) -> ObjectStore:
        return self._resolve_object_store()

    def upsert_normalized(
        self,
        *,
        normalized: NormalizedDocumentPayload,
    ) -> NormalizedDocument:
        if self._repository is None:
            raise RuntimeError("documents_repository_not_configured")
        workflow_id = normalized.document.ref.workflow_id
        return self._repository.upsert(
            normalized.document, workflow_id=workflow_id
        )


class DocumentsRepositoryAdapter(DocumentPersistenceService):
    """Adapter wiring the repository contract into graph orchestration."""

    def __init__(
        self,
        *,
        repository: Optional[DocumentsRepository] = None,
    ) -> None:
        if repository is None:
            repository = InMemoryDocumentsRepository()
        self._repository = repository

    @property
    def repository(self) -> DocumentsRepository:
        return self._repository

    def upsert_normalized(
        self,
        *,
        normalized: NormalizedDocumentPayload,
    ) -> NormalizedDocument:
        workflow_id = normalized.document.ref.workflow_id
        return self._repository.upsert(
            normalized.document, workflow_id=workflow_id
        )


__all__ = [
    "DocumentLifecycleService",
    "DocumentsApiLifecycleService",
    "DocumentPersistenceService",
    "DocumentsRepositoryAdapter",
]
