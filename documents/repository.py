"""Repository abstractions for storing normalized documents and assets."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from uuid import UUID

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import transaction
from django.utils.module_loading import import_string

from common.logging import log_context

from .contracts import (
    Asset,
    AssetRef,
    BlobLocator,
    DocumentRef,
    NormalizedDocument,
    InlineBlob,
    FileBlob,
)
from .logging_utils import (
    asset_log_fields,
    document_log_fields,
    log_call,
    log_extra_entry,
    log_extra_exit,
)
from .storage import InMemoryStorage, Storage


logger = logging.getLogger(__name__)


ACTIVE_STATE = "active"
RETIRED_STATE = "retired"
DELETED_STATE = "deleted"


_DOCUMENT_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    ACTIVE_STATE: (ACTIVE_STATE, RETIRED_STATE, DELETED_STATE),
    RETIRED_STATE: (ACTIVE_STATE, RETIRED_STATE, DELETED_STATE),
    DELETED_STATE: (DELETED_STATE,),
}


def _normalize_lifecycle_state(value: Optional[str]) -> str:
    if value is None:
        return ACTIVE_STATE
    candidate = str(value).strip().lower()
    if candidate in {ACTIVE_STATE, RETIRED_STATE, DELETED_STATE}:
        return candidate
    return ACTIVE_STATE


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    candidate = str(reason).strip()
    return candidate or None


def _normalize_policy_events(events: Iterable[object]) -> Tuple[str, ...]:
    normalized: List[str] = []
    for event in events:
        if event is None:
            continue
        candidate = str(event).strip()
        if not candidate:
            continue
        normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


def _normalize_timestamp(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class DocumentLifecycleRecord:
    tenant_id: str
    document_id: UUID
    workflow_id: Optional[str]
    state: str = field(default=ACTIVE_STATE)
    changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: Optional[str] = None
    policy_events: Tuple[str, ...] = field(default_factory=tuple)

    def as_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "tenant_id": self.tenant_id,
            "document_id": str(self.document_id),
            "state": self.state,
            "changed_at": self.changed_at.isoformat(),
        }
        if self.workflow_id is not None:
            payload["workflow_id"] = self.workflow_id
        if self.reason:
            payload["reason"] = self.reason
        if self.policy_events:
            payload["policy_events"] = list(self.policy_events)
        return payload


@dataclass
class IngestionRunRecord:
    tenant: str
    case: str
    run_id: str
    status: str
    queued_at: str
    document_ids: Tuple[str, ...] = ()
    invalid_document_ids: Tuple[str, ...] = ()
    trace_id: Optional[str] = None
    embedding_profile: Optional[str] = None
    source: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    inserted_documents: Optional[int] = None
    replaced_documents: Optional[int] = None
    skipped_documents: Optional[int] = None
    inserted_chunks: Optional[int] = None
    error: Optional[str] = None

    def as_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "run_id": self.run_id,
            "status": self.status,
            "queued_at": self.queued_at,
            "document_ids": list(self.document_ids),
            "invalid_document_ids": list(self.invalid_document_ids),
        }
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        if self.embedding_profile:
            payload["embedding_profile"] = self.embedding_profile
        if self.source:
            payload["source"] = self.source
        if self.started_at:
            payload["started_at"] = self.started_at
        if self.finished_at:
            payload["finished_at"] = self.finished_at
        if self.duration_ms is not None:
            payload["duration_ms"] = float(self.duration_ms)
        if self.inserted_documents is not None:
            payload["inserted_documents"] = int(self.inserted_documents)
        if self.replaced_documents is not None:
            payload["replaced_documents"] = int(self.replaced_documents)
        if self.skipped_documents is not None:
            payload["skipped_documents"] = int(self.skipped_documents)
        if self.inserted_chunks is not None:
            payload["inserted_chunks"] = int(self.inserted_chunks)
        if self.error:
            payload["error"] = self.error
        return payload


class DocumentLifecycleStore:
    """In-memory lifecycle persistence for document states and ingestion runs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._documents: Dict[
            Tuple[str, Optional[str], UUID], DocumentLifecycleRecord
        ] = {}
        self._ingestion_runs: Dict[Tuple[str, str], IngestionRunRecord] = {}

    # Document lifecycle -------------------------------------------------

    def record_document_state(
        self,
        *,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
        state: str,
        reason: Optional[str] = None,
        policy_events: Iterable[object] = (),
        changed_at: Optional[datetime] = None,
    ) -> DocumentLifecycleRecord:
        normalized_state = _normalize_lifecycle_state(state)
        normalized_reason = _normalize_reason(reason)
        normalized_events = _normalize_policy_events(policy_events)
        normalized_workflow = (workflow_id or None) and str(workflow_id)
        timestamp = _normalize_timestamp(changed_at)

        key = (tenant_id, normalized_workflow, document_id)

        with self._lock:
            previous = self._documents.get(key)
            if previous is not None and previous.state != normalized_state:
                allowed = _DOCUMENT_TRANSITIONS.get(previous.state, ())
                if normalized_state not in allowed:
                    raise ValueError(
                        f"invalid_lifecycle_transition:{previous.state}->{normalized_state}"
                    )

            if normalized_reason is None:
                normalized_reason = previous.reason if previous else normalized_state
            if not normalized_events and previous is not None:
                normalized_events = previous.policy_events

            record = DocumentLifecycleRecord(
                tenant_id=tenant_id,
                document_id=document_id,
                workflow_id=normalized_workflow,
                state=normalized_state,
                changed_at=timestamp,
                reason=normalized_reason,
                policy_events=normalized_events,
            )
            self._documents[key] = record
            return record

    def get_document_state(
        self,
        *,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
    ) -> Optional[DocumentLifecycleRecord]:
        key = (tenant_id, (workflow_id or None) and str(workflow_id), document_id)
        with self._lock:
            return self._documents.get(key)

    # Ingestion run status -----------------------------------------------

    def record_ingestion_run_queued(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        document_ids: Iterable[str],
        invalid_document_ids: Iterable[str],
        queued_at: str,
        trace_id: Optional[str] = None,
        embedding_profile: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, object]:
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )
        normalized_invalid = tuple(
            value
            for value in (_normalize_identifier(v) for v in invalid_document_ids)
            if value is not None
        )
        record = IngestionRunRecord(
            tenant=tenant,
            case=case,
            run_id=str(run_id),
            status="queued",
            queued_at=str(queued_at),
            document_ids=normalized_ids,
            invalid_document_ids=normalized_invalid,
            trace_id=_normalize_optional_string(trace_id),
            embedding_profile=_normalize_optional_string(embedding_profile),
            source=_normalize_optional_string(source),
        )
        key = (tenant, case)
        with self._lock:
            self._ingestion_runs[key] = record
            return record.as_payload()

    def mark_ingestion_run_running(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        started_at: str,
        document_ids: Iterable[str],
    ) -> Optional[Dict[str, object]]:
        key = (tenant, case)
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )
        with self._lock:
            record = self._ingestion_runs.get(key)
            if record is None or record.run_id != run_id:
                return None
            record.status = "running"
            record.started_at = str(started_at)
            if normalized_ids:
                record.document_ids = normalized_ids
            return record.as_payload()

    def mark_ingestion_run_completed(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        finished_at: str,
        duration_ms: float,
        inserted_documents: int,
        replaced_documents: int,
        skipped_documents: int,
        inserted_chunks: int,
        invalid_document_ids: Iterable[str],
        document_ids: Iterable[str],
        error: Optional[str],
    ) -> Optional[Dict[str, object]]:
        key = (tenant, case)
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )
        normalized_invalid = tuple(
            value
            for value in (_normalize_identifier(v) for v in invalid_document_ids)
            if value is not None
        )
        with self._lock:
            record = self._ingestion_runs.get(key)
            if record is None or record.run_id != run_id:
                return None
            record.status = "failed" if error else "succeeded"
            record.finished_at = str(finished_at)
            record.duration_ms = float(duration_ms)
            record.inserted_documents = int(inserted_documents)
            record.replaced_documents = int(replaced_documents)
            record.skipped_documents = int(skipped_documents)
            record.inserted_chunks = int(inserted_chunks)
            record.invalid_document_ids = normalized_invalid
            if normalized_ids:
                record.document_ids = normalized_ids
            record.error = _normalize_optional_string(error)
            return record.as_payload()

    def get_ingestion_run(
        self, *, tenant: str, case: str
    ) -> Optional[Dict[str, object]]:
        key = (tenant, case)
        with self._lock:
            record = self._ingestion_runs.get(key)
            if record is None:
                return None
            return record.as_payload()

    def reset(self) -> None:
        with self._lock:
            self._documents.clear()
            self._ingestion_runs.clear()


def _workflow_storage_key(value: Optional[str]) -> str:
    return (str(value).strip() if value else "").strip()


def _workflow_from_storage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _stored_reason(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _stored_policy_events(value: Optional[Iterable[object]]) -> Tuple[str, ...]:
    if not value:
        return ()
    normalized: List[str] = []
    for event in value:
        candidate = str(event).strip()
        if candidate:
            normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


def _stored_identifier_sequence(value: Optional[Iterable[object]]) -> Tuple[str, ...]:
    if not value:
        return ()
    normalized: List[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate)
    return tuple(normalized)


def _lifecycle_model():
    return apps.get_model("documents", "DocumentLifecycleState")


def _ingestion_model():
    return apps.get_model("documents", "DocumentIngestionRun")


class PersistentDocumentLifecycleStore(DocumentLifecycleStore):
    """Database-backed lifecycle persistence using Django models."""

    def __init__(self) -> None:
        super().__init__()

    def record_document_state(
        self,
        *,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
        state: str,
        reason: Optional[str] = None,
        policy_events: Iterable[object] = (),
        changed_at: Optional[datetime] = None,
    ) -> DocumentLifecycleRecord:
        normalized_state = _normalize_lifecycle_state(state)
        normalized_reason = _normalize_reason(reason)
        normalized_events = _normalize_policy_events(policy_events)
        normalized_workflow = (workflow_id or None) and str(workflow_id)
        timestamp = _normalize_timestamp(changed_at)
        workflow_key = _workflow_storage_key(normalized_workflow)

        model = _lifecycle_model()

        with transaction.atomic():
            try:
                instance = model.objects.select_for_update().get(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    workflow_id=workflow_key,
                )
            except model.DoesNotExist:  # type: ignore[attr-defined]
                instance = None

            if instance is not None and instance.state != normalized_state:
                allowed = _DOCUMENT_TRANSITIONS.get(instance.state, ())
                if normalized_state not in allowed:
                    raise ValueError(
                        f"invalid_lifecycle_transition:{instance.state}->{normalized_state}"
                    )

            if normalized_reason is None:
                normalized_reason = (
                    _stored_reason(getattr(instance, "reason", None))
                    if instance is not None
                    else normalized_state
                )
            if not normalized_events and instance is not None:
                normalized_events = _stored_policy_events(
                    getattr(instance, "policy_events", ())
                )

            if instance is None:
                instance = model(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    workflow_id=workflow_key,
                )

            instance.state = normalized_state
            instance.changed_at = timestamp
            instance.reason = normalized_reason or ""
            instance.policy_events = list(normalized_events)
            instance.save()

        return DocumentLifecycleRecord(
            tenant_id=tenant_id,
            document_id=document_id,
            workflow_id=normalized_workflow,
            state=normalized_state,
            changed_at=timestamp,
            reason=normalized_reason,
            policy_events=normalized_events,
        )

    def get_document_state(
        self,
        *,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
    ) -> Optional[DocumentLifecycleRecord]:
        workflow_key = _workflow_storage_key((workflow_id or None) and str(workflow_id))
        model = _lifecycle_model()
        try:
            instance = model.objects.get(
                tenant_id=tenant_id,
                document_id=document_id,
                workflow_id=workflow_key,
            )
        except model.DoesNotExist:  # type: ignore[attr-defined]
            return None

        return DocumentLifecycleRecord(
            tenant_id=tenant_id,
            document_id=document_id,
            workflow_id=_workflow_from_storage(instance.workflow_id),
            state=instance.state,
            changed_at=_normalize_timestamp(instance.changed_at),
            reason=_stored_reason(instance.reason),
            policy_events=_stored_policy_events(instance.policy_events),
        )

    def record_ingestion_run_queued(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        document_ids: Iterable[str],
        invalid_document_ids: Iterable[str],
        queued_at: str,
        trace_id: Optional[str] = None,
        embedding_profile: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, object]:
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )
        normalized_invalid = tuple(
            value
            for value in (_normalize_identifier(v) for v in invalid_document_ids)
            if value is not None
        )

        model = _ingestion_model()

        with transaction.atomic():
            instance, _ = model.objects.select_for_update().get_or_create(
                tenant=tenant,
                case=case,
                defaults={},
            )
            instance.run_id = str(run_id)
            instance.status = "queued"
            instance.queued_at = str(queued_at)
            instance.document_ids = list(normalized_ids)
            instance.invalid_document_ids = list(normalized_invalid)
            instance.trace_id = _normalize_optional_string(trace_id) or ""
            instance.embedding_profile = (
                _normalize_optional_string(embedding_profile) or ""
            )
            instance.source = _normalize_optional_string(source) or ""
            instance.error = ""
            instance.started_at = ""
            instance.finished_at = ""
            instance.duration_ms = None
            instance.inserted_documents = None
            instance.replaced_documents = None
            instance.skipped_documents = None
            instance.inserted_chunks = None
            instance.save()

        return self.get_ingestion_run(tenant=tenant, case=case) or {}

    def mark_ingestion_run_running(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        started_at: str,
        document_ids: Iterable[str],
    ) -> Optional[Dict[str, object]]:
        model = _ingestion_model()
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )

        with transaction.atomic():
            try:
                instance = model.objects.select_for_update().get(
                    tenant=tenant, case=case
                )
            except model.DoesNotExist:  # type: ignore[attr-defined]
                return None

            if instance.run_id != run_id:
                return None

            instance.status = "running"
            instance.started_at = str(started_at)
            if normalized_ids:
                instance.document_ids = list(normalized_ids)
            instance.save()

        return self.get_ingestion_run(tenant=tenant, case=case)

    def mark_ingestion_run_completed(
        self,
        *,
        tenant: str,
        case: str,
        run_id: str,
        finished_at: str,
        duration_ms: float,
        inserted_documents: int,
        replaced_documents: int,
        skipped_documents: int,
        inserted_chunks: int,
        invalid_document_ids: Iterable[str],
        document_ids: Iterable[str],
        error: Optional[str],
    ) -> Optional[Dict[str, object]]:
        model = _ingestion_model()
        normalized_ids = tuple(
            value
            for value in (_normalize_identifier(v) for v in document_ids)
            if value is not None
        )
        normalized_invalid = tuple(
            value
            for value in (_normalize_identifier(v) for v in invalid_document_ids)
            if value is not None
        )

        with transaction.atomic():
            try:
                instance = model.objects.select_for_update().get(
                    tenant=tenant, case=case
                )
            except model.DoesNotExist:  # type: ignore[attr-defined]
                return None

            if instance.run_id != run_id:
                return None

            instance.status = "failed" if error else "succeeded"
            instance.finished_at = str(finished_at)
            instance.duration_ms = float(duration_ms)
            instance.inserted_documents = int(inserted_documents)
            instance.replaced_documents = int(replaced_documents)
            instance.skipped_documents = int(skipped_documents)
            instance.inserted_chunks = int(inserted_chunks)
            instance.invalid_document_ids = list(normalized_invalid)
            if normalized_ids:
                instance.document_ids = list(normalized_ids)
            normalized_error = _normalize_optional_string(error)
            instance.error = normalized_error or ""
            instance.save()

        return self.get_ingestion_run(tenant=tenant, case=case)

    def get_ingestion_run(
        self, *, tenant: str, case: str
    ) -> Optional[Dict[str, object]]:
        model = _ingestion_model()
        try:
            instance = model.objects.get(tenant=tenant, case=case)
        except model.DoesNotExist:  # type: ignore[attr-defined]
            return None

        payload = IngestionRunRecord(
            tenant=instance.tenant,
            case=instance.case,
            run_id=instance.run_id,
            status=instance.status,
            queued_at=instance.queued_at,
            document_ids=_stored_identifier_sequence(instance.document_ids),
            invalid_document_ids=_stored_identifier_sequence(
                instance.invalid_document_ids
            ),
            trace_id=_stored_reason(instance.trace_id),
            embedding_profile=_stored_reason(instance.embedding_profile),
            source=_stored_reason(instance.source),
            started_at=_stored_reason(instance.started_at),
            finished_at=_stored_reason(instance.finished_at),
            duration_ms=instance.duration_ms,
            inserted_documents=instance.inserted_documents,
            replaced_documents=instance.replaced_documents,
            skipped_documents=instance.skipped_documents,
            inserted_chunks=instance.inserted_chunks,
            error=_stored_reason(instance.error),
        )
        return payload.as_payload()

    def reset(self) -> None:
        model_lifecycle = _lifecycle_model()
        model_ingestion = _ingestion_model()
        with transaction.atomic():
            model_lifecycle.objects.all().delete()
            model_ingestion.objects.all().delete()


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _normalize_identifier(value: object) -> Optional[str]:
    candidate = str(value).strip()
    if not candidate:
        return None
    return candidate


def _load_default_lifecycle_store() -> DocumentLifecycleStore:
    class_path = "documents.repository.PersistentDocumentLifecycleStore"
    try:
        if not settings.configured:
            raise ImproperlyConfigured("settings_not_configured")
        class_path = getattr(settings, "DOCUMENT_LIFECYCLE_STORE_CLASS", class_path)
        store_class = import_string(class_path)
        store_instance = store_class()
        return store_instance
    except Exception:  # pragma: no cover - defensive fallback
        logger.warning(
            "lifecycle_store_init_failed",
            extra={"class_path": class_path},
            exc_info=True,
        )
        return DocumentLifecycleStore()


DEFAULT_LIFECYCLE_STORE = _load_default_lifecycle_store()


class DocumentsRepository:
    """Abstract persistence interface for normalized documents."""

    def upsert(
        self, doc: NormalizedDocument, workflow_id: Optional[str] = None
    ) -> NormalizedDocument:
        """Create or replace a document instance."""

        raise NotImplementedError

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
        workflow_id: Optional[str] = None,
    ) -> Optional[NormalizedDocument]:
        """Fetch a document by identifiers, returning ``None`` if missing."""

        raise NotImplementedError

    def list_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = False,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List document references for a collection ordered by recency.

        The returned cursor is a best-effort marker derived from ``created_at``
        and document identifiers and may change when records are reordered.
        When ``workflow_id`` is provided only entries produced by the workflow
        are returned.
        """

        raise NotImplementedError

    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List newest document versions per document identifier.

        The returned cursor is a best-effort marker derived from ``created_at``
        and document identifiers and may change when records are reordered. When
        ``workflow_id`` is provided the latest version within that workflow is
        returned.
        """

        raise NotImplementedError

    def delete(
        self,
        tenant_id: str,
        document_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        """Soft or hard delete a document across all versions."""

        raise NotImplementedError

    def add_asset(self, asset: Asset, workflow_id: Optional[str] = None) -> Asset:
        """Persist an asset for a previously stored document."""

        raise NotImplementedError

    def get_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
    ) -> Optional[Asset]:
        """Fetch an asset by its identifier."""

        raise NotImplementedError

    def list_assets_by_document(
        self,
        tenant_id: str,
        document_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[AssetRef], Optional[str]]:
        """List asset references for a document ordered by recency.

        The returned cursor is a best-effort marker derived from ``created_at``
        and asset identifiers and may change when records are reordered. When
        ``workflow_id`` is provided the listing is restricted to that workflow.
        """

        raise NotImplementedError

    def delete_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        """Soft or hard delete an asset."""

        raise NotImplementedError


@dataclass
class _StoredDocument:
    value: NormalizedDocument
    lifecycle_state: str = ACTIVE_STATE

    def __post_init__(self) -> None:
        initial = getattr(self.value, "lifecycle_state", None) or self.lifecycle_state
        self.set_state(initial)

    def set_state(self, state: str) -> None:
        normalized = _normalize_lifecycle_state(state)
        self.lifecycle_state = normalized
        try:
            self.value.lifecycle_state = normalized
        except Exception:
            object.__setattr__(self.value, "lifecycle_state", normalized)


@dataclass
class _StoredAsset:
    value: Asset
    deleted: bool = False


class InMemoryDocumentsRepository(DocumentsRepository):
    """Thread-safe in-memory repository implementation."""

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self._lock = RLock()
        self._storage = storage or InMemoryStorage()
        self._documents: Dict[Tuple[str, str, UUID, Optional[str]], _StoredDocument] = (
            {}
        )
        self._assets: Dict[Tuple[str, str, UUID], _StoredAsset] = {}
        self._asset_index: Dict[Tuple[str, str, UUID], set[UUID]] = {}

    @log_call("docs.upsert")
    def upsert(
        self, doc: NormalizedDocument, workflow_id: Optional[str] = None
    ) -> NormalizedDocument:
        doc_copy = doc.model_copy(deep=True)
        doc_copy = self._materialize_document(doc_copy)
        ref = doc_copy.ref
        workflow = workflow_id or ref.workflow_id
        if workflow != ref.workflow_id:
            raise ValueError("workflow_mismatch")

        with log_context(
            tenant=ref.tenant_id,
            collection_id=str(ref.collection_id) if ref.collection_id else None,
            workflow_id=workflow,
        ):
            log_extra_entry(**document_log_fields(doc_copy))
            key = (ref.tenant_id, workflow, ref.document_id, ref.version)

            with self._lock:
                self._documents[key] = _StoredDocument(
                    value=doc_copy, lifecycle_state=doc_copy.lifecycle_state
                )

                if doc_copy.assets:
                    for asset in doc_copy.assets:
                        if asset.ref.workflow_id != workflow:
                            raise ValueError("asset_workflow_mismatch")
                        self._store_asset_locked(asset)

                self._refresh_document_assets_locked(
                    ref.tenant_id, ref.document_id, workflow
                )

                stored = self.get(
                    ref.tenant_id,
                    ref.document_id,
                    ref.version,
                    workflow_id=workflow,
                )
                log_extra_exit(asset_count=len(doc_copy.assets))
                return stored

    @log_call("docs.get")
    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
        workflow_id: Optional[str] = None,
    ) -> Optional[NormalizedDocument]:
        with log_context(tenant=tenant_id, workflow_id=workflow_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                version=version,
                prefer_latest=prefer_latest,
                workflow_id=workflow_id,
            )
            with self._lock:
                document: Optional[NormalizedDocument]
                if prefer_latest and version is None:
                    document = self._latest_document_locked(
                        tenant_id, document_id, workflow_id=workflow_id
                    )
                else:
                    document = self._find_document_by_version_locked(
                        tenant_id,
                        document_id,
                        version,
                        workflow_id=workflow_id,
                    )

                if document is None:
                    log_extra_exit(found=False)
                    return None

                doc_copy = document.model_copy(deep=True)
                doc_copy.assets = self._collect_assets_for_document_locked(
                    tenant_id,
                    document_id,
                    doc_copy.ref.workflow_id,
                )
                log_extra_exit(
                    found=True,
                    asset_count=len(doc_copy.assets),
                    workflow_id=doc_copy.ref.workflow_id,
                )
                return doc_copy

    @log_call("docs.list")
    def list_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = False,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List document references for a collection ordered by recency.

        Pagination cursors are best-effort markers derived from ``created_at``
        and document IDs. When ``latest_only`` is ``True`` the method delegates
        to :meth:`list_latest_by_collection` to surface only the most recent
        version per ``document_id``.
        """
        with log_context(
            tenant=tenant_id,
            collection_id=str(collection_id),
            workflow_id=workflow_id,
        ):
            log_extra_entry(
                tenant_id=tenant_id,
                collection_id=collection_id,
                limit=limit,
                cursor_present=bool(cursor),
                latest_only=latest_only,
                workflow_id=workflow_id,
            )
            if latest_only:
                refs, next_cursor = self.list_latest_by_collection(
                    tenant_id,
                    collection_id,
                    limit=limit,
                    cursor=cursor,
                    workflow_id=workflow_id,
                )
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

            with self._lock:
                entries = [
                    self._document_entry(doc)
                    for doc in self._iter_documents_locked(
                        tenant_id, collection_id, workflow_id
                    )
                ]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

    @log_call("docs.list_latest")
    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List the newest document reference per ``document_id``.

        Pagination cursors remain best-effort markers derived from ``created_at``
        and document IDs. Ties on ``created_at`` timestamps are resolved via
        lexicographic version comparison to ensure deterministic ordering.
        """
        with log_context(
            tenant=tenant_id,
            collection_id=str(collection_id),
            workflow_id=workflow_id,
        ):
            log_extra_entry(
                tenant_id=tenant_id,
                collection_id=collection_id,
                limit=limit,
                cursor_present=bool(cursor),
                workflow_id=workflow_id,
            )
            with self._lock:
                latest: Dict[UUID, NormalizedDocument] = {}
                for doc in self._iter_documents_locked(
                    tenant_id, collection_id, workflow_id
                ):
                    current = latest.get(doc.ref.document_id)
                    if current is None or self._newer(doc, current):
                        latest[doc.ref.document_id] = doc

                entries = [self._document_entry(doc) for doc in latest.values()]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

    @log_call("docs.delete")
    def delete(
        self,
        tenant_id: str,
        document_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        with log_context(tenant=tenant_id, workflow_id=workflow_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                hard=hard,
                workflow_id=workflow_id,
            )
            doc_keys = []
            with self._lock:
                for key, stored in self._documents.items():
                    key_tenant, key_workflow, key_doc, _ = key
                    if key_tenant != tenant_id or key_doc != document_id:
                        continue
                    if workflow_id is not None and key_workflow != workflow_id:
                        continue
                    doc_keys.append(key)

                if not doc_keys:
                    log_extra_exit(found=False)
                    return False

                changed = False
                for key in doc_keys:
                    stored = self._documents.get(key)
                    if stored is None:
                        continue
                    if hard:
                        del self._documents[key]
                        changed = True
                        continue
                    if stored.lifecycle_state == RETIRED_STATE:
                        continue
                    stored.set_state(RETIRED_STATE)
                    changed = True

                self._mark_assets_for_document_locked(
                    tenant_id, document_id, workflow_id=workflow_id, hard=hard
                )
                log_extra_exit(found=True, deleted=changed, workflow_id=workflow_id)
                return changed

    @log_call("assets.add")
    def add_asset(self, asset: Asset, workflow_id: Optional[str] = None) -> Asset:
        asset_copy = self._materialize_asset(asset.model_copy(deep=True))
        tenant_id = asset_copy.ref.tenant_id
        document_id = asset_copy.ref.document_id
        workflow = workflow_id or asset_copy.ref.workflow_id
        if workflow != asset_copy.ref.workflow_id:
            raise ValueError("workflow_mismatch")

        with log_context(tenant=tenant_id, workflow_id=workflow):
            log_extra_entry(**asset_log_fields(asset_copy))
            with self._lock:
                document = self._latest_document_locked(
                    tenant_id, document_id, workflow_id=workflow
                )
                if document is None:
                    any_document = self._latest_document_locked(
                        tenant_id, document_id, workflow_id=None
                    )
                    if any_document is not None:
                        raise ValueError("asset_workflow_mismatch")
                    raise ValueError("document_missing")
                if asset_copy.ref.workflow_id != document.ref.workflow_id:
                    raise ValueError("asset_workflow_mismatch")

                self._store_asset_locked(asset_copy)
                self._refresh_document_assets_locked(tenant_id, document_id, workflow)
                stored = self._snapshot_asset_locked(
                    tenant_id, workflow, asset_copy.ref.asset_id
                )
                log_extra_exit(**asset_log_fields(stored))
                return stored

    @log_call("assets.get")
    def get_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
    ) -> Optional[Asset]:
        with log_context(tenant=tenant_id, workflow_id=workflow_id):
            log_extra_entry(
                tenant_id=tenant_id,
                asset_id=asset_id,
                workflow_id=workflow_id,
            )
            with self._lock:
                stored = self._get_asset_record_locked(tenant_id, asset_id, workflow_id)
                if stored is None:
                    log_extra_exit(found=False)
                    return None
                asset_copy = stored.value.model_copy(deep=True)
                log_extra_exit(found=True, **asset_log_fields(asset_copy))
                return asset_copy

    @log_call("assets.list")
    def list_assets_by_document(
        self,
        tenant_id: str,
        document_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[AssetRef], Optional[str]]:
        with log_context(tenant=tenant_id, workflow_id=workflow_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                limit=limit,
                cursor_present=bool(cursor),
                workflow_id=workflow_id,
            )
            with self._lock:
                assets = self._collect_assets_for_document_locked(
                    tenant_id, document_id, workflow_id
                )
                entries = [self._asset_entry(asset) for asset in assets]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

    @log_call("assets.delete")
    def delete_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        with log_context(tenant=tenant_id, workflow_id=workflow_id):
            log_extra_entry(
                tenant_id=tenant_id,
                asset_id=asset_id,
                hard=hard,
                workflow_id=workflow_id,
            )
            with self._lock:
                targets: List[Tuple[str, _StoredAsset]] = []
                if workflow_id is not None:
                    stored = self._assets.get((tenant_id, workflow_id, asset_id))
                    if stored is not None:
                        targets.append((workflow_id, stored))
                else:
                    for (tenant, wf, aid), stored in self._assets.items():
                        if tenant != tenant_id or aid != asset_id:
                            continue
                        targets.append((wf, stored))

                if not targets:
                    log_extra_exit(found=False)
                    return False

                changed = False
                for wf, stored in targets:
                    key = (tenant_id, wf, asset_id)
                    doc_id = stored.value.ref.document_id
                    if hard:
                        if key in self._assets:
                            del self._assets[key]
                        index = self._asset_index.get((tenant_id, wf, doc_id))
                        if index and asset_id in index:
                            index.remove(asset_id)
                            if not index:
                                del self._asset_index[(tenant_id, wf, doc_id)]
                        changed = True
                    else:
                        if stored.deleted:
                            continue
                        stored.deleted = True
                        changed = True

                    self._refresh_document_assets_locked(tenant_id, doc_id, wf)

                log_extra_exit(found=True, deleted=changed, workflow_id=workflow_id)
                return changed

    # Internal helpers -------------------------------------------------

    def _store_asset_locked(self, asset: Asset) -> None:
        key = (asset.ref.tenant_id, asset.ref.workflow_id, asset.ref.asset_id)
        existing = self._assets.get(key)
        if existing and existing.value.ref.document_id != asset.ref.document_id:
            raise ValueError("asset_conflict")

        asset_copy = asset.model_copy(deep=True)
        self._assets[key] = _StoredAsset(value=asset_copy, deleted=False)

        index_key = (asset.ref.tenant_id, asset.ref.workflow_id, asset.ref.document_id)
        bucket = self._asset_index.setdefault(index_key, set())
        bucket.add(asset.ref.asset_id)

    def _collect_assets_for_document_locked(
        self,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
    ) -> List[Asset]:
        assets: List[Asset] = []
        if workflow_id is not None:
            asset_ids = self._asset_index.get(
                (tenant_id, workflow_id, document_id), set()
            )
            keys = [(tenant_id, workflow_id, asset_id) for asset_id in asset_ids]
        else:
            keys = [
                (tenant, wf, asset_id)
                for (tenant, wf, doc_id), asset_ids in self._asset_index.items()
                if tenant == tenant_id and doc_id == document_id
                for asset_id in asset_ids
            ]

        for key in keys:
            stored = self._assets.get(key)
            if not stored or stored.deleted:
                continue
            assets.append(stored.value.model_copy(deep=True))

        assets.sort(key=lambda asset: self._asset_entry(asset)[0])
        return assets

    def _mark_assets_for_document_locked(
        self,
        tenant_id: str,
        document_id: UUID,
        *,
        workflow_id: Optional[str],
        hard: bool,
    ) -> None:
        if workflow_id is not None:
            index_keys = [(tenant_id, workflow_id, document_id)]
        else:
            index_keys = [
                key
                for key in self._asset_index.keys()
                if key[0] == tenant_id and key[2] == document_id
            ]

        for index_key in index_keys:
            asset_ids = list(self._asset_index.get(index_key, set()))
            tenant, wf, _ = index_key
            for asset_id in asset_ids:
                stored = self._assets.get((tenant, wf, asset_id))
                if not stored:
                    continue
                if hard:
                    del self._assets[(tenant, wf, asset_id)]
                else:
                    stored.deleted = True

            if hard and index_key in self._asset_index:
                del self._asset_index[index_key]

            self._refresh_document_assets_locked(tenant, document_id, wf)

    def _has_active_document_locked(
        self, tenant_id: str, document_id: UUID, workflow_id: Optional[str]
    ) -> bool:
        for key, stored in self._documents.items():
            key_tenant, key_workflow, key_doc, _ = key
            if key_tenant != tenant_id or key_doc != document_id:
                continue
            if workflow_id is not None and key_workflow != workflow_id:
                continue
            if stored.lifecycle_state == ACTIVE_STATE:
                return True
        return False

    def _refresh_document_assets_locked(
        self,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
    ) -> None:
        for key, stored in self._documents.items():
            key_tenant, key_workflow, key_doc, _ = key
            if key_tenant != tenant_id or key_doc != document_id:
                continue
            if workflow_id is not None and key_workflow != workflow_id:
                continue
            assets = self._collect_assets_for_document_locked(
                tenant_id, document_id, key_workflow
            )
            stored.value.assets = [asset.model_copy(deep=True) for asset in assets]

    def _snapshot_asset_locked(
        self, tenant_id: str, workflow_id: str, asset_id: UUID
    ) -> Asset:
        stored = self._assets[(tenant_id, workflow_id, asset_id)]
        return stored.value.model_copy(deep=True)

    def _get_asset_record_locked(
        self,
        tenant_id: str,
        asset_id: UUID,
        workflow_id: Optional[str],
    ) -> Optional[_StoredAsset]:
        if workflow_id is not None:
            stored = self._assets.get((tenant_id, workflow_id, asset_id))
            if not stored or stored.deleted:
                return None
            return stored

        selected: Optional[_StoredAsset] = None
        for (tenant, wf, aid), stored in self._assets.items():
            if tenant != tenant_id or aid != asset_id:
                continue
            if stored.deleted:
                continue
            if selected is None:
                selected = stored
                continue
            current = stored.value
            previous = selected.value
            if current.created_at > previous.created_at:
                selected = stored
            elif (
                current.created_at == previous.created_at
                and current.ref.workflow_id > previous.ref.workflow_id
            ):
                selected = stored
        return selected

    def _materialize_document(self, doc: NormalizedDocument) -> NormalizedDocument:
        doc.blob = self._materialize_blob(
            doc.blob,
            owner_checksum=doc.checksum,
            checksum_error="document_checksum_mismatch",
        )
        doc.assets = [self._materialize_asset(asset) for asset in doc.assets]
        return doc

    def _materialize_asset(self, asset: Asset) -> Asset:
        asset.blob = self._materialize_blob(
            asset.blob,
            owner_checksum=asset.checksum,
            checksum_error="asset_checksum_mismatch",
        )
        return asset

    def _materialize_blob(
        self,
        blob: BlobLocator,
        *,
        owner_checksum: Optional[str],
        checksum_error: str,
    ) -> BlobLocator:
        if isinstance(blob, InlineBlob):
            payload = blob.decoded_payload()
            uri, sha256, size = self._storage.put(payload)
            if blob.sha256 != sha256:
                raise ValueError("inline_checksum_mismatch")
            if owner_checksum is not None and owner_checksum != sha256:
                raise ValueError(checksum_error)
            return FileBlob(type="file", uri=uri, sha256=sha256, size=size)

        blob_sha = getattr(blob, "sha256", None)
        if (
            owner_checksum is not None
            and blob_sha is not None
            and blob_sha != owner_checksum
        ):
            raise ValueError(checksum_error)
        return blob

    # Ordering helpers -------------------------------------------------

    @staticmethod
    def _encode_cursor(parts: Iterable[str]) -> str:
        """Encode a best-effort cursor based on ``created_at`` and object IDs."""
        payload = "|".join(parts)
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
        return encoded.decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str) -> List[str]:
        """Decode cursors produced by :meth:`_encode_cursor`."""
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
            text = decoded.decode("utf-8")
            return text.split("|")
        except Exception as exc:  # pragma: no cover
            raise ValueError("cursor_invalid") from exc

    def _document_entry(
        self, doc: NormalizedDocument
    ) -> Tuple[Tuple[float, str, str, str], NormalizedDocument]:
        version_key = doc.ref.version or ""
        key = (
            -doc.created_at.timestamp(),
            str(doc.ref.document_id),
            doc.ref.workflow_id,
            version_key,
        )
        return key, doc

    def _asset_entry(self, asset: Asset) -> Tuple[Tuple[float, str, str], Asset]:
        key = (
            -asset.created_at.timestamp(),
            str(asset.ref.asset_id),
            asset.ref.workflow_id,
        )
        return key, asset

    def _cursor_start(
        self,
        entries: List[Tuple[Tuple, object]],
        cursor: Optional[str],
    ) -> int:
        if not cursor:
            return 0
        parts = self._decode_cursor(cursor)
        if not parts:
            raise ValueError("cursor_invalid")
        expected_length = len(entries[0][0]) if entries else len(parts)
        if len(parts) != expected_length:
            raise ValueError("cursor_invalid")

        if len(parts) == 4:
            try:
                timestamp = datetime.fromisoformat(parts[0])
                UUID(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("cursor_invalid") from exc
            cursor_key = (-timestamp.timestamp(), parts[1], parts[2], parts[3])
        elif len(parts) == 3:
            try:
                timestamp = datetime.fromisoformat(parts[0])
                UUID(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("cursor_invalid") from exc
            cursor_key = (-timestamp.timestamp(), parts[1], parts[2])
        elif len(parts) == 2:
            try:
                timestamp = datetime.fromisoformat(parts[0])
                UUID(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("cursor_invalid") from exc
            cursor_key = (-timestamp.timestamp(), parts[1])
        else:  # pragma: no cover - defensive branch
            raise ValueError("cursor_invalid")

        index = 0
        for idx, (key, _) in enumerate(entries):
            if key <= cursor_key:
                index = idx + 1
        return index

    def _next_cursor(
        self,
        entries: List[Tuple[Tuple, object]],
        start: int,
        limit: int,
    ) -> Optional[str]:
        """Return a best-effort pagination cursor for the last returned entry."""
        end = start + limit
        if end >= len(entries):
            return None
        key, obj = entries[end - 1]
        if len(key) == 4:
            doc: NormalizedDocument = obj  # type: ignore[assignment]
            parts = [
                doc.created_at.isoformat(),
                str(doc.ref.document_id),
                doc.ref.workflow_id,
                doc.ref.version or "",
            ]
        elif len(key) == 3:
            asset: Asset = obj  # type: ignore[assignment]
            parts = [
                asset.created_at.isoformat(),
                str(asset.ref.asset_id),
                asset.ref.workflow_id,
            ]
        else:  # pragma: no cover - defensive branch
            raise ValueError("cursor_invalid")
        return self._encode_cursor(parts)

    def _find_document_by_version_locked(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str],
        *,
        workflow_id: Optional[str],
    ) -> Optional[NormalizedDocument]:
        selected: Optional[NormalizedDocument] = None
        for stored in self._documents.values():
            if stored.lifecycle_state != ACTIVE_STATE:
                continue
            doc = stored.value
            if doc.ref.tenant_id != tenant_id or doc.ref.document_id != document_id:
                continue
            if workflow_id is not None and doc.ref.workflow_id != workflow_id:
                continue
            if doc.ref.version != version:
                continue
            if selected is None or self._newer(doc, selected):
                selected = doc
        return selected

    def _iter_documents_locked(
        self,
        tenant_id: str,
        collection_id: Optional[UUID],
        workflow_id: Optional[str],
    ) -> Iterable[NormalizedDocument]:
        for stored in self._documents.values():
            if stored.lifecycle_state != ACTIVE_STATE:
                continue
            doc = stored.value
            if doc.ref.tenant_id != tenant_id:
                continue
            if workflow_id is not None and doc.ref.workflow_id != workflow_id:
                continue
            if collection_id is not None and doc.ref.collection_id != collection_id:
                continue
            yield doc

    @staticmethod
    def _newer(left: NormalizedDocument, right: NormalizedDocument) -> bool:
        """Return ``True`` when ``left`` is considered more recent than ``right``.

        ``created_at`` takes precedence; equal timestamps fall back to comparing
        the version strings lexicographically (empty string for ``None``) so tie
        breaks stay deterministic without assuming semantic versioning.
        """
        if left.created_at > right.created_at:
            return True
        if left.created_at < right.created_at:
            return False
        left_version = left.ref.version or ""
        right_version = right.ref.version or ""
        return left_version > right_version

    def _latest_document_locked(
        self,
        tenant_id: str,
        document_id: UUID,
        workflow_id: Optional[str],
    ) -> Optional[NormalizedDocument]:
        latest: Optional[NormalizedDocument] = None
        for stored in self._documents.values():
            if stored.lifecycle_state != ACTIVE_STATE:
                continue
            doc = stored.value
            if doc.ref.tenant_id != tenant_id or doc.ref.document_id != document_id:
                continue
            if workflow_id is not None and doc.ref.workflow_id != workflow_id:
                continue
            if latest is None or self._newer(doc, latest):
                latest = doc
        return latest


__all__ = [
    "DocumentsRepository",
    "InMemoryDocumentsRepository",
]
