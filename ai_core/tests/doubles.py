"""Test doubles and mocks for AI Core tests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from threading import RLock
from typing import Any, Optional
from uuid import UUID
from dataclasses import dataclass

from documents.repository import (
    DocumentLifecycleRecord,
    DocumentLifecycleStore,
    IngestionRunRecord,
    _normalize_identifier,
    _normalize_lifecycle_state,
    _normalize_optional_string,
    _normalize_policy_events,
    _normalize_reason,
    _normalize_timestamp,
    _DOCUMENT_TRANSITIONS,
)


@dataclass
class MockTenant:
    schema_name: str
    domain_url: str = "example.com"


class MockTenantContext:
    @classmethod
    def from_request(cls, request: Any, **kwargs: Any) -> Optional[MockTenant]:
        # Extract tenant from header or meta, similar to real implementation but no DB
        if hasattr(request, "headers"):
            tenant_id = request.headers.get("X-Tenant-Id")
            if tenant_id:
                return MockTenant(schema_name=tenant_id)

        meta = getattr(request, "META", {})
        tenant_id = meta.get("HTTP_X_TENANT_ID") or meta.get("tenant_id")
        if tenant_id:
            return MockTenant(schema_name=tenant_id)

        return None

    @classmethod
    def resolve_identifier(cls, identifier: str, **kwargs: Any) -> Optional[MockTenant]:
        return MockTenant(schema_name=identifier)


class MemoryDocumentLifecycleStore(DocumentLifecycleStore):
    """In-memory lifecycle persistence for document states and ingestion runs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._documents: dict[
            tuple[str, Optional[str], UUID], DocumentLifecycleRecord
        ] = {}
        self._ingestion_runs: dict[tuple[str, str], IngestionRunRecord] = {}

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
        tenant_id: str,
        case: Optional[str],
        run_id: str,
        document_ids: Iterable[str],
        invalid_document_ids: Iterable[str],
        queued_at: str,
        trace_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        embedding_profile: Optional[str] = None,
        source: Optional[str] = None,
    ) -> dict[str, object]:
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
            tenant_id=tenant_id,
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
        key = (tenant_id, case or "")
        with self._lock:
            self._ingestion_runs[key] = record
            return record.as_payload()

    def mark_ingestion_run_running(
        self,
        *,
        tenant_id: str,
        case: Optional[str],
        run_id: str,
        started_at: str,
        document_ids: Iterable[str],
    ) -> Optional[dict[str, object]]:
        key = (tenant_id, case or "")
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
        tenant_id: str,
        case: Optional[str],
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
    ) -> Optional[dict[str, object]]:
        key = (tenant_id, case or "")
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
        self, *, tenant_id: str, case: Optional[str]
    ) -> Optional[dict[str, object]]:
        key = (tenant_id, case or "")
        with self._lock:
            record = self._ingestion_runs.get(key)
            if record is None:
                return None
            return record.as_payload()

    def reset(self) -> None:
        with self._lock:
            self._documents.clear()
            self._ingestion_runs.clear()
