"""Helpers to persist and retrieve ingestion run status metadata."""

from __future__ import annotations

from typing import Iterable, Sequence

from documents.repository import DEFAULT_LIFECYCLE_STORE


_LIFECYCLE_STORE = DEFAULT_LIFECYCLE_STORE


def _as_iterable(values: Iterable[str] | None) -> Iterable[str]:
    if not values:
        return ()
    return values


def record_ingestion_run_queued(
    tenant: str,
    case: str,
    run_id: str,
    document_ids: Sequence[str],
    *,
    queued_at: str,
    trace_id: str | None = None,
    embedding_profile: str | None = None,
    source: str | None = None,
    invalid_document_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Persist a queued ingestion run entry for later status lookups."""

    return _LIFECYCLE_STORE.record_ingestion_run_queued(
        tenant=tenant,
        case=case,
        run_id=run_id,
        document_ids=_as_iterable(document_ids),
        invalid_document_ids=_as_iterable(invalid_document_ids),
        queued_at=queued_at,
        trace_id=trace_id,
        embedding_profile=embedding_profile,
        source=source,
    )


def mark_ingestion_run_running(
    tenant: str,
    case: str,
    run_id: str,
    *,
    started_at: str,
    document_ids: Iterable[str] | None = None,
) -> dict[str, object] | None:
    """Update the persisted status when the worker begins processing."""

    return _LIFECYCLE_STORE.mark_ingestion_run_running(
        tenant=tenant,
        case=case,
        run_id=run_id,
        started_at=started_at,
        document_ids=_as_iterable(document_ids),
    )


def mark_ingestion_run_completed(
    tenant: str,
    case: str,
    run_id: str,
    *,
    finished_at: str,
    duration_ms: float,
    inserted_documents: int,
    replaced_documents: int,
    skipped_documents: int,
    inserted_chunks: int,
    invalid_document_ids: Iterable[str] | None = None,
    document_ids: Iterable[str] | None = None,
    error: str | None = None,
) -> dict[str, object] | None:
    """Persist the final state of an ingestion run."""

    return _LIFECYCLE_STORE.mark_ingestion_run_completed(
        tenant=tenant,
        case=case,
        run_id=run_id,
        finished_at=finished_at,
        duration_ms=duration_ms,
        inserted_documents=inserted_documents,
        replaced_documents=replaced_documents,
        skipped_documents=skipped_documents,
        inserted_chunks=inserted_chunks,
        invalid_document_ids=_as_iterable(invalid_document_ids),
        document_ids=_as_iterable(document_ids),
        error=error,
    )


def get_latest_ingestion_run(tenant: str, case: str) -> dict[str, object] | None:
    """Return the latest recorded ingestion run for the tenant/case pair."""

    payload = _LIFECYCLE_STORE.get_ingestion_run(tenant=tenant, case=case)
    if payload is None:
        return None
    if "run_id" not in payload:
        return None
    return payload


__all__ = [
    "get_latest_ingestion_run",
    "mark_ingestion_run_completed",
    "mark_ingestion_run_running",
    "record_ingestion_run_queued",
]
