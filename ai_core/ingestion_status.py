"""Helpers to persist and retrieve ingestion run status metadata."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence

from .infra import object_store


def _ingestion_status_path(tenant: str, case: str) -> str:
    tenant_segment = object_store.sanitize_identifier(tenant)
    case_segment = object_store.sanitize_identifier(case)
    return f"{tenant_segment}/{case_segment}/ingestion/run_status.json"


def _normalise_ids(document_ids: Iterable[str] | None) -> list[str]:
    if not document_ids:
        return []
    return [str(value) for value in document_ids if value]


def _ensure_payload(
    existing: Mapping[str, object] | None,
    run_id: str,
) -> MutableMapping[str, object]:
    if isinstance(existing, Mapping) and existing.get("run_id") == run_id:
        return dict(existing)
    return {"run_id": run_id}


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

    status_path = _ingestion_status_path(tenant, case)
    try:
        existing = object_store.read_json(status_path)
    except FileNotFoundError:
        existing = None

    payload = _ensure_payload(existing, run_id)
    payload.update(
        {
            "status": "queued",
            "queued_at": queued_at,
            "document_ids": _normalise_ids(document_ids),
            "invalid_document_ids": _normalise_ids(invalid_document_ids),
        }
    )
    if trace_id:
        payload["trace_id"] = trace_id
    if embedding_profile:
        payload["embedding_profile"] = embedding_profile
    if source:
        payload["source"] = source

    object_store.write_json(status_path, payload)
    return payload


def mark_ingestion_run_running(
    tenant: str,
    case: str,
    run_id: str,
    *,
    started_at: str,
    document_ids: Iterable[str] | None = None,
) -> dict[str, object] | None:
    """Update the persisted status when the worker begins processing."""

    status_path = _ingestion_status_path(tenant, case)
    try:
        existing = object_store.read_json(status_path)
    except FileNotFoundError:
        return None

    if not isinstance(existing, Mapping):
        existing = None
    elif existing.get("run_id") != run_id:
        return None

    payload = _ensure_payload(existing, run_id)

    payload.update(
        {
            "status": "running",
            "started_at": started_at,
        }
    )
    if document_ids:
        payload["document_ids"] = _normalise_ids(document_ids)

    object_store.write_json(status_path, payload)
    return payload


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

    status_path = _ingestion_status_path(tenant, case)
    try:
        existing = object_store.read_json(status_path)
    except FileNotFoundError:
        existing = None

    if not isinstance(existing, Mapping):
        existing = None
    elif existing.get("run_id") != run_id:
        return None

    payload = _ensure_payload(existing, run_id)

    payload.update(
        {
            "status": "failed" if error else "succeeded",
            "finished_at": finished_at,
            "duration_ms": float(duration_ms),
            "inserted_documents": int(inserted_documents),
            "replaced_documents": int(replaced_documents),
            "skipped_documents": int(skipped_documents),
            "inserted_chunks": int(inserted_chunks),
            "invalid_document_ids": _normalise_ids(invalid_document_ids),
        }
    )
    if document_ids:
        payload["document_ids"] = _normalise_ids(document_ids)
    if error:
        payload["error"] = str(error)
    else:
        payload.pop("error", None)

    object_store.write_json(status_path, payload)
    return payload


def get_latest_ingestion_run(
    tenant: str, case: str
) -> dict[str, object] | None:
    """Return the latest recorded ingestion run for the tenant/case pair."""

    status_path = _ingestion_status_path(tenant, case)
    try:
        payload = object_store.read_json(status_path)
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
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
