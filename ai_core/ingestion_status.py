"""Helpers to persist and retrieve ingestion run status metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable, Sequence

from redis import Redis

from ai_core.infra.config import get_config


class RedisIngestionStatusStore:
    """Redis-backed persistence for ingestion run lifecycle metadata."""

    def __init__(
        self,
        redis_client: Redis | None = None,
        *,
        key_prefix: str = "ai-core:ingestion:status",
    ) -> None:
        self._redis: Redis | None = redis_client
        self._key_prefix = key_prefix

    def _client(self) -> Redis:
        if self._redis is None:
            self._redis = _redis_client()
        return self._redis

    def _key(self, tenant_id: str, case: str | None) -> str:
        tenant_key = str(tenant_id).strip()
        case_key = str(case).strip() if case else "_uncased_"
        return f"{self._key_prefix}:{tenant_key}:{case_key}"

    def _load(self, tenant_id: str, case: str | None) -> dict[str, object] | None:
        raw = self._client().get(self._key(tenant_id, case))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _persist(
        self, tenant_id: str, case: str | None, payload: dict[str, object]
    ) -> None:
        self._client().set(self._key(tenant_id, case), json.dumps(payload))

    def record_ingestion_run_queued(
        self,
        *,
        tenant_id: str,
        case: str | None,
        run_id: str,
        document_ids: Iterable[str],
        invalid_document_ids: Iterable[str],
        queued_at: str,
        trace_id: str | None = None,
        collection_id: str | None = None,
        embedding_profile: str | None = None,
        source: str | None = None,
    ) -> dict[str, object]:
        normalized_ids = _normalize_sequence(document_ids)
        normalized_invalid = _normalize_sequence(invalid_document_ids)
        payload: dict[str, object] = {
            "run_id": str(run_id),
            "status": "queued",
            "queued_at": str(queued_at),
            "document_ids": list(normalized_ids),
            "invalid_document_ids": list(normalized_invalid),
        }
        tenant_identifier = _normalize_identifier(tenant_id)
        if tenant_identifier is not None:
            payload["tenant_id"] = tenant_identifier
        optional_values = {
            "trace_id": trace_id,
            "collection_id": collection_id,
            "embedding_profile": embedding_profile,
            "source": source,
        }
        for key, value in optional_values.items():
            normalized = _normalize_optional_string(value)
            if normalized is not None:
                payload[key] = normalized

        self._persist(tenant_id, case, payload)
        return payload

    def mark_ingestion_run_running(
        self,
        *,
        tenant_id: str,
        case: str | None,
        run_id: str,
        started_at: str,
        document_ids: Iterable[str],
    ) -> dict[str, object] | None:
        payload = self._load(tenant_id, case)
        if payload is None or payload.get("run_id") != run_id:
            return None

        payload["status"] = "running"
        payload["started_at"] = str(started_at)
        tenant_identifier = _normalize_identifier(tenant_id)
        if tenant_identifier is not None:
            payload.setdefault("tenant_id", tenant_identifier)

        normalized_ids = _normalize_sequence(document_ids)
        if normalized_ids:
            payload["document_ids"] = list(normalized_ids)

        self._persist(tenant_id, case, payload)
        return payload

    def mark_ingestion_run_completed(
        self,
        *,
        tenant_id: str,
        case: str | None,
        run_id: str,
        finished_at: str,
        duration_ms: float,
        inserted_documents: int,
        replaced_documents: int,
        skipped_documents: int,
        inserted_chunks: int,
        invalid_document_ids: Iterable[str],
        document_ids: Iterable[str],
        error: str | None,
    ) -> dict[str, object] | None:
        payload = self._load(tenant_id, case)
        if payload is None or payload.get("run_id") != run_id:
            return None

        payload["status"] = "failed" if error else "succeeded"
        payload["finished_at"] = str(finished_at)
        payload["duration_ms"] = float(duration_ms)
        payload["inserted_documents"] = int(inserted_documents)
        payload["replaced_documents"] = int(replaced_documents)
        payload["skipped_documents"] = int(skipped_documents)
        payload["inserted_chunks"] = int(inserted_chunks)
        payload["invalid_document_ids"] = list(
            _normalize_sequence(invalid_document_ids)
        )

        normalized_ids = _normalize_sequence(document_ids)
        if normalized_ids:
            payload["document_ids"] = list(normalized_ids)

        normalized_error = _normalize_optional_string(error)
        if normalized_error is not None:
            payload["error"] = normalized_error
        else:
            payload.pop("error", None)
        tenant_identifier = _normalize_identifier(tenant_id)
        if tenant_identifier is not None:
            payload.setdefault("tenant_id", tenant_identifier)

        self._persist(tenant_id, case, payload)
        return payload

    def get_ingestion_run(
        self, *, tenant_id: str, case: str | None
    ) -> dict[str, object] | None:
        payload = self._load(tenant_id, case)
        if payload is None:
            return None
        tenant_identifier = _normalize_identifier(tenant_id)
        if tenant_identifier is not None:
            payload.setdefault("tenant_id", tenant_identifier)
        document_ids = payload.get("document_ids")
        if not isinstance(document_ids, list):
            payload["document_ids"] = list(_normalize_sequence(document_ids or ()))
        invalid_ids = payload.get("invalid_document_ids")
        if not isinstance(invalid_ids, list):
            payload["invalid_document_ids"] = list(
                _normalize_sequence(invalid_ids or ())
            )
        return payload


@lru_cache(maxsize=1)
def _redis_client() -> Redis:
    return Redis.from_url(get_config().redis_url, decode_responses=True)


def _normalize_sequence(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        candidate = _normalize_identifier(value)
        if candidate is not None:
            normalized.append(candidate)
    return tuple(normalized)


def _normalize_identifier(value: object) -> str | None:
    candidate = str(value).strip()
    return candidate or None


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


_LIFECYCLE_STORE: RedisIngestionStatusStore = RedisIngestionStatusStore()


def _as_iterable(values: Iterable[str] | None) -> Iterable[str]:
    if not values:
        return ()
    return values


def record_ingestion_run_queued(
    tenant_id: str,
    case: str | None,
    run_id: str,
    document_ids: Sequence[str],
    *,
    queued_at: str,
    trace_id: str | None = None,
    collection_id: str | None = None,
    embedding_profile: str | None = None,
    source: str | None = None,
    invalid_document_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Persist a queued ingestion run entry for later status lookups."""

    return _LIFECYCLE_STORE.record_ingestion_run_queued(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        document_ids=_as_iterable(document_ids),
        invalid_document_ids=_as_iterable(invalid_document_ids),
        queued_at=queued_at,
        trace_id=trace_id,
        collection_id=collection_id,
        embedding_profile=embedding_profile,
        source=source,
    )


def mark_ingestion_run_running(
    tenant_id: str,
    case: str | None,
    run_id: str,
    *,
    started_at: str,
    document_ids: Iterable[str] | None = None,
) -> dict[str, object] | None:
    """Update the persisted status when the worker begins processing."""

    return _LIFECYCLE_STORE.mark_ingestion_run_running(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        started_at=started_at,
        document_ids=_as_iterable(document_ids),
    )


def mark_ingestion_run_completed(
    tenant_id: str,
    case: str | None,
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
        tenant_id=tenant_id,
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


def get_latest_ingestion_run(
    tenant_id: str, case: str | None
) -> dict[str, object] | None:
    """Return the latest recorded ingestion run for the tenant/case pair."""

    payload = _LIFECYCLE_STORE.get_ingestion_run(tenant_id=tenant_id, case=case)
    if payload is None:
        return None
    if "run_id" not in payload:
        return None
    return payload


__all__ = [
    "RedisIngestionStatusStore",
    "get_latest_ingestion_run",
    "mark_ingestion_run_completed",
    "mark_ingestion_run_running",
    "record_ingestion_run_queued",
]
