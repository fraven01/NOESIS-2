"""Service-facing facades for ingestion and deletion flows.

These helpers centralise queue/outbox handling for ingestion, collection
creation and document deletion so that upstream tasks and LangGraph nodes
only depend on a single boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import UUID, uuid4

from django.utils import timezone

from ai_core.contracts.scope import ScopeContext

_DELETE_OUTBOX: list[dict[str, Any]] = []
_COLLECTION_OUTBOX: list[dict[str, Any]] = []


@dataclass(frozen=True)
class QueuedDeleteRequest:
    """Structured payload for queued hard delete requests."""

    tenant_id: str
    trace_id: str
    invocation_id: str
    ingestion_run_id: str
    document_ids: tuple[str, ...]
    queued_at: str
    case_id: str | None = None
    tenant_schema: str | None = None
    reason: str | None = None
    ticket_ref: str | None = None
    actor: Mapping[str, object] | None = None


def _coerce_uuid(value: object) -> str:
    return str(UUID(str(value)))


def ensure_collection(scope: ScopeContext, *, collection_id: str | None) -> dict[str, Any]:
    """Queue a collection ensure request with full scope context."""

    payload = {
        "tenant_id": scope.tenant_id,
        "collection_id": collection_id,
        "trace_id": scope.trace_id,
        "invocation_id": scope.invocation_id,
        "ingestion_run_id": scope.ingestion_run_id or scope.run_id,
        "tenant_schema": scope.tenant_schema,
        "queued_at": timezone.now().isoformat(),
    }
    _COLLECTION_OUTBOX.append(payload)
    return payload


def ingest_document(
    scope: ScopeContext,
    *,
    meta: Mapping[str, Any],
    chunks_path: str,
    embedding_state: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Embed and upsert chunks via the ingestion service boundary."""

    from ai_core import tasks as ingestion_tasks

    embed_result = ingestion_tasks.embed(dict(meta), chunks_path)
    embeddings_path = str(embed_result.get("path"))

    upsert_kwargs: dict[str, Any] = {}
    if embedding_state:
        tenant_schema = embedding_state.get("tenant_schema")
        if tenant_schema is not None:
            upsert_kwargs["tenant_schema"] = tenant_schema
        vector_client = embedding_state.get("client")
        if vector_client is not None:
            upsert_kwargs["vector_client"] = vector_client
        vector_factory = embedding_state.get("client_factory")
        if vector_factory is not None:
            upsert_kwargs["vector_client_factory"] = vector_factory

    inserted = ingestion_tasks.upsert(meta, embeddings_path, **upsert_kwargs)

    result: MutableMapping[str, Any] = {
        "status": "upserted" if inserted else "skipped",
        "chunks_inserted": int(inserted),
        "embeddings_path": embeddings_path,
        "trace_id": scope.trace_id,
        "ingestion_run_id": scope.ingestion_run_id,
        "case_id": scope.case_id,
        "tenant_schema": scope.tenant_schema,
    }

    if meta.get("embedding_profile"):
        result["embedding_profile"] = meta.get("embedding_profile")
    if meta.get("vector_space_id"):
        result["vector_space_id"] = meta.get("vector_space_id")

    return result


def delete_document(
    scope: ScopeContext,
    document_ids: Sequence[object],
    *,
    reason: str,
    ticket_ref: str,
    actor: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Queue a hard delete request with end-to-end context."""

    from ai_core.rag.vector_client import get_default_client

    normalized_ids = []
    for doc_id in document_ids:
        try:
            normalized_ids.append(_coerce_uuid(doc_id))
        except Exception:
            continue

    vector_client = get_default_client()
    delete_result = vector_client.hard_delete_documents(
        tenant_id=str(scope.tenant_id),
        document_ids=normalized_ids,
    )

    request = QueuedDeleteRequest(
        tenant_id=str(scope.tenant_id),
        trace_id=str(scope.trace_id),
        invocation_id=str(scope.invocation_id),
        ingestion_run_id=str(scope.ingestion_run_id or uuid4()),
        document_ids=tuple(normalized_ids),
        queued_at=timezone.now().isoformat(),
        case_id=str(scope.case_id) if scope.case_id else None,
        tenant_schema=scope.tenant_schema,
        reason=reason,
        ticket_ref=ticket_ref,
        actor=actor,
    )
    _DELETE_OUTBOX.append(request.__dict__)

    documents_deleted = int(delete_result.get("documents", 0))
    chunks_deleted = int(delete_result.get("chunks", 0))
    embeddings_deleted = int(delete_result.get("embeddings", 0))
    invalid_count = len(document_ids) - len(normalized_ids)
    missing_count = max(len(normalized_ids) - documents_deleted, 0)
    not_found = max(invalid_count + missing_count, 0)

    return {
        "status": "queued",
        "trace_id": request.trace_id,
        "ingestion_run_id": request.ingestion_run_id,
        "documents_requested": len(document_ids),
        "documents_deleted": documents_deleted,
        "chunks_deleted": chunks_deleted,
        "embeddings_deleted": embeddings_deleted,
        "not_found": not_found,
        "queued_at": request.queued_at,
        "tenant_schema": request.tenant_schema,
        "case_id": request.case_id,
        "actor": actor,
        "reason": reason,
        "ticket_ref": ticket_ref,
        "visibility": "deleted",
    }


__all__ = [
    "DELETE_OUTBOX",
    "QueuedDeleteRequest",
    "delete_document",
    "ensure_collection",
    "ingest_document",
]

# Re-export queues for test visibility
DELETE_OUTBOX = _DELETE_OUTBOX
