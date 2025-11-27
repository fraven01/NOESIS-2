"""Service-facing facades for ingestion and deletion flows.

These helpers centralise queue/outbox handling for ingestion, collection
creation and document deletion so that upstream tasks and LangGraph nodes
only depend on a single boundary.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import UUID, uuid4

from django.utils import timezone

from ai_core.contracts.scope import ScopeContext
from customers.tenant_context import TenantContext
from django_tenants.utils import schema_context
from documents.domain_service import (
    DocumentDomainService,
    IngestionDispatcher,
)

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
    dispatcher: IngestionDispatcher | None = None,
) -> MutableMapping[str, Any]:
    """Persist document metadata and queue vector ingestion."""

    dispatcher_fn = dispatcher
    if dispatcher_fn is None:
        raise ValueError("ingestion_dispatcher_required")

    tenant_identifier = meta.get("tenant_id") or scope.tenant_id
    if not tenant_identifier:
        raise ValueError("tenant_id_required")

    try:
        tenant = TenantContext.resolve_identifier(tenant_identifier, allow_pk=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("invalid_tenant_identifier") from exc

    collection_identifier = meta.get("collection_id")
    collections: tuple[str, ...]
    if collection_identifier:
        collections = (str(collection_identifier),)
    else:
        collections = ()

    metadata = dict(meta)
    metadata["chunks_path"] = chunks_path

    document_id = metadata.get("document_id")
    document_uuid: UUID | None = None
    if document_id:
        try:
            document_uuid = UUID(str(document_id))
        except Exception:
            document_uuid = None

    content_hash = metadata.get("hash") or metadata.get("content_hash")
    if not content_hash:
        raise ValueError("content_hash_required")

    source = (
        metadata.get("source")
        or metadata.get("origin_uri")
        or metadata.get("external_id")
        or metadata.get("workflow_id")
        or "unknown"
    )

    service = DocumentDomainService(ingestion_dispatcher=dispatcher_fn)
    tenant_schema_ctx = (
        schema_context(scope.tenant_schema)
        if scope.tenant_schema
        else nullcontext()
    )

    with tenant_schema_ctx:
        ingest_result = service.ingest_document(
            tenant=tenant,
            source=str(source),
            content_hash=str(content_hash),
            metadata=metadata,
            collections=collections,
            embedding_profile=metadata.get("embedding_profile"),
            scope=metadata.get("scope"),
            dispatcher=dispatcher_fn,
            document_id=document_uuid,
        )

    return {
        "status": "queued",
        "chunks_inserted": 0,
        "trace_id": scope.trace_id,
        "ingestion_run_id": scope.ingestion_run_id,
        "case_id": scope.case_id,
        "tenant_schema": scope.tenant_schema,
        "document_id": str(ingest_result.document.id),
        "collection_ids": [str(cid) for cid in ingest_result.collection_ids],
        "embedding_profile": metadata.get("embedding_profile"),
        "vector_space_id": metadata.get("vector_space_id"),
    }


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
