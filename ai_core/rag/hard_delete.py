"""Automated hard-delete task for pgvector documents."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from celery import shared_task
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from psycopg2 import sql

from ai_core.authz.visibility import Visibility
from ai_core.infra import tracing
from ai_core.rag.vector_store import get_default_router, reset_default_router
from ai_core.rag import vector_client
from common.celery import ScopedTask
from common.logging import get_log_context, get_logger
from organizations.models import OrgMembership
from profiles.models import UserProfile

logger = get_logger(__name__)


class HardDeleteAuthorisationError(PermissionDenied):
    """Raised when the caller is not permitted to execute the hard delete."""


@dataclass
class HardDeleteStats:
    """Summary of the rows affected by a hard delete run."""

    requested: int
    documents: int
    chunks: int
    embeddings: int
    deleted_ids: list[str]
    not_found: int


def _normalise_document_ids(document_ids: Sequence[object]) -> list[uuid.UUID]:
    """Return a unique list of valid document UUIDs."""

    normalised: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in document_ids:
        if raw in {None, "", "None"}:
            continue
        try:
            value = uuid.UUID(str(raw))
        except (TypeError, ValueError):
            continue
        if value in seen:
            continue
        seen.add(value)
        normalised.append(value)
    return normalised


def _resolve_actor(actor: Mapping[str, object] | None) -> tuple[str, str]:
    """Validate the *actor* context and return an operator label and mode."""

    if actor is None:
        raise HardDeleteAuthorisationError(
            "Hard delete requires an authenticated operator"
        )

    allowed_keys = getattr(settings, "RAG_INTERNAL_KEYS", ()) or ()
    internal_key = actor.get("internal_key") if isinstance(actor, Mapping) else None
    if isinstance(internal_key, str) and internal_key in allowed_keys:
        label = str(actor.get("label") or actor.get("name") or internal_key)
        return label, "service_key"

    user_id = actor.get("user_id") if isinstance(actor, Mapping) else None
    if user_id is None:
        raise HardDeleteAuthorisationError(
            "Hard delete requires an admin user or service key"
        )

    user_model = get_user_model()
    try:
        user = user_model.objects.get(pk=user_id)
    except user_model.DoesNotExist as exc:  # pragma: no cover - defensive guard
        raise HardDeleteAuthorisationError("Unknown user for hard delete") from exc

    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist as exc:
        raise HardDeleteAuthorisationError(
            "User profile is required for hard delete"
        ) from exc

    if not profile.is_active:
        raise HardDeleteAuthorisationError("Inactive users cannot run hard delete")

    label = str(actor.get("label") or actor.get("name") or user.get_username())

    if profile.role == UserProfile.Roles.ADMIN:
        return label, "user_admin"

    if OrgMembership.objects.filter(user=user, role=OrgMembership.Role.ADMIN).exists():
        return label, "org_admin"

    raise HardDeleteAuthorisationError(
        "Hard delete requires an admin membership in the active organisation"
    )


def _resolve_store(tenant_id: str, tenant_schema: str | None) -> tuple[object, str]:
    """Return the backing store and resolved scope for *tenant_id*."""

    router = get_default_router()
    scope = router._resolve_scope(str(tenant_id), tenant_schema)  # type: ignore[attr-defined]
    if scope is None:
        scope = router.default_scope
    store = router._get_store(scope)  # type: ignore[attr-defined]
    return store, scope


def _schema_for_store(store: object) -> str:
    schema = getattr(store, "_schema", None)
    if isinstance(schema, str) and schema:
        return schema
    return "rag"


def _collect_stats(
    store: object,
    tenant_uuid: uuid.UUID,
    doc_ids: Sequence[uuid.UUID],
) -> HardDeleteStats:
    if not doc_ids:
        return HardDeleteStats(
            requested=0,
            documents=0,
            chunks=0,
            embeddings=0,
            deleted_ids=[],
            not_found=0,
        )

    deleted_ids: list[str] = []
    chunk_count = 0
    embedding_count = 0

    with store.connection() as conn:  # type: ignore[attr-defined]
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE tenant_id = %s AND id = ANY(%s)
                    """,
                    (tenant_uuid, list(doc_ids)),
                )
                existing_ids = [row[0] for row in cur.fetchall()]
                if not existing_ids:
                    conn.rollback()
                    return HardDeleteStats(
                        requested=len(doc_ids),
                        documents=0,
                        chunks=0,
                        embeddings=0,
                        deleted_ids=[],
                        not_found=len(doc_ids),
                    )

                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ANY(%s)",
                    (existing_ids,),
                )
                chunk_row = cur.fetchone()
                if chunk_row and chunk_row[0] is not None:
                    chunk_count = int(chunk_row[0])

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM embeddings
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE document_id = ANY(%s)
                    )
                    """,
                    (existing_ids,),
                )
                embedding_row = cur.fetchone()
                if embedding_row and embedding_row[0] is not None:
                    embedding_count = int(embedding_row[0])

                cur.execute(
                    """
                    DELETE FROM documents
                    WHERE tenant_id = %s AND id = ANY(%s)
                    RETURNING id
                    """,
                    (tenant_uuid, existing_ids),
                )
                deleted_ids = [str(row[0]) for row in cur.fetchall()]

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return HardDeleteStats(
        requested=len(doc_ids),
        documents=len(deleted_ids),
        chunks=chunk_count,
        embeddings=embedding_count,
        deleted_ids=deleted_ids,
        not_found=len(doc_ids) - len(deleted_ids),
    )


def _run_vacuum(store: object, schema: str) -> bool:
    vacuum_enabled = getattr(settings, "RAG_HARD_DELETE_VACUUM", False)
    if not vacuum_enabled:
        return False

    with store.connection() as conn:  # type: ignore[attr-defined]
        prev_autocommit = getattr(conn, "autocommit", False)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("VACUUM (VERBOSE, ANALYZE) {}.{}").format(
                        sql.Identifier(schema), sql.Identifier("documents")
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "rag.hard_delete.vacuum_failed",
                extra={"schema": schema, "error": str(exc)},
            )
            return False
        finally:
            conn.autocommit = prev_autocommit
    return True


def _run_reindex(store: object, schema: str, stats: HardDeleteStats) -> bool:
    threshold = getattr(settings, "RAG_HARD_DELETE_REINDEX_THRESHOLD", None)
    if not threshold:
        return False
    try:
        threshold_value = int(threshold)
    except (TypeError, ValueError):
        return False
    if stats.documents < max(1, threshold_value):
        return False

    with store.connection() as conn:  # type: ignore[attr-defined]
        prev_autocommit = getattr(conn, "autocommit", False)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("REINDEX TABLE CONCURRENTLY {}.{}").format(
                        sql.Identifier(schema), sql.Identifier("documents")
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "rag.hard_delete.reindex_failed",
                extra={"schema": schema, "error": str(exc)},
            )
            return False
        finally:
            conn.autocommit = prev_autocommit
    return True


def _emit_span(
    trace_id: str | None,
    metadata: MutableMapping[str, object | None],
) -> None:
    if not trace_id:
        return
    tracing.emit_span(
        trace_id=str(trace_id),
        node_name="rag.hard_delete",
        metadata=dict(metadata),
    )


@shared_task(base=ScopedTask, name="rag.hard_delete")
def hard_delete(  # type: ignore[override]
    tenant_id: str,
    document_ids: Sequence[object],
    reason: str,
    ticket_ref: str,
    *,
    actor: Mapping[str, object] | None = None,
    tenant_schema: str | None = None,
    trace_id: str | None = None,  # noqa: ARG001
) -> Mapping[str, object]:
    """Physically delete documents and related rows from the vector store."""

    if not tenant_id:
        raise ValueError("tenant_id is required for rag.hard_delete")

    operator, actor_mode = _resolve_actor(actor)
    requested_ids = _normalise_document_ids(document_ids)
    tenant_uuid = uuid.UUID(str(tenant_id))

    store, scope = _resolve_store(tenant_id, tenant_schema)
    schema = _schema_for_store(store)

    start = time.perf_counter()
    stats = _collect_stats(store, tenant_uuid, requested_ids)
    vacuum_performed = _run_vacuum(store, schema)
    reindex_performed = _run_reindex(store, schema, stats)
    reset_default_router()
    vector_client.reset_default_client()

    duration_ms = (time.perf_counter() - start) * 1000
    completed_at = timezone.now().isoformat()

    log_payload = {
        "tenant": str(tenant_uuid),
        "scope": scope,
        "schema": schema,
        "documents_requested": stats.requested,
        "documents_deleted": stats.documents,
        "chunks_deleted": stats.chunks,
        "embeddings_deleted": stats.embeddings,
        "not_found": stats.not_found,
        "reason": reason,
        "ticket_ref": ticket_ref,
        "operator": operator,
        "mode": actor_mode,
        "duration_ms": duration_ms,
        "vacuum": vacuum_performed,
        "reindex": reindex_performed,
    }

    logger.info("rag.hard_delete.audit", extra=log_payload)

    _emit_span(trace_id, log_payload)

    return {
        "tenant_id": str(tenant_uuid),
        "scope": scope,
        "schema": schema,
        "documents_requested": stats.requested,
        "documents_deleted": stats.documents,
        "chunks_deleted": stats.chunks,
        "embeddings_deleted": stats.embeddings,
        "not_found": stats.not_found,
        "deleted_ids": stats.deleted_ids,
        "vacuum_performed": vacuum_performed,
        "reindex_performed": reindex_performed,
        "operator": operator,
        "actor_mode": actor_mode,
        "reason": reason,
        "ticket_ref": ticket_ref,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "visibility": Visibility.DELETED.value,
    }


__all__ = ["hard_delete"]
