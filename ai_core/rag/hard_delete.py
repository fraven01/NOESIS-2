"""Queued hard-delete task for pgvector documents."""

from __future__ import annotations

import uuid
from typing import Mapping, MutableMapping, Sequence

from celery import shared_task
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.utils import timezone

from ai_core.authz.visibility import Visibility
from ai_core.contracts.scope import ScopeContext
from ai_core.infra.observability import record_span
from ai_core.tool_contracts.base import tool_context_from_meta
from common.celery import ScopedTask
from common.logging import get_logger
from documents.domain_service import DocumentDomainService
from documents.models import Document
from organizations.models import OrgMembership
from profiles.models import UserProfile

logger = get_logger(__name__)


class HardDeleteAuthorisationError(PermissionDenied):
    """Raised when the caller is not permitted to execute the hard delete."""


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

    if profile.role == UserProfile.Roles.TENANT_ADMIN:
        return label, "user_admin"

    if OrgMembership.objects.filter(user=user, role=OrgMembership.Role.ADMIN).exists():
        return label, "org_admin"

    raise HardDeleteAuthorisationError(
        "Hard delete requires an admin membership in the active organisation"
    )


def _emit_span(
    trace_id: str | None, metadata: MutableMapping[str, object | None]
) -> None:
    if not trace_id:
        return
    attributes = dict(metadata)
    attributes.setdefault("trace_id", str(trace_id))
    record_span("rag.hard_delete", attributes=attributes)


def _derive_session_salt(
    trace_id: str | None,
    case_id: str | None,
    tenant_id: str | None,
) -> str | None:
    parts = [trace_id, case_id, tenant_id]
    filtered = [str(part) for part in parts if part]
    if filtered:
        return "||".join(filtered)
    return None


def _build_scope(
    tenant_id: uuid.UUID,
    *,
    trace_id: str | None,
    ingestion_run_id: str | None,
    tenant_schema: str | None,
    case_id: str | None,  # Kept for backward compatibility but not used in ScopeContext
) -> ScopeContext:
    # BREAKING CHANGE (Option A): case_id no longer in ScopeContext
    # case_id is now a business domain ID, not infrastructure ID
    return ScopeContext(
        tenant_id=str(tenant_id),
        trace_id=trace_id or str(uuid.uuid4()),
        invocation_id=str(uuid.uuid4()),
        ingestion_run_id=ingestion_run_id or str(uuid.uuid4()),
        tenant_schema=tenant_schema,
    )


def _dispatch_delete_to_reaper(payload: Mapping[str, object]) -> None:
    trace_id = payload.get("trace_id") or payload.get("traceID")
    attributes = {
        "tenant_id": payload.get("tenant_id"),
        "document_ids": payload.get("document_ids"),
        "reason": payload.get("reason"),
        "trace_id": trace_id,
    }
    record_span("rag.reaper.dispatch", attributes=attributes)

    from ai_core.rag.vector_client import get_default_client

    vector_client = get_default_client()
    vector_client.hard_delete_documents(
        tenant_id=str(payload["tenant_id"]),
        document_ids=[uuid.UUID(doc_id) for doc_id in payload["document_ids"]],
    )


@shared_task(
    base=ScopedTask,
    name="rag.hard_delete",
    queue="rag_delete",
)
def hard_delete(  # type: ignore[override]
    state: Mapping[str, object],
    meta: Mapping[str, object] | None = None,
    *,
    actor: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Queue a hard delete for documents in the vector store."""

    state_payload = dict(state or {})
    if not isinstance(meta, Mapping):
        raise ValueError("meta with tool_context is required for rag.hard_delete")
    tool_context = tool_context_from_meta(meta)

    def _coerce_str(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        try:
            return str(value).strip() or None
        except Exception:
            return None

    tenant_id = _coerce_str(tool_context.scope.tenant_id)
    case_id = _coerce_str(tool_context.business.case_id)
    trace_id = _coerce_str(tool_context.scope.trace_id)
    tenant_schema = _coerce_str(tool_context.scope.tenant_schema)
    ingestion_run_id = _coerce_str(tool_context.scope.ingestion_run_id)
    reason = _coerce_str(state_payload.get("reason"))
    ticket_ref = _coerce_str(state_payload.get("ticket_ref"))
    document_ids = state_payload.get("document_ids") or []
    if not isinstance(document_ids, (list, tuple, set)):
        document_ids = [document_ids]

    if not tenant_id:
        raise ValueError("tenant_id is required for rag.hard_delete")
    if not reason:
        raise ValueError("reason is required for rag.hard_delete")
    if not ticket_ref:
        raise ValueError("ticket_ref is required for rag.hard_delete")

    operator, actor_mode = _resolve_actor(actor)
    requested_ids = _normalise_document_ids(document_ids)
    tenant_uuid = uuid.UUID(str(tenant_id))

    scope = _build_scope(
        tenant_uuid,
        trace_id=trace_id,
        ingestion_run_id=ingestion_run_id,
        tenant_schema=tenant_schema,
        case_id=case_id,
    )

    actor_payload = dict(actor or {})
    actor_payload.update({"operator": operator, "mode": actor_mode})

    def _dispatch_delete(payload: Mapping[str, object]) -> None:
        enriched: dict[str, object | None] = {
            **payload,
            "trace_id": scope.trace_id,
            "invocation_id": scope.invocation_id,
            "ingestion_run_id": scope.ingestion_run_id,
            "queued_at": timezone.now().isoformat(),
            # BREAKING CHANGE (Option A): case_id from function parameter, not scope
            "case_id": case_id,  # Was: scope.case_id
            "tenant_schema": scope.tenant_schema,
            "reason": reason,
            "ticket_ref": ticket_ref,
            "actor": actor_payload,
        }
        _dispatch_delete_to_reaper(enriched)

    domain_service = DocumentDomainService(deletion_dispatcher=_dispatch_delete)
    documents = list(
        Document.objects.filter(id__in=requested_ids, tenant_id=tenant_uuid)
    )
    for document in documents:
        domain_service.delete_document(document, reason=reason)

    deleted_ids = [str(doc.id) for doc in documents]

    log_payload = {
        "tenant": str(tenant_uuid),
        "documents_requested": len(requested_ids),
        "documents_deleted": len(deleted_ids),
        "reason": reason,
        "ticket_ref": ticket_ref,
        "operator": operator,
        "mode": actor_mode,
        "queued_at": timezone.now().isoformat(),
        "trace_id": scope.trace_id,
        "ingestion_run_id": scope.ingestion_run_id,
    }

    if case_id:
        log_payload["case_id"] = str(case_id)

    if tenant_schema:
        log_payload["tenant_schema"] = tenant_schema

    session_salt = _derive_session_salt(trace_id, case_id, tenant_id)
    if session_salt:
        log_payload["session_salt"] = session_salt

    logger.info("rag.hard_delete.audit", extra=log_payload)
    _emit_span(trace_id, log_payload)

    return {
        "status": "deleted",
        "operator": operator,
        "actor_mode": actor_mode,
        "deleted_ids": deleted_ids,
        "not_found": len(requested_ids) - len(deleted_ids),
        "visibility": Visibility.DELETED.value,
    }


__all__ = ["hard_delete"]
