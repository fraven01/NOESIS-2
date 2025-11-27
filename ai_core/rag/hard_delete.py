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
from common.celery import ScopedTask
from common.logging import get_logger
from documents.domain_service import DocumentDomainService
from documents.models import Document
from organizations.models import OrgMembership
from profiles.models import UserProfile
from .vector_client import get_default_client

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

    if profile.role == UserProfile.Roles.ADMIN:
        return label, "user_admin"

    if OrgMembership.objects.filter(user=user, role=OrgMembership.Role.ADMIN).exists():
        return label, "org_admin"

    raise HardDeleteAuthorisationError(
        "Hard delete requires an admin membership in the active organisation"
    )


def _emit_span(trace_id: str | None, metadata: MutableMapping[str, object | None]) -> None:
    if not trace_id:
        return
    try:
        record_span(
            "rag.hard_delete",
            trace_id=str(trace_id),
            attributes=dict(metadata),
        )
    except TypeError:
        try:
            record_span(str(trace_id), "rag.hard_delete", dict(metadata))  # type: ignore[misc]
        except Exception:
            pass


def _build_scope(
    tenant_id: uuid.UUID,
    *,
    trace_id: str | None,
    ingestion_run_id: str | None,
    tenant_schema: str | None,
    case_id: str | None,
) -> ScopeContext:
    return ScopeContext(
        tenant_id=str(tenant_id),
        trace_id=trace_id or str(uuid.uuid4()),
        invocation_id=str(uuid.uuid4()),
        ingestion_run_id=ingestion_run_id or str(uuid.uuid4()),
        tenant_schema=tenant_schema,
        case_id=case_id,
    )


@shared_task(
    base=ScopedTask,
    name="rag.hard_delete",
    queue="rag_delete",
    accepts_scope=True,
)
def hard_delete(  # type: ignore[override]
    tenant_id: str,
    document_ids: Sequence[object],
    reason: str,
    ticket_ref: str,
    *,
    actor: Mapping[str, object] | None = None,
    tenant_schema: str | None = None,
    case_id: str | None = None,
    trace_id: str | None = None,  # noqa: ARG001
    ingestion_run_id: str | None = None,
    session_salt: str | None = None,
    session_scope: Sequence[str] | None = None,  # noqa: ARG001
) -> Mapping[str, object]:
    """Queue a hard delete for documents in the vector store."""

    if not tenant_id:
        raise ValueError("tenant_id is required for rag.hard_delete")

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

    domain_service = DocumentDomainService(vector_store=get_default_client())
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
