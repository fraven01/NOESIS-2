"""Upload-related helpers for document ingestion views."""

from __future__ import annotations

import logging
from typing import Mapping
from uuid import UUID

from django.core.files.uploadedfile import UploadedFile
from rest_framework import status
from rest_framework.response import Response
from pydantic import ValidationError

from ai_core.contracts import BusinessContext, ScopeContext
from ai_core.graphs.transition_contracts import (
    GraphTransition,
    StandardTransitionResult,
)
from documents.contracts import DocumentMeta
from documents.domain_service import DocumentDomainService
from documents.models import DocumentCollection

from .graph_support import _error_response

logger = logging.getLogger(__name__)


def _derive_workflow_id(
    metadata: Mapping[str, object],
    business_context: BusinessContext | None = None,
) -> str:
    """Derive workflow ID from metadata or business context.

    BREAKING CHANGE (Option A): case_id is now in business_context, not scope_context.
    """
    candidate = metadata.get("workflow_id")
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if candidate:
            return candidate.replace(":", "_")

    # BREAKING CHANGE (Option A): use business_context only
    case_id = ""
    if business_context:
        case_id = str(business_context.case_id or "").strip()

    if not case_id:
        return "upload"
    return case_id.replace(":", "_")


def _infer_media_type(upload: UploadedFile) -> str:
    content_type = getattr(upload, "content_type", None)
    if isinstance(content_type, str):
        content_type = content_type.split(";")[0].strip().lower()
    if not content_type:
        return "application/octet-stream"
    return content_type


def _build_document_meta(
    scope_context: ScopeContext,
    metadata_obj: Mapping[str, object],
    external_id: str,
    *,
    media_type: str | None = None,
    business_context: BusinessContext | None = None,
) -> DocumentMeta:
    """Build DocumentMeta from scope and metadata.

    BREAKING CHANGE (Option A): business_context parameter added for workflow_id derivation.
    """
    workflow_id = _derive_workflow_id(metadata_obj, business_context)
    payload: dict[str, object] = {
        "tenant_id": str(scope_context.tenant_id),
        "workflow_id": workflow_id,
    }

    optional_fields = (
        "title",
        "language",
        "tags",
        "origin_uri",
        "crawl_timestamp",
        "external_ref",
        "parse_stats",
        "pipeline_config",
    )
    for field in optional_fields:
        if field in metadata_obj:
            payload[field] = metadata_obj[field]

    external_ref = dict(payload.get("external_ref") or {})
    if external_id:
        external_ref.setdefault("external_id", external_id)
    if media_type:
        external_ref.setdefault("media_type", media_type)
    if external_ref:
        payload["external_ref"] = external_ref

    return DocumentMeta(**payload)


def _ensure_document_collection_record(
    *,
    scope_context: ScopeContext,
    collection_id: UUID | str,
    source: str | None = None,
    key: str | None = None,
    label: str | None = None,
    business_context: BusinessContext | None = None,
) -> None:
    """Create or update a DocumentCollection row for manual developer uploads.

    BREAKING CHANGE (Option A): case_id is now in business_context, not scope_context.
    """

    if not collection_id:
        return
    try:
        collection_uuid = (
            collection_id
            if isinstance(collection_id, UUID)
            else UUID(str(collection_id))
        )
    except (TypeError, ValueError, AttributeError):
        extra_payload = {
            "scope_context": scope_context.model_dump(mode="json", exclude_none=True)
        }
        extra_payload["collection_id"] = collection_id
        logger.warning(
            "document_collection.ensure.invalid_id",
            extra=extra_payload,
        )
        return

    tenant_identifier = scope_context.tenant_id
    tenant_obj = None
    if tenant_identifier:
        try:
            from customers.tenant_context import TenantContext

            tenant_obj = TenantContext.resolve_identifier(
                tenant_identifier, allow_pk=True
            )
        except Exception:
            logger.exception(
                "document_collection.ensure_tenant_resolution_failed",
                extra={"tenant_id": tenant_identifier},
            )
            return

    if tenant_obj is None:
        logger.warning(
            "document_collection.ensure_missing_tenant",
            extra={"tenant_id": tenant_identifier},
        )
        return

    case_obj = None
    # BREAKING CHANGE (Option A): use business_context only
    case_id = business_context.case_id if business_context else None
    if case_id:
        from cases.models import Case

        case_obj = Case.objects.filter(
            tenant=tenant_obj,
            external_id=str(case_id).strip(),
        ).first()

    collection = DocumentCollection.objects.filter(
        tenant=tenant_obj, collection_id=collection_uuid
    ).first()
    if collection is None and key:
        collection = DocumentCollection.objects.filter(
            tenant=tenant_obj, key=key
        ).first()

    metadata_payload = {"source": source} if source else {}

    if collection is None:
        try:
            DocumentCollection.objects.create(
                tenant=tenant_obj,
                case=case_obj,
                name=label or key or "Manual Collection",
                key=key or str(collection_uuid),
                collection_id=collection_uuid,
                type="manual",
                visibility="tenant",
                metadata=metadata_payload,
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "document_collection.ensure_record_failed",
                extra={
                    "tenant_id": tenant_identifier,
                    "collection_id": str(collection_uuid),
                },
                exc_info=True,
            )
        return

    updates: dict[str, object] = {}
    if collection.collection_id != collection_uuid:
        updates["collection_id"] = collection_uuid
    if case_obj is not None and collection.case_id != case_obj.id:
        updates["case"] = case_obj
    if key and collection.key != key:
        updates["key"] = key
    if label and collection.name != label:
        updates["name"] = label
    if not collection.type:
        updates["type"] = "manual"
    if not collection.visibility:
        updates["visibility"] = "tenant"
    if metadata_payload:
        merged_metadata = dict(collection.metadata or {})
        if merged_metadata.get("source") != source:
            merged_metadata["source"] = source
            updates["metadata"] = merged_metadata

    if updates:
        for field, value in updates.items():
            setattr(collection, field, value)
        try:
            collection.save(update_fields=list(updates))
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "document_collection.ensure_record_update_failed",
                extra={
                    "tenant_id": tenant_identifier,
                    "tenant_schema": getattr(tenant_obj, "schema_name", None),
                    "collection_id": str(collection_uuid),
                },
                exc_info=True,
            )


def _ensure_collection_with_warning(
    service: DocumentDomainService,
    tenant,
    identifier: object,
    *,
    embedding_profile: str | None,
    scope: str | None,
) -> DocumentCollection | None:
    """Ensure a collection exists; create missing IDs with a warning (review later)."""

    try:
        collection_uuid = UUID(str(identifier))
    except Exception:
        collection_uuid = None

    if collection_uuid is not None:
        exists = DocumentCollection.objects.filter(
            tenant=tenant, collection_id=collection_uuid
        ).exists()
        if not exists:
            logger.warning(
                "documents.collection_missing_created",
                extra={
                    "tenant_id": str(getattr(tenant, "id", tenant)),
                    "collection_id": str(collection_uuid),
                    "reason": "missing_reference",
                },
            )

    return service.ensure_collection(
        tenant=tenant,
        key=str(identifier),
        embedding_profile=embedding_profile,
        scope=scope,
        collection_id=collection_uuid,
    )


def _coerce_transition_result(
    transition: object,
) -> StandardTransitionResult | None:
    if isinstance(transition, GraphTransition):
        return transition.result
    if isinstance(transition, StandardTransitionResult):
        return transition
    if isinstance(transition, Mapping):
        try:
            return StandardTransitionResult.model_validate(transition)
        except ValidationError:
            return None
    return None


def _map_upload_graph_skip(
    decision: str, transitions: Mapping[str, object]
) -> Response:
    """Translate upload graph skip decisions into HTTP error responses."""

    accept_transition = _coerce_transition_result(transitions.get("accept_upload"))
    guardrail_transition = _coerce_transition_result(
        transitions.get("delta_and_guardrails")
    )
    guardrail_section = guardrail_transition.guardrail if guardrail_transition else None

    if decision == "skip_guardrail":
        policy_events: tuple[str, ...]
        if guardrail_section:
            policy_events = guardrail_section.policy_events or ()
        else:
            policy_events = ()
        if policy_events:
            detail = f"Upload blocked by policy: {', '.join(policy_events)}."
        else:
            detail = "Upload blocked by guardrails."
        return _error_response(
            detail,
            "guardrail_rejected",
            status.HTTP_400_BAD_REQUEST,
        )

    if decision == "skip_disallowed_mime":
        mime = None
        if accept_transition is not None:
            context = accept_transition.context
            if isinstance(context, Mapping):
                candidate = context.get("mime")
                if candidate is not None:
                    mime = str(candidate)
        if mime:
            detail = f"MIME type '{mime}' is not allowed for uploads."
        else:
            detail = "MIME type is not allowed for uploads."
        return _error_response(
            detail,
            "mime_not_allowed",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if decision == "skip_oversize":
        max_bytes: int | None = None
        if accept_transition is not None:
            context = accept_transition.context
            if isinstance(context, Mapping):
                candidate = context.get("max_bytes")
                if isinstance(candidate, (int, float)):
                    max_bytes = int(candidate)
        if max_bytes is not None:
            detail = f"Uploaded file exceeds the allowed size of {max_bytes} bytes."
        else:
            detail = "Uploaded file exceeds the allowed size."
        return _error_response(
            detail,
            "file_too_large",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    if decision == "skip_invalid_input":
        reason = accept_transition.reason if accept_transition else "unknown"
        return _error_response(
            f"Upload payload is invalid: {reason}",
            "invalid_upload_payload",
            status.HTTP_400_BAD_REQUEST,
        )

    return _error_response(
        "Upload rejected due to validation failure.",
        "upload_rejected",
        status.HTTP_400_BAD_REQUEST,
    )
