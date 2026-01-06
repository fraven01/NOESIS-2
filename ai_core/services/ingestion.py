"""Ingestion run orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from importlib import import_module
from uuid import UUID, uuid4

from django.utils import timezone
from pydantic import ValidationError
from rest_framework import status
from rest_framework.response import Response

from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.tools import InputError
from common.celery import with_scope_apply_async

from ..case_events import emit_ingestion_case_event as _emit_ingestion_case_event
from ..infra import object_store
from ..ingestion import partition_document_ids, run_ingestion
from ..ingestion_status import record_ingestion_run_queued
from ..rag.ingestion_contracts import (
    map_ingestion_error_to_status,
    resolve_ingestion_profile as _base_resolve_ingestion_profile,
)
from ..schemas import RagIngestionRunRequest
from .graph_support import _error_response

logger = logging.getLogger(__name__)


def _resolve_emit_ingestion_case_event():  # type: ignore[no-untyped-def]
    """Allow tests to monkeypatch ingestion case events via ai_core.services."""
    try:
        services = import_module("ai_core.services")
        candidate = getattr(services, "emit_ingestion_case_event", None)
        if callable(candidate):
            return candidate
    except (ImportError, AttributeError):
        pass
    return _emit_ingestion_case_event


# Allow tests to monkeypatch the run_ingestion task via ai_core.views.run_ingestion
# while keeping a sane default binding for production code paths.
RUN_INGESTION = run_ingestion


def _get_run_ingestion_task():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        task = getattr(views, "run_ingestion", None)
        if task is not None:
            return task
    except (ImportError, AttributeError):
        pass
    return RUN_INGESTION


def _enqueue_ingestion_task(
    task: object,
    *,
    state: Mapping[str, object],
    meta: Mapping[str, object],
) -> None:
    signature = task.s(state, meta)
    try:
        context = tool_context_from_meta(meta)
        scope_dict = context.scope.model_dump(mode="json", exclude_none=True)
    except (TypeError, ValueError):
        scope_dict = {}
    with_scope_apply_async(signature, scope_dict)


def _get_partition_document_ids():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "partition_document_ids", None)
        if callable(fn):
            return fn
    except Exception:
        pass
    return partition_document_ids


def _resolve_ingestion_profile(profile: str):  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "resolve_ingestion_profile", None)
        if callable(fn):
            return fn(profile)
    except Exception:
        pass
    return _base_resolve_ingestion_profile(profile)


def _persist_collection_scope(
    tenant_id: str, case_id: str, document_ids: Iterable[str], collection_id: str
) -> None:
    tenant_segment = object_store.sanitize_identifier(tenant_id)
    case_segment = object_store.sanitize_identifier(case_id or "uncased")
    for document_id in document_ids:
        if not document_id:
            continue
        path = f"{tenant_segment}/{case_segment}/uploads/{document_id}.meta.json"
        try:
            metadata = object_store.read_json(path)
        except FileNotFoundError:
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["collection_id"] = collection_id
        object_store.write_json(path, metadata)


def _ensure_document_collection(
    *,
    collection_id: object,
    tenant_identifier: object,
    case_identifier: object | None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    try:
        collection_uuid = UUID(str(collection_id))
    except (TypeError, ValueError):
        logger.debug(
            "collection_model.invalid_collection_id",
            extra={"collection_id": collection_id},
        )
        return

    from customers.tenant_context import TenantContext

    try:
        tenant = TenantContext.resolve_identifier(tenant_identifier, allow_pk=True)
    except Exception:
        logger.exception(
            "collection_model.tenant_resolution_failed",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    if tenant is None:
        logger.info(
            "collection_model.missing_tenant",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    try:
        from cases.models import Case
        from documents.models import DocumentCollection
    except Exception:
        logger.exception(
            "collection_model.import_failed",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    meta_payload = metadata if isinstance(metadata, Mapping) else {}

    def _coerce_label(*keys: str) -> str | None:
        for key in keys:
            value = meta_payload.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return None

    name = _coerce_label("collection_name", "name", "label", "title")
    key = _coerce_label("collection_key", "key", "slug")
    fallback_label = str(collection_uuid)
    if not name:
        name = fallback_label
    if not key:
        key = name

    case_obj = None
    if case_identifier:
        case_identifier_str = str(case_identifier)
        case_obj = Case.objects.filter(
            external_id=case_identifier_str, tenant=tenant
        ).first()

        if case_obj is None:
            try:
                case_uuid = UUID(case_identifier_str)
            except Exception:
                logger.debug(
                    "collection_model.case_resolution_failed",
                    extra={
                        "collection_id": str(collection_uuid),
                        "case_identifier": case_identifier,
                    },
                )
            else:
                case_obj = Case.objects.filter(id=case_uuid, tenant=tenant).first()

    try:
        DocumentCollection.objects.get_or_create(
            id=collection_uuid,
            defaults={
                "tenant": tenant,
                "case": case_obj,
                "collection_id": collection_uuid,
                "name": name,
                "key": key,
                "metadata": meta_payload,
                "type": "",
                "visibility": "",
            },
        )
    except Exception:
        logger.exception(
            "collection_model.get_or_create_failed",
            extra={
                "collection_id": str(collection_uuid),
                "tenant_id": getattr(tenant, "schema_name", str(tenant)),
            },
        )
        return


def start_ingestion_run(
    request_data: dict, meta: dict, idempotency_key: str | None
) -> Response:
    """
    Orchestrates the start of a RAG ingestion run.

    This function contains the business logic that was previously in the
    `RagIngestionRunView`.
    """
    try:
        validated_data = RagIngestionRunRequest.model_validate(request_data)
    except ValidationError as exc:
        # NOTE: Consider a more detailed error mapping for production
        return _error_response(
            str(exc), "validation_error", status.HTTP_400_BAD_REQUEST
        )

    tool_context = tool_context_from_meta(meta)
    tenant_schema = tool_context.scope.tenant_schema or meta.get("tenant_schema")

    collection_scope = getattr(validated_data, "collection_id", None)
    # BREAKING CHANGE (Option A): collection_id goes to business_context, not scope_context
    if collection_scope:
        updated_business = tool_context.business.model_copy(
            update={"collection_id": collection_scope}
        )
        tool_context = tool_context.model_copy(update={"business": updated_business})
        meta["business_context"] = updated_business.model_dump(
            mode="json", exclude_none=True
        )
        meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)

    if collection_scope:
        _ensure_document_collection(
            collection_id=collection_scope,
            tenant_identifier=tenant_schema or tool_context.scope.tenant_id,
            case_identifier=tool_context.business.case_id,
            metadata=request_data if isinstance(request_data, Mapping) else None,
        )

    try:
        normalized_profile = (
            str(validated_data.embedding_profile).strip()
            if hasattr(validated_data, "embedding_profile")
            else str(validated_data.get("embedding_profile", "")).strip()
        )
        profile_binding = _resolve_ingestion_profile(normalized_profile)
    except InputError as exc:
        return _error_response(
            exc.message,
            exc.code,
            map_ingestion_error_to_status(exc.code),
        )
    resolved_profile_id = profile_binding.profile_id

    ingestion_run_id = uuid4().hex
    queued_at = timezone.now().isoformat()

    # BREAKING CHANGE (Option A): case_id from business_context
    valid_document_ids, invalid_document_ids = _get_partition_document_ids()(
        tool_context.scope.tenant_id,
        tool_context.business.case_id,
        validated_data.document_ids,
    )

    to_dispatch = (
        valid_document_ids if valid_document_ids else validated_data.document_ids
    )
    if collection_scope:
        _persist_collection_scope(
            tool_context.scope.tenant_id,
            tool_context.business.case_id,
            to_dispatch,
            collection_scope,
        )
    if not tool_context.scope.ingestion_run_id:
        updated_scope = tool_context.scope.model_copy(
            update={"ingestion_run_id": ingestion_run_id}
        )
        tool_context = tool_context.model_copy(update={"scope": updated_scope})
        meta["scope_context"] = updated_scope.model_dump(mode="json", exclude_none=True)
        meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)
    state_payload: dict[str, object] = {
        "tenant_id": tool_context.scope.tenant_id,
        "case_id": tool_context.business.case_id,
        "document_ids": to_dispatch,
        "embedding_profile": resolved_profile_id,
        "tenant_schema": tenant_schema,
        "run_id": ingestion_run_id,
        "trace_id": tool_context.scope.trace_id,
    }
    if collection_scope:
        state_payload["collection_id"] = collection_scope
    # BREAKING CHANGE (Option A): case_id from business_context
    _enqueue_ingestion_task(
        _get_run_ingestion_task(),
        state=state_payload,
        meta=dict(meta),
    )

    record_ingestion_run_queued(
        tenant_id=tool_context.scope.tenant_id,
        case=tool_context.business.case_id,
        run_id=ingestion_run_id,
        document_ids=to_dispatch,
        queued_at=queued_at,
        trace_id=tool_context.scope.trace_id,
        embedding_profile=validated_data.embedding_profile,
        source=validated_data.source,
    )
    emit_ingestion_case_event = _resolve_emit_ingestion_case_event()
    emit_ingestion_case_event(
        tool_context.scope.tenant_id,
        tool_context.business.case_id,
        run_id=ingestion_run_id,
        context="queued",
    )

    idempotent = bool(idempotency_key)
    response_payload = {
        "status": "queued",
        "queued_at": queued_at,
        "ingestion_run_id": ingestion_run_id,
        "trace_id": tool_context.scope.trace_id,
        "idempotent": idempotent,
    }

    if invalid_document_ids:
        response_payload["invalid_ids"] = invalid_document_ids
    else:
        response_payload["invalid_ids"] = []

    return Response(response_payload, status=status.HTTP_202_ACCEPTED)
