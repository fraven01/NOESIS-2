"""Document upload orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from uuid import uuid4

from django.apps import apps
from django.core.files.uploadedfile import UploadedFile
from pydantic import ValidationError
from rest_framework import status
from rest_framework.response import Response
from django_tenants.utils import schema_context

from documents.collection_service import (
    CollectionService,
    DEFAULT_MANUAL_COLLECTION_LABEL as MANUAL_COLLECTION_LABEL,
    DEFAULT_MANUAL_COLLECTION_SLUG as MANUAL_COLLECTION_SLUG,
)
from ai_core.infra import object_store
from ai_core.tool_contracts.base import tool_context_from_meta
from common.celery import with_scope_apply_async
from customers.tenant_context import TenantContext
from documents.tasks import upload_document_task

from ..ingestion_utils import make_fallback_external_id
from ..schemas import RagUploadMetadata
from .graph_support import _error_response
from .upload_support import (
    _derive_workflow_id,
    _ensure_document_collection_record,
    _infer_media_type,
)

logger = logging.getLogger(__name__)


def handle_document_upload(
    upload: UploadedFile,
    metadata_raw: str | bytes | None,
    meta: dict,
    idempotency_key: str | None,
) -> Response:
    """
    Orchestrates the upload of a document and dispatches ingestion to a worker.

    This function contains the business logic that was previously in the
    `RagUploadView`.
    """
    metadata_obj: dict[str, object] | None = None
    if metadata_raw not in (None, ""):
        if isinstance(metadata_raw, (bytes, bytearray)):
            try:
                metadata_text = metadata_raw.decode("utf-8")
            except UnicodeDecodeError:
                return _error_response(
                    "Metadata must be valid JSON.",
                    "invalid_metadata",
                    status.HTTP_400_BAD_REQUEST,
                )
        else:
            metadata_text = str(metadata_raw)

        metadata_text = metadata_text.strip()
        if metadata_text:
            try:
                metadata_obj = json.loads(metadata_text)
            except json.JSONDecodeError:
                return _error_response(
                    "Metadata must be valid JSON.",
                    "invalid_metadata",
                    status.HTTP_400_BAD_REQUEST,
                )

    if metadata_obj is not None and not isinstance(metadata_obj, dict):
        return _error_response(
            "Metadata must be a JSON object when provided.",
            "invalid_metadata",
            status.HTTP_400_BAD_REQUEST,
        )

    if metadata_obj is None:
        metadata_obj = {}

    # BREAKING CHANGE (Option A - Strict Separation):
    # scope_context is infrastructure only, business_context has business IDs
    tool_context = tool_context_from_meta(meta)
    scope_context = tool_context.scope
    business_context = tool_context.business

    header_collection = business_context.collection_id
    if header_collection and not metadata_obj.get("collection_id"):
        metadata_obj["collection_id"] = header_collection

    try:
        metadata_model = RagUploadMetadata.model_validate(metadata_obj)
    except ValidationError as exc:
        return _error_response(
            str(exc), "invalid_metadata", status.HTTP_400_BAD_REQUEST
        )

    metadata_obj.update(metadata_model.model_dump())

    tenant_identifier = scope_context.tenant_id
    tenant_obj = None
    if tenant_identifier:
        try:
            tenant_obj = TenantContext.resolve_identifier(
                tenant_identifier, allow_pk=True
            )
        except Exception:
            pass

    if tenant_obj is None:
        return _error_response(
            "Tenant could not be resolved for upload.",
            "tenant_not_found",
            status.HTTP_400_BAD_REQUEST,
        )

    manual_collection_scope = str(CollectionService.manual_collection_uuid(tenant_obj))

    manual_scope_assigned = False

    existing_scope = metadata_obj.get("collection_id")
    if existing_scope:
        existing_scope_str = str(existing_scope)
        metadata_obj["collection_id"] = existing_scope_str
        if existing_scope_str == manual_collection_scope:
            manual_scope_assigned = True

    if not metadata_obj.get("collection_id"):
        manual_scope_assigned = True
        try:
            collection_service = CollectionService()
            collection_service.ensure_manual_collection(tenant_obj)
        except Exception:  # pragma: no cover - defensive guard
            # BREAKING CHANGE (Option A): case_id from business_context
            logger.warning(
                "upload.ensure_manual_collection_failed",
                extra={
                    "tenant_id": scope_context.tenant_id,
                    "case_id": business_context.case_id,
                },
                exc_info=True,
            )
        metadata_obj["collection_id"] = manual_collection_scope

    # BREAKING CHANGE (Option A): Pass business_context to _ensure_document_collection_record
    if manual_scope_assigned and metadata_obj.get("collection_id"):
        _ensure_document_collection_record(
            scope_context=scope_context,
            collection_id=metadata_obj["collection_id"],
            source="upload",
            key=MANUAL_COLLECTION_SLUG,
            label=MANUAL_COLLECTION_LABEL,
            business_context=business_context,
        )

    updated_business = business_context
    updated = False

    if not updated_business.workflow_id:
        derived_workflow_id = _derive_workflow_id(metadata_obj, business_context)
        updated_business = updated_business.model_copy(
            update={"workflow_id": derived_workflow_id}
        )
        updated = True

    collection_id = metadata_obj.get("collection_id")
    if collection_id and not updated_business.collection_id:
        updated_business = updated_business.model_copy(
            update={"collection_id": str(collection_id)}
        )
        updated = True

    if updated:
        business_context = updated_business

    ingestion_run_id = scope_context.ingestion_run_id or uuid4().hex
    if ingestion_run_id != scope_context.ingestion_run_id:
        scope_context = scope_context.model_copy(
            update={"ingestion_run_id": ingestion_run_id}
        )

    tool_context = tool_context.model_copy(
        update={"scope": scope_context, "business": business_context}
    )
    meta["scope_context"] = scope_context.model_dump(mode="json", exclude_none=True)
    meta["business_context"] = business_context.model_dump(
        mode="json", exclude_none=True
    )
    meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)

    original_name = getattr(upload, "name", "") or "upload.bin"
    if not metadata_obj.get("title"):
        metadata_obj["title"] = original_name
    if business_context.workflow_id and "workflow_id" not in metadata_obj:
        metadata_obj["workflow_id"] = business_context.workflow_id

    document_uuid = uuid4()

    file_bytes = upload.read()
    if not isinstance(file_bytes, (bytes, bytearray)):
        file_bytes = bytes(file_bytes)

    supplied_external = metadata_obj.get("external_id")
    if isinstance(supplied_external, str):
        supplied_external = supplied_external.strip()
    else:
        supplied_external = None

    if supplied_external:
        external_id = supplied_external
    else:
        external_id = make_fallback_external_id(
            original_name,
            getattr(upload, "size", None) or len(file_bytes),
            file_bytes,
        )

    metadata_obj["external_id"] = external_id
    external_ref = metadata_obj.get("external_ref")
    external_ref_payload: dict[str, object] | None = None
    if isinstance(external_ref, dict):
        external_ref_payload = dict(external_ref)
    elif external_ref is None:
        external_ref_payload = {}
    if external_ref_payload is not None and external_id:
        external_ref_payload.setdefault("external_id", external_id)
        if external_ref_payload:
            metadata_obj["external_ref"] = external_ref_payload

    checksum = hashlib.sha256(file_bytes).hexdigest()
    source = str(metadata_obj.get("source") or "upload")

    # Fail fast if the same document (tenant/source/hash) already exists.
    with schema_context(tenant_obj.schema_name):
        Document = apps.get_model("documents", "Document")
        existing_document = (
            Document.objects.filter(tenant=tenant_obj, source=source, hash=checksum)
            .only("id", "lifecycle_state", "soft_deleted_at")
            .first()
        )
    if existing_document and not existing_document.soft_deleted_at:
        return _error_response(
            "Document with the same content already exists.",
            "duplicate_document",
            status.HTTP_409_CONFLICT,
        )

    # Improved MIME type detection
    detected_mime = _infer_media_type(upload)
    if detected_mime == "application/octet-stream":
        guessed_type, _ = mimetypes.guess_type(original_name)
        if guessed_type:
            detected_mime = guessed_type

    document_metadata_payload = dict(metadata_obj)
    document_metadata_payload.setdefault("filename", original_name)
    document_metadata_payload.setdefault("content_hash", checksum)
    document_metadata_payload.setdefault("content_type", detected_mime)
    document_metadata_payload.setdefault("source", source)
    document_metadata_payload["external_id"] = external_id
    document_metadata_payload["document_id"] = str(document_uuid)

    try:
        tenant_segment = object_store.sanitize_identifier(scope_context.tenant_id)
        workflow_segment = object_store.sanitize_identifier(
            business_context.workflow_id or "upload"
        )
        case_segment = object_store.sanitize_identifier(
            business_context.case_id or business_context.workflow_id or "upload"
        )
        metadata_paths = [
            "/".join(
                [
                    tenant_segment,
                    workflow_segment,
                    "uploads",
                    f"{document_uuid}.meta.json",
                ]
            )
        ]
        if case_segment != workflow_segment:
            metadata_paths.append(
                "/".join(
                    [
                        tenant_segment,
                        case_segment,
                        "uploads",
                        f"{document_uuid}.meta.json",
                    ]
                )
            )
        for metadata_path in metadata_paths:
            object_store.write_json(metadata_path, document_metadata_payload)
    except Exception:
        logger.exception(
            "upload.metadata_persist_failed",
            extra={
                "tenant_id": scope_context.tenant_id,
                "case_id": business_context.case_id,
                "document_id": str(document_uuid),
            },
        )
        return _error_response(
            "Upload metadata persistence failed.",
            "upload_metadata_persist_failed",
            status.HTTP_502_BAD_GATEWAY,
        )

    try:
        signature = upload_document_task.s(
            file_bytes=file_bytes,
            filename=original_name,
            content_type=detected_mime,
            metadata=document_metadata_payload,
            meta=meta,
        )
        with_scope_apply_async(
            signature,
            scope_context.model_dump(mode="json", exclude_none=True),
            task_id=ingestion_run_id,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "upload.dispatch_failed",
            extra={
                "tenant_id": scope_context.tenant_id,
                "trace_id": scope_context.trace_id,
                "invocation_id": scope_context.invocation_id,
                "document_id": str(document_uuid),
            },
        )
        return _error_response(
            "Upload dispatch failed.",
            "upload_dispatch_failed",
            status.HTTP_502_BAD_GATEWAY,
        )

    response_payload = {
        "status": "accepted",
        "document_id": str(document_uuid),
        "collection_id": metadata_obj.get("collection_id"),
        "workflow_id": business_context.workflow_id,
        "tenant_id": scope_context.tenant_id,
        "trace_id": scope_context.trace_id,
        "ingestion_run_id": ingestion_run_id,
    }
    return Response(response_payload, status=status.HTTP_202_ACCEPTED)
