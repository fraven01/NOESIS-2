"""Document upload orchestration."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
from collections.abc import Mapping
from uuid import UUID, uuid4

from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone
from pydantic import ValidationError
from rest_framework import status
from rest_framework.response import Response

from ai_core.contracts.business import BusinessContext
from ai_core.rag.collections import (
    MANUAL_COLLECTION_LABEL,
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
    manual_collection_uuid,
)
from ai_core.rag.vector_client import get_default_client
from ai_core.tool_contracts.base import tool_context_from_meta
from customers.tenant_context import TenantContext
from documents.contracts import DocumentRef, InlineBlob, NormalizedDocument
from documents.domain_service import DocumentDomainService

from ..ingestion_utils import make_fallback_external_id
from ..schemas import RagUploadMetadata
from .graph_support import _error_response
from .repository import _get_documents_repository
from .upload_support import (
    _build_document_meta,
    _ensure_collection_with_warning,
    _ensure_document_collection_record,
    _infer_media_type,
    _map_upload_graph_skip,
)

logger = logging.getLogger(__name__)


def handle_document_upload(
    upload: UploadedFile,
    metadata_raw: str | bytes | None,
    meta: dict,
    idempotency_key: str | None,
) -> Response:
    """
    Orchestrates the upload of a document, its metadata, and immediate ingestion.

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

    metadata_obj = metadata_model.model_dump()

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

    manual_collection_scope = str(manual_collection_uuid(tenant_obj))

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
            ensure_manual_collection(tenant_obj)
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

    tenant = tenant_obj
    domain_service = (
        DocumentDomainService(vector_store=get_default_client()) if tenant else None
    )
    ensured_collection = None
    if metadata_obj.get("collection_id") and domain_service and tenant:
        if manual_scope_assigned:
            ensured_collection = domain_service.ensure_collection(
                tenant=tenant,
                key=MANUAL_COLLECTION_SLUG,
                collection_id=UUID(manual_collection_scope),
                embedding_profile=metadata_obj.get("embedding_profile"),
                scope=metadata_obj.get("scope"),
            )
        else:
            ensured_collection = _ensure_collection_with_warning(
                domain_service,
                tenant,
                metadata_obj["collection_id"],
                embedding_profile=metadata_obj.get("embedding_profile"),
                scope=metadata_obj.get("scope"),
            )
        if ensured_collection is not None:
            metadata_obj["collection_id"] = str(ensured_collection.collection_id)

    document_uuid = uuid4()

    file_bytes = upload.read()
    if not isinstance(file_bytes, (bytes, bytearray)):
        file_bytes = bytes(file_bytes)

    original_name = getattr(upload, "name", "") or "upload.bin"

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

    checksum = hashlib.sha256(file_bytes).hexdigest()
    encoded_blob = base64.b64encode(file_bytes).decode("ascii")

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

    audit_meta = {
        "created_by_user_id": scope_context.user_id,
        "initiated_by_user_id": scope_context.user_id,
        "last_hop_service_id": scope_context.service_id,
    }
    audit_meta = {key: value for key, value in audit_meta.items() if value}

    if domain_service and tenant:
        ingest_result = domain_service.ingest_document(
            tenant=tenant,
            source=document_metadata_payload.get("origin_uri")
            or document_metadata_payload.get("external_id")
            or original_name,
            content_hash=checksum,
            metadata=document_metadata_payload,
            audit_meta=audit_meta,
            collections=(
                (ensured_collection,) if ensured_collection is not None else ()
            ),
            embedding_profile=metadata_obj.get("embedding_profile"),
            scope=metadata_obj.get("scope"),
            dispatcher=lambda *_: None,
        )
        document_uuid = ingest_result.document.id
        collection_ids = ingest_result.collection_ids
    else:
        document_uuid = uuid4()
        collection_ids: tuple[UUID, ...] = ()

    if not metadata_obj.get("title"):
        metadata_obj["title"] = original_name

    # BREAKING CHANGE (Option A): Pass business_context to _build_document_meta
    document_meta = _build_document_meta(
        scope_context,
        metadata_obj,
        external_id,
        media_type=detected_mime,
        business_context=business_context,
    )
    if not business_context.workflow_id:
        business_context = business_context.model_copy(
            update={"workflow_id": document_meta.workflow_id}
        )

    logger.debug(
        "upload.collection_resolution",
        extra={
            "tenant_id": scope_context.tenant_id,
            "trace_id": scope_context.trace_id,
            "invocation_id": scope_context.invocation_id,
            "collection_ids": collection_ids,
            "metadata_collection_id": metadata_obj.get("collection_id"),
        },
    )
    ref_payload: dict[str, object] = {
        "tenant_id": document_meta.tenant_id,
        "workflow_id": document_meta.workflow_id,
        "document_id": document_uuid,
        "collection_id": (
            metadata_obj.get("collection_id")
            if metadata_obj.get("collection_id")
            else (collection_ids[0] if collection_ids else None)
        ),
        "version": metadata_obj.get("version"),
    }
    document_ref = DocumentRef(**ref_payload)

    blob = InlineBlob(
        type="inline",
        media_type=detected_mime,
        base64=encoded_blob,
        sha256=checksum,
        size=len(file_bytes),
    )

    normalized_document = NormalizedDocument(
        ref=document_ref,
        meta=document_meta,
        blob=blob,
        checksum=checksum,
        created_at=timezone.now(),
        source="upload",
    )

    ingestion_run_id = uuid4().hex

    try:
        repository = _get_documents_repository()

        # Prepare input for Universal Ingestion Graph
        # We pre-build the NormalizedDocument here to handle Django-specific file handling
        graph_input = {
            "normalized_document": normalized_document.model_dump(),
        }

        # BREAKING CHANGE (Option A - Full ToolContext Migration):
        # Universal ingestion graph now expects nested ToolContext structure
        business = BusinessContext(
            case_id=business_context.case_id,
            workflow_id=business_context.workflow_id,
            collection_id=(
                str(ensured_collection.collection_id)
                if ensured_collection
                else str(metadata_obj.get("collection_id"))
            ),
        )

        scope_payload = scope_context.model_dump(mode="json", exclude_none=True)
        graph_state = {
            "input": graph_input,
            "context": {
                "scope": {
                    **scope_payload,
                    "invocation_id": scope_context.invocation_id or uuid4().hex,
                    "ingestion_run_id": ingestion_run_id,
                },
                "business": business.model_dump(mode="json"),
                "metadata": {
                    # Runtime dependencies passed in metadata
                    "runtime_repository": repository,
                    "audit_meta": audit_meta,
                },
            },
        }

        # Invoke Universal Ingestion Graph
        # Imports are lazy to avoid circular dependency issues during module load if any
        from ai_core.graphs.technical.universal_ingestion_graph import (
            UniversalIngestionError as UploadIngestionError,
            build_universal_ingestion_graph,
        )

        universal_graph = build_universal_ingestion_graph()
        result_state = universal_graph.invoke(graph_state)

        output = result_state.get("output") or {}
        decision = output.get("decision", "error")
        reason = output.get("reason", "unknown")

        # P1 Fix: Handle both 'error' and 'failed' decisions from the graph
        if decision in ("error", "failed"):
            detail = f"Ingestion failed: {reason}"
            return _error_response(
                detail,
                "ingestion_failed",
                status.HTTP_502_BAD_GATEWAY,
            )

        if decision == "skipped":
            transitions = result_state.get("transitions") or {}
            if not isinstance(transitions, Mapping):
                transitions = {}
            skip_reason = output.get("reason_code")
            if skip_reason:
                skip_reason = str(skip_reason)
            if skip_reason and skip_reason.startswith("skip_"):
                return _map_upload_graph_skip(skip_reason, transitions)
    except UploadIngestionError as exc:
        return _error_response(
            str(exc),
            "ingestion_failed",
            status.HTTP_502_BAD_GATEWAY,
        )
    except ValidationError as exc:
        return _error_response(
            str(exc), "invalid_upload_payload", status.HTTP_400_BAD_REQUEST
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "upload.ingestion_failed",
            extra={
                "tenant_id": scope_context.tenant_id,
                "trace_id": scope_context.trace_id,
                "invocation_id": scope_context.invocation_id,
                "document_id": str(document_uuid),
            },
        )
        return _error_response(
            "Upload ingestion failed.",
            "ingestion_failed",
            status.HTTP_502_BAD_GATEWAY,
        )

    response_payload = {
        "status": "accepted",
        "document_id": str(document_uuid),
        "collection_id": metadata_obj.get("collection_id"),
        "workflow_id": document_meta.workflow_id,
        "tenant_id": document_meta.tenant_id,
        "trace_id": scope_context.trace_id,
        "ingestion_run_id": ingestion_run_id,
    }
    return Response(response_payload, status=status.HTTP_202_ACCEPTED)
