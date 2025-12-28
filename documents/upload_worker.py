"""Upload Worker for processing asynchronous document uploads.

This worker handles the processing of uploaded files, including:
1. Persisting the raw content to the Object Store (FileBlob)
2. Creating the initial NormalizedDocument state
3. Dispatching the ingestion graph
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple, Dict
from uuid import UUID, uuid4


from ai_core.infra.blob_writers import ObjectStoreBlobWriter
from ai_core.infra import object_store
from ai_core.ids.http_scope import normalize_task_context
from ai_core.tasks import run_ingestion_graph
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    FileBlob,
    NormalizedDocument,
)
from documents.domain_service import DocumentDomainService
from documents.models import DocumentCollection
from common.celery import with_scope_apply_async
from common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerPublishResult:
    """Result of a worker publish operation."""

    status: str
    document_id: str
    task_id: str
    state: Mapping[str, Any]


class UploadWorker:
    """Worker for processing uploaded documents asynchronously.

    This worker follows the same pattern as the CrawlerWorker, but optimized
    for single-file uploads provided via API.
    """

    def __init__(self) -> None:
        self._domain_service: DocumentDomainService | None = None

    def process(
        self,
        upload_file: Any,  # Duck-typed UploadedFile
        *,
        tenant_id: str,
        case_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        document_metadata: Optional[Mapping[str, Any]] = None,
        ingestion_overrides: Optional[Mapping[str, Any]] = None,
    ) -> WorkerPublishResult:
        """Process upload and dispatch to ingestion graph.

        Args:
            upload_file: File-like object with .read(), .name, .content_type
            tenant_id: Tenant identifier
            case_id: Optional case identifier
            trace_id: Distributed tracing ID
            invocation_id: Unique invocation ID for this processing run
            document_metadata: Optional metadata provided by the uploader
            ingestion_overrides: Optional overrides for the ingestion process

        Returns:
            WorkerPublishResult with status and IDs
        """
        # 1. Read file
        file_bytes = upload_file.read()
        if not isinstance(file_bytes, (bytes, bytearray)):
            file_bytes = bytes(file_bytes)

        # 2. Persist to Object Store
        payload_path, checksum, size = self._persist_payload(
            file_bytes, tenant_id, case_id
        )

        # 3. Build metadata
        raw_meta = dict(document_metadata or {})
        raw_meta.setdefault("filename", upload_file.name)
        raw_meta.setdefault("content_type", upload_file.content_type)
        raw_meta.setdefault("content_hash", checksum)
        raw_meta.setdefault("content_length", size)
        raw_meta.setdefault("source", "upload")

        # 4. Collection ID handling
        collection_id = raw_meta.get("collection_id")
        if not collection_id and ingestion_overrides:
            collection_id = ingestion_overrides.get("collection_id")

        # 5. Register document (ID generation & visibility)
        resolved_document_id = self._register_document(
            tenant_id=tenant_id,
            filename=upload_file.name,
            content_hash=checksum,
            metadata=raw_meta,
            collection_identifier=collection_id,
            ingestion_overrides=ingestion_overrides,
        )

        # 6. Compose state (IDENTICAL to Crawler!)
        state = self._compose_state(
            payload_path,
            checksum,
            size,
            tenant_id,
            case_id,
            trace_id,
            resolved_document_id,
            raw_meta,
            ingestion_overrides,
        )

        # 7. Compose meta (IDENTICAL to Crawler!)
        meta = self._compose_meta(tenant_id, case_id, trace_id, invocation_id)

        # Observability: Track Celery payload metrics (MVP invariant verification)
        import json

        celery_payload_bytes = len(json.dumps(state, default=str).encode("utf-8"))
        blob_uri = (
            state.get("normalized_document_input", {}).get("blob", {}).get("uri", "")
        )
        uri_scheme = blob_uri.split("://")[0] if "://" in blob_uri else "unknown"
        logger.info(
            "upload.celery_dispatch",
            extra={
                "celery_payload_bytes": celery_payload_bytes,
                "blob_size_bytes": size,
                "uri_scheme": uri_scheme,
                "blob_uri": blob_uri[:100],  # Truncate for logging
            },
        )

        # 8. Dispatch
        signature = run_ingestion_graph.s(state, meta)
        scope_context = meta.get("scope_context")
        scope = dict(scope_context) if isinstance(scope_context, Mapping) else {}
        async_result = with_scope_apply_async(signature, scope)

        return WorkerPublishResult(
            status="published",
            document_id=resolved_document_id
            or state["normalized_document_input"]["ref"]["document_id"],
            task_id=async_result.id,
            state=state,
        )

    def _persist_payload(
        self,
        payload: bytes,
        tenant_id: str,
        case_id: Optional[str],
    ) -> Tuple[str, str, int]:
        """Persist raw upload payload to object store."""
        # Sanitize identifiers
        safe_tenant = object_store.sanitize_identifier(tenant_id)
        safe_case = object_store.sanitize_identifier(case_id) if case_id else "default"

        # Structure: <tenant>/<case>/uploads/<uuid>
        writer = ObjectStoreBlobWriter(prefix=[safe_tenant, safe_case, "uploads"])
        uri, checksum, size = writer.put(payload)
        return uri, checksum, size

    def _register_document(
        self,
        tenant_id: str,
        filename: str,
        content_hash: str,
        metadata: Dict[str, Any],
        collection_identifier: Optional[Any],
        ingestion_overrides: Optional[Mapping[str, Any]],
    ) -> Optional[str]:
        """Register document in DomainService to ensure it exists in DB."""
        try:
            from django_tenants.utils import tenant_context
            from customers.tenant_context import TenantContext

            # Resolve tenant
            tenant = TenantContext.resolve_identifier(tenant_id)
            if not tenant:
                return None

            service = self._get_domain_service()

            with tenant_context(tenant):
                collections = []
                if collection_identifier:
                    ensured = self._ensure_collection_with_warning(
                        service,
                        tenant,
                        collection_identifier,
                        embedding_profile=(
                            ingestion_overrides.get("embedding_profile")
                            if ingestion_overrides
                            else None
                        ),
                        scope=(
                            ingestion_overrides.get("scope")
                            if ingestion_overrides
                            else None
                        ),
                    )
                    if ensured:
                        collections.append(ensured)

                # We use a no-op dispatcher because we are about to dispatch the graph manually
                result = service.ingest_document(
                    tenant=tenant,
                    source=metadata.get("origin_uri") or filename,
                    content_hash=content_hash,
                    metadata=metadata,
                    collections=collections,
                    embedding_profile=(
                        ingestion_overrides.get("embedding_profile")
                        if ingestion_overrides
                        else None
                    ),
                    scope=(
                        ingestion_overrides.get("scope")
                        if ingestion_overrides
                        else None
                    ),
                    dispatcher=lambda *args: None,
                )
                return str(result.document.id)
        except Exception:
            logger.exception(
                "upload_worker.registration_failed",
                extra={"tenant_id": tenant_id, "filename": filename},
            )
            return None

    def _compose_state(
        self,
        payload_path: str,
        checksum: str,
        size: int,
        tenant_id: str,
        case_id: Optional[str],
        trace_id: Optional[str],
        resolved_document_id: Optional[str],
        raw_meta: Dict[str, Any],
        ingestion_overrides: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Compose the state dictionary for the ingestion graph."""
        state: Dict[str, Any] = dict(ingestion_overrides or {})
        state["tenant_id"] = tenant_id
        if case_id:
            state["case_id"] = case_id
        if trace_id:
            state["trace_id"] = trace_id

        # Raw document wrapper
        state["raw_document"] = {
            "metadata": raw_meta,
            "payload_path": payload_path,
        }
        if resolved_document_id:
            state["raw_document"]["document_id"] = resolved_document_id

        state["raw_payload_path"] = payload_path

        # Build NormalizedDocumentInput
        workflow_id = raw_meta.get("workflow_id") or "default"

        # Determine Doc ID
        if resolved_document_id:
            doc_uuid = UUID(resolved_document_id)
        else:
            doc_uuid = uuid4()

        # Determine Collection ID
        collection_id = raw_meta.get("collection_id")
        clean_collection_id: Optional[UUID] = None
        if collection_id:
            try:
                clean_collection_id = UUID(str(collection_id))
            except (ValueError, TypeError):
                pass

        ref = DocumentRef(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            document_id=doc_uuid,
            collection_id=clean_collection_id,
        )

        # Meta
        meta_obj = DocumentMeta(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            title=raw_meta.get("title") or raw_meta.get("filename"),
            language=raw_meta.get("language"),
            origin_uri=raw_meta.get("origin_uri"),
            tags=raw_meta.get("tags") or [],
            external_ref=raw_meta.get("external_ref"),
            pipeline_config={
                "media_type": raw_meta.get("content_type") or "application/octet-stream"
            },
        )

        # Blob (FileBlob)
        blob_obj = FileBlob(
            type="file",
            uri=payload_path,
            size=size,
            sha256=checksum,
            media_type=raw_meta.get("content_type") or "application/octet-stream",
        )

        normalized_doc = NormalizedDocument(
            ref=ref,
            meta=meta_obj,
            blob=blob_obj,
            checksum=checksum,
            source="upload",
            lifecycle_state="active",
            created_at=datetime.now(timezone.utc),
        )

        state["normalized_document_input"] = normalized_doc.model_dump(mode="json")
        return state

    def _compose_meta(
        self,
        tenant_id: str,
        case_id: Optional[str],
        trace_id: Optional[str],
        invocation_id: Optional[str],
    ) -> Dict[str, Any]:
        """Compose metadata for the graph execution context.

        BREAKING CHANGE (Option A - Strict Separation):
        Returns both scope_context (infrastructure) and business_context (domain).
        """
        from ai_core.contracts.business import BusinessContext

        # Build ScopeContext (infrastructure only, no business IDs)
        scope = normalize_task_context(
            tenant_id=tenant_id,
            case_id=case_id,  # DEPRECATED parameter, not used in ScopeContext
            service_id="upload-worker",
            trace_id=trace_id,
            invocation_id=invocation_id,
        )

        # Build BusinessContext (business domain IDs)
        business = BusinessContext(
            case_id=case_id,
            workflow_id=case_id,  # Default workflow_id to case_id if not provided
        )

        return {
            "scope_context": scope.model_dump(mode="json"),
            "business_context": business.model_dump(mode="json", exclude_none=True),
        }

    def _get_domain_service(self) -> DocumentDomainService:
        if self._domain_service is None:
            from ai_core.rag.vector_client import get_default_client

            self._domain_service = DocumentDomainService(
                vector_store=get_default_client()
            )
        return self._domain_service

    def _ensure_collection_with_warning(
        self,
        service: DocumentDomainService,
        tenant: Any,
        collection_id: Any,
        embedding_profile: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Optional[DocumentCollection]:
        from documents.contracts import MANUAL_COLLECTION_SLUG

        try:
            # Try strictly as UUID
            if isinstance(collection_id, str):
                collection_uuid = UUID(collection_id)
            elif isinstance(collection_id, UUID):
                collection_uuid = collection_id
            else:
                return None

            return service.ensure_collection(
                tenant=tenant,
                key=MANUAL_COLLECTION_SLUG,  # Start with manual slug default
                collection_id=collection_uuid,
                embedding_profile=embedding_profile,
                scope=scope,
            )
        except Exception:
            logger.warning("upload_worker.collection_ensure_failed", exc_info=True)
            return None
