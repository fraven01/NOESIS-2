"""Development-only endpoints for manual ingestion and collection management."""

from __future__ import annotations

import logging
from typing import Mapping
from uuid import UUID

from django.conf import settings
from django.shortcuts import get_object_or_404
from django_tenants.utils import schema_context
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response

from customers.tenant_context import TenantContext
from documents.collection_service import CollectionService
from documents.domain_service import DocumentDomainService
from documents.lifecycle import DocumentLifecycleState
from documents.models import Document
from documents.serializers import DocumentSerializer


logger = logging.getLogger(__name__)


def _require_debug() -> None:
    if not settings.DEBUG:
        raise RuntimeError("development APIs are only available in DEBUG mode")


class DocumentDevViewSet(viewsets.ViewSet):
    """Allow manual ingestion and inspection during development."""

    permission_classes = [AllowAny]

    def _resolve_tenant(self, tenant_id: str):
        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
        if tenant is None:
            raise ValueError("tenant_not_found")
        return tenant

    @action(detail=False, methods=["post"], url_path="(?P<tenant_id>[^/]+)/ingest")
    def ingest(self, request: Request, tenant_id: str) -> Response:
        _require_debug()

        payload = request.data or {}
        source = payload.get("source") or "dev-api"
        content_hash = payload.get("content_hash")
        if not content_hash:
            return Response(
                {"detail": "content_hash is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tenant = self._resolve_tenant(tenant_id)
        metadata = payload.get("metadata") or {}
        collections = payload.get("collections") or []
        embedding_profile = payload.get("embedding_profile")
        scope = payload.get("scope")

        # Do not bypass the full ingestion flow: require a dispatcher so the graph
        # can normalize and persist the document properly.
        service = DocumentDomainService()

        with schema_context(tenant.schema_name):
            try:
                result = service.ingest_document(
                    tenant=tenant,
                    source=str(source),
                    content_hash=str(content_hash),
                    metadata=metadata,
                    collections=collections,
                    embedding_profile=embedding_profile,
                    scope=scope,
                    initial_lifecycle_state=DocumentLifecycleState.PENDING,
                )
            except ValueError as exc:
                # Keep dev-only endpoint aligned with production flow: refuse
                # ingestion if the dispatcher (graph) is not available.
                if str(exc) == "ingestion_dispatcher_required":
                    return Response(
                        {
                            "detail": "ingestion dispatcher is required; use the normal upload/crawler flow so the graph can normalize the document"
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                raise

        document = result.document
        return Response(
            {
                "document_id": str(document.id),
                "collections": [str(cid) for cid in result.collection_ids],
                "lifecycle_state": document.lifecycle_state,
            },
            status=status.HTTP_201_CREATED,
        )

    @action(
        detail=False,
        methods=["get"],
        url_path="(?P<tenant_id>[^/]+)/(?P<document_id>[^/]+)",
    )
    def retrieve_document(
        self, request: Request, tenant_id: str, document_id: str
    ) -> Response:
        _require_debug()

        tenant = self._resolve_tenant(tenant_id)
        with schema_context(tenant.schema_name):
            document = get_object_or_404(Document, pk=UUID(str(document_id)))
        return Response(DocumentSerializer(document).data)

    @action(
        detail=False,
        methods=["delete"],
        url_path="(?P<tenant_id>[^/]+)/(?P<document_id>[^/]+)/delete",
    )
    def delete_document(
        self, request: Request, tenant_id: str, document_id: str
    ) -> Response:
        """Soft or hard delete a document.

        Query params:
            hard: If 'true', permanently delete. Otherwise soft delete (retire).
        """
        _require_debug()

        tenant = self._resolve_tenant(tenant_id)
        hard_delete = request.query_params.get("hard", "").lower() == "true"

        with schema_context(tenant.schema_name):
            document = get_object_or_404(Document, pk=UUID(str(document_id)))

            if hard_delete:
                # Hard delete - remove from database
                doc_id = str(document.id)

                # Also clean up from vector store to prevent orphaned search results
                try:
                    from ai_core.rag.vector_client import get_default_client

                    vector_client = get_default_client()
                    vector_client.hard_delete_documents(
                        tenant_id=str(tenant.id),
                        document_ids=[document.id],
                    )
                except Exception as exc:
                    # Log but proceed with DB delete since this is a dev tool
                    logger.warning(
                        "dev_api.vector_cleanup_failed",
                        extra={"document_id": doc_id, "error": str(exc)},
                    )

                document.delete()
                return Response(
                    {
                        "status": "deleted",
                        "document_id": doc_id,
                        "mode": "hard",
                    }
                )
            else:
                # Soft delete - mark as retired
                from django.utils import timezone

                document.lifecycle_state = "retired"
                document.lifecycle_updated_at = timezone.now()
                document.save(update_fields=["lifecycle_state", "lifecycle_updated_at"])

                # Also update vector store to prevent search surfacing retired docs
                try:
                    from ai_core.rag.vector_client import get_default_client

                    vector_client = get_default_client()
                    vector_client.update_lifecycle_state(
                        tenant_id=str(tenant.id),
                        document_ids=[document.id],
                        state="retired",
                        reason="soft_delete_dev_api",
                    )
                except Exception as exc:
                    logger.warning(
                        "dev_api.vector_lifecycle_update_failed",
                        extra={
                            "document_id": str(document.id),
                            "error": str(exc),
                        },
                    )

                return Response(
                    {
                        "status": "retired",
                        "document_id": str(document.id),
                        "mode": "soft",
                        "lifecycle_state": document.lifecycle_state,
                    }
                )


class CollectionDevViewSet(viewsets.ViewSet):
    """Development helper to ensure collections exist."""

    permission_classes = [AllowAny]
    _service = CollectionService()

    @action(detail=False, methods=["post"], url_path="(?P<tenant_id>[^/]+)/ensure")
    def ensure(self, request: Request, tenant_id: str) -> Response:
        _require_debug()

        tenant_obj = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
        if tenant_obj is None:
            return Response(
                {"detail": "tenant not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        payload: Mapping[str, object] = request.data or {}
        key = str(payload.get("key") or "")
        if not key:
            return Response(
                {"detail": "key is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        name = payload.get("name") or key
        metadata = payload.get("metadata") or {}

        collection_id = self._service.ensure_manual_collection(
            tenant=tenant_obj,
            slug=key,
            label=str(name),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

        return Response(
            {
                "collection_id": collection_id,
                "key": key,
                "name": name,
            },
            status=status.HTTP_200_OK,
        )
