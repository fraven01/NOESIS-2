"""Development-only endpoints for manual ingestion and collection management."""

from __future__ import annotations

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

        service = DocumentDomainService(
            allow_missing_ingestion_dispatcher_for_tests=True
        )

        with schema_context(tenant.schema_name):
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
        return Response(
            {
                "document_id": str(document.id),
                "tenant_id": str(document.tenant_id),
                "source": document.source,
                "hash": document.hash,
                "metadata": document.metadata,
                "lifecycle_state": document.lifecycle_state,
                "lifecycle_updated_at": document.lifecycle_updated_at,
            }
        )


class CollectionDevViewSet(viewsets.ViewSet):
    """Development helper to ensure collections exist."""

    permission_classes = [AllowAny]
    _service = CollectionService()

    @action(detail=False, methods=["post"], url_path="(?P<tenant_id>[^/]+)/ensure")
    def ensure(self, request: Request, tenant_id: str) -> Response:
        _require_debug()

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
            tenant=tenant_id,
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
