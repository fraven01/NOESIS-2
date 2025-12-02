"""Helpers for managing RAG collection identifiers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from common.logging import get_logger
from customers.models import Tenant
from documents.collection_service import (
    CollectionService,
    DEFAULT_MANUAL_COLLECTION_LABEL,
    DEFAULT_MANUAL_COLLECTION_SLUG,
)
from .vector_client import PgVectorClient

if TYPE_CHECKING:
    from documents.models import DocumentCollection

MANUAL_COLLECTION_SLUG = DEFAULT_MANUAL_COLLECTION_SLUG
MANUAL_COLLECTION_LABEL = DEFAULT_MANUAL_COLLECTION_LABEL

_LOGGER = get_logger(__name__)
_COLLECTION_SERVICE: CollectionService | None = None


def _get_collection_service(
    *, vector_client: object | None = None
) -> CollectionService:
    if vector_client is not None:
        return CollectionService(vector_client=vector_client)
    global _COLLECTION_SERVICE
    if _COLLECTION_SERVICE is None:
        _COLLECTION_SERVICE = CollectionService()
    return _COLLECTION_SERVICE


def manual_collection_uuid(
    tenant: object,
    *,
    slug: str = MANUAL_COLLECTION_SLUG,
) -> UUID:
    """Return the deterministic UUID for the tenant's manual collection."""

    return CollectionService.manual_collection_uuid(tenant, slug=slug)


def ensure_manual_collection(
    tenant: object,
    *,
    slug: str = MANUAL_COLLECTION_SLUG,
    label: str = MANUAL_COLLECTION_LABEL,
    client: Optional[PgVectorClient] = None,
) -> str:
    """Ensure the manual collection exists for ``tenant_id`` and return its UUID."""

    warnings.warn(
        "ensure_manual_collection is deprecated, use CollectionService instead",
        DeprecationWarning,
        stacklevel=2,
    )
    service = _get_collection_service(vector_client=client)
    return service.ensure_manual_collection(
        tenant=tenant,
        slug=slug,
        label=label,
    )


def _resolve_tenant(tenant_id: object) -> "Tenant | None":
    from customers.tenant_context import TenantContext

    return TenantContext.resolve_identifier(tenant_id, allow_pk=True)


def _get_document_collection_model() -> type["DocumentCollection"]:
    from documents.models import DocumentCollection

    return DocumentCollection


def ensure_manual_collection_model(
    *,
    tenant_id: object,
    collection_uuid: UUID,
    slug: str = MANUAL_COLLECTION_SLUG,
    label: str = MANUAL_COLLECTION_LABEL,
) -> "DocumentCollection | None":
    """Ensure the ORM representation of the manual collection exists."""

    tenant = _resolve_tenant(tenant_id)
    if tenant is None:
        _LOGGER.info(
            "manual_collection.model_missing_tenant",
            extra={"tenant_id": str(tenant_id)},
        )
        return None

    DocumentCollection = _get_document_collection_model()
    try:
        collection, _ = DocumentCollection.objects.get_or_create(
            tenant=tenant,
            key=slug,
            defaults={
                "id": collection_uuid,
                "collection_id": collection_uuid,
                "name": label,
                "type": "",
                "visibility": "",
                "metadata": {},
            },
        )
        return collection
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "manual_collection.model_get_or_create_failed",
            extra={
                "tenant_id": str(tenant),
                "collection_id": str(collection_uuid),
            },
        )
        return None


__all__ = [
    "MANUAL_COLLECTION_LABEL",
    "MANUAL_COLLECTION_SLUG",
    "ensure_manual_collection",
    "ensure_manual_collection_model",
    "manual_collection_uuid",
]
