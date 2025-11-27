"""Helpers for managing RAG collection identifiers."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Optional
from uuid import UUID, NAMESPACE_URL, uuid5

from common.logging import get_logger
from customers.models import Tenant
from documents.domain_service import DocumentDomainService
from .vector_client import PgVectorClient, get_default_client

if TYPE_CHECKING:
    from customers.models import Tenant
    from documents.models import DocumentCollection

MANUAL_COLLECTION_SLUG = "manual-search"
MANUAL_COLLECTION_LABEL = "Manual Search"

_LOGGER = get_logger(__name__)


def _normalise_tenant_uuid(tenant_id: object) -> UUID:
    """Return a UUID for ``tenant_id`` compatible with vector client mapping."""

    if isinstance(tenant_id, UUID):
        return tenant_id

    text = str(tenant_id or "").strip()
    if not text or text.lower() == "none":
        raise ValueError("tenant_id_required")
    with suppress(ValueError, TypeError):
        return UUID(text)

    normalised = text.lower()
    derived = uuid5(NAMESPACE_URL, f"tenant:{normalised}")
    _LOGGER.info(
        "manual_collection.tenant_id_coerced",
        extra={
            "tenant_id": text,
            "normalised_tenant_id": normalised,
            "derived_tenant_uuid": str(derived),
        },
    )
    return derived


def manual_collection_uuid(
    tenant_id: object,
    *,
    slug: str = MANUAL_COLLECTION_SLUG,
) -> UUID:
    """Return the deterministic UUID for the tenant's manual collection."""

    tenant_uuid = _normalise_tenant_uuid(tenant_id)
    slug_text = str(slug or "").strip().lower()
    if not slug_text:
        raise ValueError("manual_collection_slug_required")
    return uuid5(NAMESPACE_URL, f"collection:{tenant_uuid}:{slug_text}")


def ensure_manual_collection(
    tenant_id: object,
    *,
    slug: str = MANUAL_COLLECTION_SLUG,
    label: str = MANUAL_COLLECTION_LABEL,
    client: Optional[PgVectorClient] = None,
) -> str:
    """Ensure the manual collection exists for ``tenant_id`` and return its UUID."""

    collection_uuid = manual_collection_uuid(tenant_id, slug=slug)
    tenant = _resolve_tenant(tenant_id)
    if tenant is None:
        raise ValueError("manual_collection_requires_tenant")

    vector_client = client or get_default_client()
    domain_service = DocumentDomainService(vector_store=vector_client)

    collection = domain_service.ensure_collection(
        tenant=tenant,
        key=slug,
        name=label,
        metadata={"slug": slug, "label": label},
        collection_id=collection_uuid,
        scope="manual",
    )

    return str(collection.collection_id)


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
