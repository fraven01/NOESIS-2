"""Higher level helpers for managing document collections."""

from __future__ import annotations

from typing import Mapping
from uuid import UUID, NAMESPACE_URL, uuid5

from common.logging import get_logger
from customers.tenant_context import TenantContext
from customers.models import Tenant

from .domain_service import DocumentDomainService

DEFAULT_MANUAL_COLLECTION_SLUG = "manual-search"
DEFAULT_MANUAL_COLLECTION_LABEL = "Manual Search"

logger = get_logger(__name__)


class CollectionType:
    """Simple collection type markers."""

    SYSTEM = "system"
    USER = "user"


class CollectionService:
    """Encapsulates collection specific logic on top of the domain service."""

    def __init__(
        self,
        *,
        domain_service: DocumentDomainService | None = None,
        vector_client: object | None = None,
    ):
        if domain_service is not None:
            self._domain = domain_service
        else:
            from ai_core.rag.vector_client import get_default_client

            self._domain = DocumentDomainService(
                vector_store=vector_client or get_default_client()
            )

    def ensure_collection(
        self,
        *,
        tenant,
        key: str,
        name: str | None = None,
        embedding_profile: str | None = None,
        scope: str | None = None,
        metadata: Mapping[str, object] | None = None,
        collection_id: UUID | None = None,
        collection_type: str = CollectionType.USER,
    ):
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("collection_type", collection_type)

        return self._domain.ensure_collection(
            tenant=tenant,
            key=key,
            name=name or key,
            embedding_profile=embedding_profile,
            scope=scope,
            metadata=metadata_payload,
            collection_id=collection_id,
        )

    def ensure_manual_collection(
        self,
        tenant: Tenant | object,
        *,
        slug: str = DEFAULT_MANUAL_COLLECTION_SLUG,
        label: str = DEFAULT_MANUAL_COLLECTION_LABEL,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        tenant_obj = self._resolve_tenant(tenant)
        if tenant_obj is None:
            raise ValueError("tenant_not_found")

        collection_uuid = self.manual_collection_uuid(tenant_obj, slug=slug)
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("slug", slug)
        metadata_payload.setdefault("label", label)
        metadata_payload.setdefault("collection_type", CollectionType.SYSTEM)

        collection = self.ensure_collection(
            tenant=tenant_obj,
            key=slug,
            name=label,
            embedding_profile=None,
            scope="manual",
            metadata=metadata_payload,
            collection_id=collection_uuid,
            collection_type=CollectionType.SYSTEM,
        )
        return str(collection.collection_id)

    @staticmethod
    def manual_collection_uuid(
        tenant: Tenant | object,
        *,
        slug: str = DEFAULT_MANUAL_COLLECTION_SLUG,
    ) -> UUID:
        tenant_seed = CollectionService._canonical_tenant_seed(tenant)
        tenant_uuid = CollectionService._normalise_tenant_uuid(tenant_seed)
        slug_text = str(slug or "").strip().lower()
        if not slug_text:
            raise ValueError("manual_collection_slug_required")
        return uuid5(NAMESPACE_URL, f"collection:{tenant_uuid}:{slug_text}")

    @staticmethod
    def _canonical_tenant_seed(candidate: object) -> str:
        tenant_obj = CollectionService._resolve_tenant(candidate)
        if tenant_obj is not None and getattr(tenant_obj, "schema_name", None):
            return tenant_obj.schema_name
        text = str(candidate or "").strip()
        if not text or text.lower() == "none":
            raise ValueError("tenant_id_required")
        return text.lower()

    @staticmethod
    def _normalise_tenant_uuid(tenant_id: object) -> UUID:
        if isinstance(tenant_id, UUID):
            return tenant_id

        text = str(tenant_id or "").strip()
        if not text or text.lower() == "none":
            raise ValueError("tenant_id_required")

        try:
            return UUID(text)
        except (ValueError, TypeError):
            pass

        return uuid5(NAMESPACE_URL, f"tenant:{text.lower()}")

    @staticmethod
    def _resolve_tenant(candidate: object) -> Tenant | None:
        if isinstance(candidate, Tenant):
            return candidate
        try:
            return TenantContext.resolve_identifier(candidate, allow_pk=True)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception(
                "collection_service.resolve_tenant_failed",
                extra={"tenant_identifier": str(candidate)},
            )
            return None


__all__ = [
    "CollectionService",
    "CollectionType",
    "DEFAULT_MANUAL_COLLECTION_LABEL",
    "DEFAULT_MANUAL_COLLECTION_SLUG",
]
