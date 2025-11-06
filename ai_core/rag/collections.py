"""Helpers for managing RAG collection identifiers."""

from __future__ import annotations

from contextlib import suppress
from typing import Optional
from uuid import UUID, NAMESPACE_URL, uuid5

from psycopg2 import Error as PsycopgError, sql

from common.logging import get_logger
from .vector_client import PgVectorClient, get_default_client

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
    tenant_uuid = _normalise_tenant_uuid(tenant_id)
    vector_client = client or get_default_client()

    try:
        with vector_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (tenant_id, id, slug, version_label)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (tenant_id, id) DO UPDATE
                        SET slug = EXCLUDED.slug,
                            version_label = EXCLUDED.version_label
                        """
                    ).format(vector_client._table("collections")),
                    (
                        str(tenant_uuid),
                        str(collection_uuid),
                        slug,
                        label,
                    ),
                )
            conn.commit()
    except PsycopgError as exc:  # pragma: no cover - defensive guard
        _LOGGER.warning(
            "manual_collection.ensure_failed",
            extra={
                "tenant_id": str(tenant_uuid),
                "collection_id": str(collection_uuid),
                "error": str(exc),
            },
        )
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "manual_collection.ensure_unexpected_error",
            extra={
                "tenant_id": str(tenant_uuid),
                "collection_id": str(collection_uuid),
            },
        )
        raise

    return str(collection_uuid)


__all__ = [
    "MANUAL_COLLECTION_LABEL",
    "MANUAL_COLLECTION_SLUG",
    "ensure_manual_collection",
    "manual_collection_uuid",
]
