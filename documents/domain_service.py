"""Domain-level helpers for document and collection lifecycle management."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from django.db import transaction
from django.utils import timezone

from customers.models import Tenant

from .models import Document, DocumentCollection, DocumentCollectionMembership

logger = logging.getLogger(__name__)

IngestionDispatcher = Callable[[UUID, Sequence[UUID], str | None, str | None], None]
DeletionDispatcher = Callable[[Mapping[str, object]], None]


class DocumentDomainService:
    """Encapsulates document lifecycle logic and queue dispatching."""

    def __init__(
        self,
        *,
        ingestion_dispatcher: IngestionDispatcher | None = None,
        deletion_dispatcher: DeletionDispatcher | None = None,
        vector_store: object | None = None,
    ) -> None:
        self._ingestion_dispatcher = ingestion_dispatcher
        self._deletion_dispatcher = deletion_dispatcher
        self._vector_store = vector_store

    def ingest_document(
        self,
        *,
        tenant: Tenant,
        source: str,
        content_hash: str,
        metadata: Mapping[str, object] | None = None,
        collections: Iterable[str | UUID | DocumentCollection] = (),
        embedding_profile: str | None = None,
        scope: str | None = None,
        dispatcher: IngestionDispatcher | None = None,
    ) -> Document:
        """Persist or update a document and queue downstream processing.

        The operation is idempotent for a given ``(source, content_hash)`` tuple
        within a tenant. It will create missing collections, manage
        memberships and schedule an ingestion job after the database
        transaction committed successfully.
        """

        document, created = Document.objects.update_or_create(
            tenant=tenant,
            source=source,
            hash=content_hash,
            defaults={"metadata": dict(metadata or {})},
        )
        if not created and metadata:
            document.metadata = dict(metadata)
            document.save(update_fields=["metadata", "updated_at"])

        collection_instances: list[DocumentCollection] = []
        for collection in collections:
            if isinstance(collection, DocumentCollection):
                instance = collection
            else:
                instance = self.ensure_collection(
                    tenant=tenant,
                    key=str(collection),
                    embedding_profile=embedding_profile,
                    scope=scope,
                )
            collection_instances.append(instance)

        collection_ids: list[UUID] = []
        for collection in collection_instances:
            _, _ = DocumentCollectionMembership.objects.get_or_create(
                document=document,
                collection=collection,
            )
            collection_ids.append(collection.id)

        dispatcher_fn = dispatcher or self._ingestion_dispatcher
        if dispatcher_fn:
            transaction.on_commit(
                lambda: dispatcher_fn(
                    document.id,
                    tuple(collection_ids),
                    embedding_profile,
                    scope,
                )
            )
        else:
            logger.info(
                "document_ingest_dispatch_skipped",
                extra={
                    "document_id": str(document.id),
                    "collection_ids": [str(cid) for cid in collection_ids],
                    "scope": scope,
                },
            )

        return document

    def ensure_collection(
        self,
        *,
        tenant: Tenant,
        key: str,
        name: str | None = None,
        embedding_profile: str | None = None,
        scope: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> DocumentCollection:
        """Return an existing collection or create a new one within a tenant."""

        defaults = {
            "name": name or key,
            "collection_id": uuid4(),
            "embedding_profile": embedding_profile or "",
            "metadata": dict(metadata or {}),
        }
        collection, created = DocumentCollection.objects.get_or_create(
            tenant=tenant, key=key, defaults=defaults
        )

        if created and self._vector_store is not None:
            ensure_fn = getattr(self._vector_store, "ensure_collection", None)
            if callable(ensure_fn):
                try:
                    ensure_fn(
                        tenant_id=str(tenant.id),
                        collection_id=str(collection.collection_id),
                        embedding_profile=embedding_profile,
                        scope=scope,
                    )
                except Exception:
                    logger.warning(
                        "vector_store_ensure_collection_failed",
                        extra={
                            "tenant_id": str(tenant.id),
                            "collection_id": str(collection.collection_id),
                            "scope": scope,
                        },
                        exc_info=True,
                    )
        return collection

    def delete_document(
        self,
        document: Document,
        *,
        soft_delete: bool = False,
        reason: str | None = None,
        dispatcher: DeletionDispatcher | None = None,
    ) -> None:
        """Delete a document and emit a vector deletion request."""

        if soft_delete:
            if hasattr(document, "soft_deleted_at"):
                setattr(document, "soft_deleted_at", timezone.now())
                document.save(update_fields=["soft_deleted_at", "updated_at"])
            else:
                logger.warning(
                    "soft_delete_flag_ignored_missing_field",
                    extra={"model": "Document", "id": str(document.id)},
                )

        payload = {
            "type": "document_delete",
            "document_id": str(document.id),
            "tenant_id": str(document.tenant_id),
            "reason": reason,
        }
        dispatcher_fn = dispatcher or self._deletion_dispatcher
        if dispatcher_fn:
            transaction.on_commit(lambda: dispatcher_fn(payload))
        else:
            logger.info("document_delete_outbox", extra=payload)

        if not soft_delete:
            document.delete()

    def delete_collection(
        self,
        collection: DocumentCollection,
        *,
        soft_delete: bool = False,
        reason: str | None = None,
        dispatcher: DeletionDispatcher | None = None,
    ) -> None:
        """Delete a collection and emit vector cleanup instructions."""

        if soft_delete:
            if hasattr(collection, "soft_deleted_at"):
                setattr(collection, "soft_deleted_at", timezone.now())
                collection.save(update_fields=["soft_deleted_at", "updated_at"])
            else:
                logger.warning(
                    "soft_delete_flag_ignored_missing_field",
                    extra={"model": "DocumentCollection", "id": str(collection.id)},
                )

        payload = {
            "type": "collection_delete",
            "collection_id": str(collection.collection_id),
            "tenant_id": str(collection.tenant_id),
            "reason": reason,
        }
        dispatcher_fn = dispatcher or self._deletion_dispatcher
        if dispatcher_fn:
            transaction.on_commit(lambda: dispatcher_fn(payload))
        else:
            logger.info("collection_delete_outbox", extra=payload)

        if not soft_delete:
            collection.delete()
