"""Domain-level helpers for document and collection lifecycle management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from django.db import transaction
from django.db.models import Count
from django.utils import timezone
from psycopg2 import sql

from ai_core.infra.observability import record_span
from customers.models import Tenant

from .models import Document, DocumentCollection, DocumentCollectionMembership

logger = logging.getLogger(__name__)

IngestionDispatcher = Callable[[UUID, Sequence[UUID], str | None, str | None], None]
DeletionDispatcher = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class PersistedDocumentIngest:
    """Result bundle for document ingestion persistence."""

    document: Document
    collection_ids: tuple[UUID, ...]


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
        document_id: UUID | None = None,
    ) -> PersistedDocumentIngest:
        """Persist or update a document and queue downstream processing.

        The operation is idempotent for a given ``(source, content_hash)`` tuple
        within a tenant. It will create missing collections, manage memberships
        and schedule an ingestion job after the database transaction committed
        successfully. A dispatcher is required so that vector ingestion runs
        only after persistence completed.
        """

        dispatcher_fn = dispatcher or self._ingestion_dispatcher
        if dispatcher_fn is None:
            raise ValueError("ingestion_dispatcher_required")

        metadata_payload = dict(metadata or {})
        if document_id is not None:
            metadata_payload.setdefault("document_id", str(document_id))

        with transaction.atomic():
            document, created = Document.objects.update_or_create(
                tenant=tenant,
                source=source,
                hash=content_hash,
                defaults={
                    "metadata": metadata_payload,
                    **({"id": document_id} if document_id is not None else {}),
                },
            )
            if not created and metadata:
                document.metadata = dict(metadata_payload)
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

            transaction.on_commit(
                lambda: dispatcher_fn(
                    document.id,
                    tuple(collection_ids),
                    embedding_profile,
                    scope,
                )
            )

        return PersistedDocumentIngest(
            document=document, collection_ids=tuple(collection_ids)
        )

    def ensure_collection(
        self,
        *,
        tenant: Tenant,
        key: str,
        name: str | None = None,
        embedding_profile: str | None = None,
        scope: str | None = None,
        metadata: Mapping[str, object] | None = None,
        collection_id: UUID | None = None,
    ) -> DocumentCollection:
        """Return an existing collection or create a new one within a tenant."""

        collection_uuid = collection_id or uuid4()
        defaults = {
            "name": name or key,
            "collection_id": collection_uuid,
            "embedding_profile": embedding_profile or "",
            "metadata": dict(metadata or {}),
        }

        attributes = {
            "tenant_id": str(tenant.id),
            "collection_key": key,
            "collection_id": str(collection_uuid),
            "scope": scope,
        }
        record_span("documents.ensure_collection", attributes=attributes)

        with transaction.atomic():
            collection, created = DocumentCollection.objects.select_for_update().get_or_create(
                tenant=tenant, key=key, defaults=defaults
            )

            if collection_id is not None and collection.collection_id != collection_id:
                raise ValueError(
                    "collection_id_mismatch",
                    collection.collection_id,
                    collection_id,
                )

            if created:
                logger.info(
                    "documents.collection.created",
                    extra={
                        "tenant_id": str(tenant.id),
                        "collection_id": str(collection.collection_id),
                        "key": key,
                    },
                )

            self._ensure_vector_collection(
                tenant=tenant,
                collection=collection,
                embedding_profile=embedding_profile,
                scope=scope,
            )

        logger.info(
            "documents.collection.vector_sync_success",
            extra={
                "tenant_id": str(tenant.id),
                "collection_id": str(collection.collection_id),
                "key": key,
            },
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

        attributes = {
            "tenant_id": str(document.tenant_id),
            "document_id": str(document.id),
            "soft_delete": soft_delete,
            "reason": reason,
        }
        record_span("documents.delete_document", attributes=attributes)

        dispatcher_fn = dispatcher or self._deletion_dispatcher
        vector_store = self._require_vector_store()

        with transaction.atomic():
            vector_store.hard_delete_documents(
                tenant_id=str(document.tenant_id),
                document_ids=[document.id],
            )

            payload = {
                "type": "document_delete",
                "document_id": str(document.id),
                "tenant_id": str(document.tenant_id),
                "reason": reason,
            }
            if dispatcher_fn:
                transaction.on_commit(lambda: dispatcher_fn(payload))
            else:
                logger.info("document_delete_outbox", extra=payload)

            if soft_delete:
                if hasattr(document, "soft_deleted_at"):
                    setattr(document, "soft_deleted_at", timezone.now())
                    document.save(update_fields=["soft_deleted_at", "updated_at"])
                else:
                    logger.warning(
                        "soft_delete_flag_ignored_missing_field",
                        extra={"model": "Document", "id": str(document.id)},
                    )
                return

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

        soft_delete_requested = soft_delete
        if soft_delete_requested:
            if hasattr(collection, "soft_deleted_at"):
                setattr(collection, "soft_deleted_at", timezone.now())
                collection.save(update_fields=["soft_deleted_at", "updated_at"])
            else:
                logger.warning(
                    "soft_delete_flag_ignored_missing_field",
                    extra={"model": "DocumentCollection", "id": str(collection.id)},
                )

        attributes = {
            "tenant_id": str(collection.tenant_id),
            "collection_id": str(collection.collection_id),
            "soft_delete": soft_delete_requested,
            "reason": reason,
        }
        record_span("documents.delete_collection", attributes=attributes)

        dispatcher_fn = dispatcher or self._deletion_dispatcher
        vector_store = self._require_vector_store()

        with transaction.atomic():
            related_document_ids = list(
                collection.documents.values_list("id", flat=True)
            )
            if related_document_ids:
                exclusive_document_ids = list(
                    DocumentCollectionMembership.objects.filter(
                        document_id__in=related_document_ids
                    )
                    .values("document_id")
                    .annotate(collection_count=Count("collection", distinct=True))
                    .filter(collection_count=1)
                    .values_list("document_id", flat=True)
                )

                if exclusive_document_ids:
                    vector_store.hard_delete_documents(
                        tenant_id=str(collection.tenant_id),
                        document_ids=exclusive_document_ids,
                    )

            self._delete_vector_collection_record(collection)

            payload = {
                "type": "collection_delete",
                "collection_id": str(collection.collection_id),
                "tenant_id": str(collection.tenant_id),
                "reason": reason,
            }
            if dispatcher_fn:
                transaction.on_commit(lambda: dispatcher_fn(payload))
            else:
                logger.info("collection_delete_outbox", extra=payload)

            if soft_delete_requested:
                return

            collection.delete()

    def _ensure_vector_collection(
        self,
        *,
        tenant: Tenant,
        collection: DocumentCollection,
        embedding_profile: str | None,
        scope: str | None,
    ) -> None:
        vector_store = self._require_vector_store()

        ensure_fn = getattr(vector_store, "ensure_collection", None)
        if callable(ensure_fn):
            ensure_fn(
                tenant_id=str(tenant.id),
                collection_id=str(collection.collection_id),
                embedding_profile=embedding_profile,
                scope=scope,
            )
            return

        connection = getattr(vector_store, "connection", None)
        table_resolver = getattr(vector_store, "_table", None)
        if callable(connection) and callable(table_resolver):
            collections_table = table_resolver("collections")
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {} (tenant_id, id)
                            VALUES (%s, %s)
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """
                        ).format(collections_table),
                        (str(tenant.id), str(collection.collection_id)),
                    )
                conn.commit()
            return

        raise RuntimeError("Vector store client does not support collection sync")

    def _delete_vector_collection_record(self, collection: DocumentCollection) -> None:
        vector_store = self._require_vector_store()

        delete_fn = getattr(vector_store, "delete_collection", None)
        if callable(delete_fn):
            delete_fn(
                tenant_id=str(collection.tenant_id),
                collection_id=str(collection.collection_id),
            )
            return

        connection = getattr(vector_store, "connection", None)
        table_resolver = getattr(vector_store, "_table", None)
        if callable(connection) and callable(table_resolver):
            collections_table = table_resolver("collections")
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE tenant_id = %s AND id = %s"
                        ).format(collections_table),
                        (str(collection.tenant_id), str(collection.collection_id)),
                    )
                conn.commit()
            return

        raise RuntimeError("Vector store client does not support collection deletion")

    def _require_vector_store(self):
        if self._vector_store is None:
            raise RuntimeError("Vector store client is required for this operation")
        return self._vector_store
