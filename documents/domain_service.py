"""Domain-level helpers for document and collection lifecycle management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from django.db import transaction
from django.db.models import Count
from django.utils import timezone

from ai_core.infra.observability import record_span
from customers.models import Tenant

from .models import Document, DocumentCollection, DocumentCollectionMembership
from .lifecycle import DocumentLifecycleState, VALID_TRANSITIONS

logger = logging.getLogger(__name__)

IngestionDispatcher = Callable[[UUID, Sequence[UUID], str | None, str | None], None]
DeletionDispatcher = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class PersistedDocumentIngest:
    """Result bundle for document ingestion persistence."""

    document: Document
    collection_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class DocumentIngestSpec:
    """Specification payload for a single bulk ingestion entry."""

    source: str
    content_hash: str
    metadata: Mapping[str, object]
    collections: Sequence[str | UUID | DocumentCollection] = ()
    embedding_profile: str | None = None
    scope: str | None = None
    document_id: UUID | None = None
    initial_lifecycle_state: DocumentLifecycleState | str = (
        DocumentLifecycleState.PENDING
    )


@dataclass(frozen=True)
class BulkIngestRecord:
    """Successful bulk ingestion record."""

    spec: DocumentIngestSpec
    result: PersistedDocumentIngest


@dataclass(frozen=True)
class BulkIngestResult:
    """Aggregate result for bulk ingestion operations."""

    records: list[BulkIngestRecord]
    failed: list[tuple[DocumentIngestSpec, Exception]]

    @property
    def ingested(self) -> list[PersistedDocumentIngest]:
        return [entry.result for entry in self.records]

    @property
    def succeeded(self) -> int:
        return len(self.records)

    @property
    def total(self) -> int:
        return self.succeeded + len(self.failed)


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
        initial_lifecycle_state: (
            DocumentLifecycleState | str
        ) = DocumentLifecycleState.PENDING,
        allow_missing_ingestion_dispatcher_for_tests: bool | None = None,
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
            logger.error(
                "ingestion_dispatcher_missing",
                extra={
                    "tenant_id": str(tenant.id),
                    "source": source,
                    "scope": scope,
                    "embedding_profile": embedding_profile,
                },
            )
            raise ValueError("ingestion_dispatcher_required")

        metadata_payload = dict(metadata or {})
        if document_id is not None:
            metadata_payload.setdefault("document_id", str(document_id))

        lifecycle_state = DocumentLifecycleState(initial_lifecycle_state)
        lifecycle_timestamp = timezone.now()

        with transaction.atomic():
            document, created = Document.objects.update_or_create(
                tenant=tenant,
                source=source,
                hash=content_hash,
                defaults={
                    "metadata": metadata_payload,
                    "lifecycle_state": lifecycle_state.value,
                    "lifecycle_updated_at": lifecycle_timestamp,
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

            if dispatcher_fn:
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

    def bulk_ingest_documents(
        self,
        *,
        tenant: Tenant,
        documents: Sequence[DocumentIngestSpec],
        dispatcher: IngestionDispatcher | None = None,
        allow_missing_ingestion_dispatcher_for_tests: bool | None = None,
    ) -> BulkIngestResult:
        """Ingest multiple documents in sequence and aggregate the outcome."""

        if not documents:
            return BulkIngestResult(records=[], failed=[])

        dispatcher_fn = dispatcher or self._ingestion_dispatcher
        allow_missing_dispatcher = allow_missing_ingestion_dispatcher_for_tests or False
        if dispatcher_fn is None and not allow_missing_dispatcher:
            raise ValueError("ingestion_dispatcher_required")

        collection_cache: dict[str, DocumentCollection] = {}
        successes: list[BulkIngestRecord] = []
        failures: list[tuple[DocumentIngestSpec, Exception]] = []

        for spec in documents:
            try:
                collection_instances: list[DocumentCollection] = []
                for collection in spec.collections:
                    if isinstance(collection, DocumentCollection):
                        instance = collection
                    else:
                        cache_key = str(collection)
                        instance = collection_cache.get(cache_key)
                        if instance is None:
                            instance = self.ensure_collection(
                                tenant=tenant,
                                key=cache_key,
                                embedding_profile=spec.embedding_profile,
                                scope=spec.scope,
                            )
                            collection_cache[cache_key] = instance
                    collection_instances.append(instance)

                result = self.ingest_document(
                    tenant=tenant,
                    source=spec.source,
                    content_hash=spec.content_hash,
                    metadata=spec.metadata,
                    collections=collection_instances,
                    embedding_profile=spec.embedding_profile,
                    scope=spec.scope,
                    dispatcher=dispatcher_fn,
                    document_id=spec.document_id,
                    initial_lifecycle_state=spec.initial_lifecycle_state,
                    allow_missing_ingestion_dispatcher_for_tests=allow_missing_dispatcher,
                )
                successes.append(BulkIngestRecord(spec=spec, result=result))
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append((spec, exc))
                logger.exception(
                    "documents.bulk_ingest_failed",
                    extra={
                        "tenant_id": str(tenant.id),
                        "source": spec.source,
                        "hash": spec.content_hash,
                    },
                )

        return BulkIngestResult(records=successes, failed=failures)

    def update_lifecycle_state(
        self,
        *,
        document: Document,
        new_state: DocumentLifecycleState | str,
        reason: str | None = None,
        validate_transition: bool = True,
    ) -> Document:
        """Update document lifecycle state and emit observability breadcrumbs."""

        desired_state = DocumentLifecycleState(new_state)
        current_state = DocumentLifecycleState(
            document.lifecycle_state or DocumentLifecycleState.PENDING.value
        )

        if validate_transition:
            allowed = VALID_TRANSITIONS.get(current_state, set())
            if desired_state not in allowed:
                raise ValueError(
                    f"invalid_lifecycle_transition: {current_state.value}->{desired_state.value}"
                )

        document.lifecycle_state = desired_state.value
        document.lifecycle_updated_at = timezone.now()
        document.save(
            update_fields=["lifecycle_state", "lifecycle_updated_at", "updated_at"]
        )

        logger.info(
            "document_lifecycle_updated",
            extra={
                "document_id": str(document.id),
                "tenant_id": str(document.tenant_id),
                "new_state": desired_state.value,
                "previous_state": current_state.value,
                "reason": reason,
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
            (
                collection,
                created,
            ) = DocumentCollection.objects.select_for_update().get_or_create(
                tenant=tenant, key=key, defaults=defaults
            )

            if collection_id is not None and collection.collection_id != collection_id:
                logger.warning(
                    "documents.collection.id_mismatch_ignored",
                    extra={
                        "expected": str(collection_id),
                        "actual": str(collection.collection_id),
                        "key": key,
                        "tenant_id": str(tenant.id),
                    },
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

            transaction.on_commit(
                lambda: self._ensure_vector_collection(
                    tenant=tenant,
                    collection=collection,
                    embedding_profile=embedding_profile,
                    scope=scope,
                )
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

        doc_id = document.id
        attributes = {
            "tenant_id": str(document.tenant_id),
            "document_id": str(doc_id),
            "soft_delete": soft_delete,
            "reason": reason,
        }
        record_span("documents.delete_document", attributes=attributes)

        dispatcher_fn = dispatcher or self._deletion_dispatcher
        if dispatcher_fn is None:
            logger.error(
                "deletion_dispatcher_missing",
                extra={
                    "tenant_id": str(document.tenant_id),
                    "document_id": str(doc_id),
                },
            )
            raise ValueError("deletion_dispatcher_required")

        payload = {
            "type": "document_delete",
            "document_id": str(doc_id),
            "document_ids": (str(doc_id),),
            "tenant_id": str(document.tenant_id),
            "reason": reason,
        }

        def _dispatch_cleanup() -> None:
            attempts = 0
            while True:
                attempts += 1
                try:
                    dispatcher_fn(payload)
                    logger.info(
                        "documents.delete_document.dispatched",
                        extra={**payload, "attempt": attempts},
                    )
                    return
                except Exception:
                    logger.exception(
                        "document_delete_dispatch_failed",
                        extra={**payload, "attempt": attempts},
                    )
                    if attempts >= 3:
                        raise

        with transaction.atomic():
            DocumentCollectionMembership.objects.filter(document=document).delete()

            if soft_delete:
                document.soft_deleted_at = timezone.now()
                document.save(update_fields=["soft_deleted_at", "updated_at"])
            else:
                document.delete()

            transaction.on_commit(_dispatch_cleanup)

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
        attributes = {
            "tenant_id": str(collection.tenant_id),
            "collection_id": str(collection.collection_id),
            "soft_delete": soft_delete_requested,
            "reason": reason,
        }
        record_span("documents.delete_collection", attributes=attributes)

        dispatcher_fn = dispatcher or self._deletion_dispatcher
        if dispatcher_fn is None:
            logger.error(
                "deletion_dispatcher_missing",
                extra={
                    "tenant_id": str(collection.tenant_id),
                    "collection_id": str(collection.collection_id),
                },
            )
            raise ValueError("deletion_dispatcher_required")

        vector_store = self._require_vector_store()

        with transaction.atomic():
            related_document_ids = list(
                collection.documents.values_list("id", flat=True)
            )
            exclusive_document_ids: list[UUID] = []
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

            payload = {
                "type": "collection_delete",
                "collection_id": str(collection.collection_id),
                "tenant_id": str(collection.tenant_id),
                "reason": reason,
            }

            transaction.on_commit(
                lambda exclusive_document_ids=exclusive_document_ids, payload=payload: self._dispatch_collection_cleanup(
                    exclusive_document_ids=exclusive_document_ids,
                    payload=payload,
                    dispatcher_fn=dispatcher_fn,
                    vector_store=vector_store,
                )
            )

            if soft_delete_requested:
                collection.soft_deleted_at = timezone.now()
                collection.save(update_fields=["soft_deleted_at", "updated_at"])
            else:
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
        if not callable(ensure_fn):
            raise RuntimeError("Vector store client must support collection ensure")

        payload = {
            "tenant_id": str(tenant.id),
            "collection_id": str(collection.collection_id),
            "embedding_profile": embedding_profile,
            "scope": scope,
        }
        try:
            ensure_fn(**payload)
        except Exception:
            logger.exception(
                "documents.collection.vector_sync_failed",
                extra=payload,
            )
            raise

        logger.info("documents.collection.vector_sync_success", extra=payload)

    def _delete_vector_collection_record(
        self,
        *,
        tenant_id: str,
        collection_id: UUID | str,
        vector_store: object | None = None,
    ) -> None:
        vector_store = vector_store or self._require_vector_store()

        delete_fn = getattr(vector_store, "delete_collection", None)
        if not callable(delete_fn):
            raise RuntimeError("Vector store client must support collection deletion")

        payload = {
            "tenant_id": str(tenant_id),
            "collection_id": str(collection_id),
        }
        try:
            delete_fn(**payload)
        except Exception:
            logger.exception(
                "documents.collection.vector_delete_failed",
                extra=payload,
            )
            raise

        logger.info("documents.collection.vector_delete_success", extra=payload)

    def _dispatch_collection_cleanup(
        self,
        *,
        exclusive_document_ids: Sequence[UUID] | Sequence[str],
        payload: Mapping[str, object],
        dispatcher_fn: DeletionDispatcher,
        vector_store: object,
    ) -> None:
        extra = dict(payload)
        try:
            if exclusive_document_ids:
                vector_store.hard_delete_documents(
                    tenant_id=str(payload["tenant_id"]),
                    document_ids=list(exclusive_document_ids),
                )

            self._delete_vector_collection_record(
                tenant_id=str(payload["tenant_id"]),
                collection_id=payload["collection_id"],
                vector_store=vector_store,
            )
        except Exception:
            logger.exception(
                "documents.collection.vector_cleanup_failed",
                extra={**extra, "exclusive_document_ids": list(exclusive_document_ids)},
            )
            raise

        try:
            dispatcher_fn(payload)
            logger.info("documents.collection.delete_dispatched", extra=extra)
        except Exception:
            logger.exception("documents.collection.delete_dispatch_failed", extra=extra)
            raise

    def _require_vector_store(self):
        if self._vector_store is None:
            raise RuntimeError("Vector store client is required for this operation")
        return self._vector_store


__all__ = [
    "DocumentDomainService",
    "IngestionDispatcher",
    "DeletionDispatcher",
    "PersistedDocumentIngest",
    "DocumentIngestSpec",
    "BulkIngestRecord",
    "BulkIngestResult",
]
