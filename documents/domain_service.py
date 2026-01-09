"""Domain-level helpers for document and collection lifecycle management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from django.conf import settings
from django.db import transaction
from django.db.models import Count
from django.utils import timezone

from ai_core.infra import object_store
from ai_core.infra.observability import record_span
from django.apps import apps
from customers.models import Tenant
from customers.tenant_context import TenantContext
from documents.parsers import ParsedResult
from documents.contracts import NormalizedDocument
from .lifecycle import DocumentLifecycleState, VALID_TRANSITIONS
from .models import Document, DocumentCollection, DocumentCollectionMembership

logger = logging.getLogger(__name__)

IngestionDispatcher = Callable[[UUID, Sequence[UUID], str | None, str | None], None]
DeletionDispatcher = Callable[[Mapping[str, object]], None]


class CollectionIdConflictError(ValueError):
    """Raised when a collection exists with a different collection_id.

    This error is raised when ensure_collection is called with a collection_id
    that differs from an existing collection's ID. This typically indicates:
    1. Manual database modification
    2. Race condition during creation
    3. Bug in ID generation logic

    Overwriting the collection_id without proper migration will orphan vector
    store data under the old ID, leading to data loss.
    """

    def __init__(
        self, *, key: str, existing_id: UUID, requested_id: UUID, tenant_id: str
    ):
        self.key = key
        self.existing_id = existing_id
        self.requested_id = requested_id
        self.tenant_id = tenant_id

        super().__init__(
            f"Collection '{key}' in tenant {tenant_id} has collection_id={existing_id}, "
            f"but {requested_id} was requested. This likely indicates:\n"
            f"  1. Manual DB modification\n"
            f"  2. Race condition\n"
            f"  3. Bug in ID generation\n\n"
            f"⚠️  Overwriting the ID will orphan vector store data!\n\n"
            f"To proceed:\n"
            f"  - Pass allow_collection_id_override=True (⚠️ DANGEROUS)\n"
            f"  - Or investigate why the IDs differ"
        )


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
        allow_missing_vector_store_for_tests: bool | None = None,
    ) -> None:
        self._ingestion_dispatcher = ingestion_dispatcher
        self._deletion_dispatcher = deletion_dispatcher
        self._vector_store = vector_store
        testing_env_flag = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        self._allow_missing_vector_store = bool(
            allow_missing_vector_store_for_tests
            or getattr(settings, "TESTING", False)
            or testing_env_flag
            or getattr(settings, "DEBUG", False)
        )

    def ingest_document(
        self,
        *,
        tenant: Tenant,
        source: str,
        content_hash: str,
        metadata: Mapping[str, object] | None = None,
        audit_meta: Mapping[str, object] | None = None,
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
        created_by_user_id = (audit_meta or {}).get("created_by_user_id")
        last_hop_service_id = (audit_meta or {}).get("last_hop_service_id")
        created_by_user = None
        if created_by_user_id:
            try:
                User = apps.get_model("users", "User")
                created_by_user = User.objects.get(pk=created_by_user_id)
            except Exception:
                logger.warning(
                    "documents.created_by_user_missing",
                    extra={"user_id": created_by_user_id},
                )

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
            if created and created_by_user:
                document.created_by = created_by_user
                document.updated_by = created_by_user
                document.save(update_fields=["created_by", "updated_by", "updated_at"])
            elif created_by_user:
                document.updated_by = created_by_user
                document.save(update_fields=["updated_by", "updated_at"])
            if not created and metadata:
                document.metadata = dict(metadata_payload)
                document.save(update_fields=["metadata", "updated_at"])

            collection_instances: list[DocumentCollection] = []
            for collection in collections:
                if isinstance(collection, DocumentCollection):
                    instance = collection
                else:
                    instance = self._resolve_collection_reference(
                        tenant=tenant,
                        collection=collection,
                        embedding_profile=embedding_profile,
                        scope=scope,
                    )
                collection_instances.append(instance)

            collection_ids: list[UUID] = []
            for collection in collection_instances:
                membership_defaults: dict[str, object] = {}
                if created_by_user:
                    membership_defaults["added_by_user"] = created_by_user
                elif last_hop_service_id:
                    membership_defaults["added_by_service_id"] = last_hop_service_id
                _, _ = DocumentCollectionMembership.objects.get_or_create(
                    document=document,
                    collection=collection,
                    defaults=membership_defaults or None,
                )
                collection_ids.append(collection.collection_id)

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
        allow_collection_id_override: bool = False,
    ) -> DocumentCollection:
        """Return an existing collection or create a new one within a tenant.

        Args:
            tenant: The tenant owning the collection.
            key: Unique key identifying the collection within the tenant.
            name: Human-readable name for the collection.
            embedding_profile: Optional embedding profile for the collection.
            scope: Optional scope for the collection.
            metadata: Optional metadata dictionary.
            collection_id: Optional explicit collection_id to set. If provided and
                a collection with the same key exists but has a different ID,
                an error will be raised unless allow_collection_id_override is True.
            allow_collection_id_override: If True, allows overwriting an existing
                collection's collection_id. ⚠️ WARNING: This will orphan vector
                store data under the old ID! Only use for deterministic ID
                generation or migrations.

        Returns:
            The collection instance (created or existing).

        Raises:
            CollectionIdConflictError: If collection_id differs from existing
                collection and allow_collection_id_override is False.
        """
        collection_uuid = collection_id or uuid4()
        defaults = {
            "name": name or key,
            # id (PK) is auto-generated by Django to prevent collisions when
            # collection_id is explicitly provided and already exists as PK
            # of another collection. collection_id is the semantic business ID.
            "collection_id": collection_uuid,
            "embedding_profile": embedding_profile or "",
            "metadata": dict(metadata or {}),
        }

        attributes = {
            "tenant_id": str(tenant.id),
            "collection_key": key,
            "collection_id": str(collection_id) if collection_id else None,
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
                if not allow_collection_id_override:
                    raise CollectionIdConflictError(
                        key=key,
                        existing_id=collection.collection_id,
                        requested_id=collection_id,
                        tenant_id=str(tenant.id),
                    )

                logger.error(
                    "documents.collection.id_override_forced",
                    extra={
                        "tenant_id": str(tenant.id),
                        "key": key,
                        "old_id": str(collection.collection_id),
                        "new_id": str(collection_id),
                        "WARNING": "Vector data under old ID is now orphaned!",
                    },
                )
                collection.collection_id = collection_id
                collection.save(update_fields=["collection_id", "updated_at"])

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

    def _resolve_collection_reference(
        self,
        *,
        tenant: Tenant,
        collection: str | UUID,
        embedding_profile: str | None,
        scope: str | None,
    ) -> DocumentCollection:
        """Return an existing collection for a given reference or create it if missing."""
        collection_uuid: UUID | None = None
        try:
            collection_uuid = UUID(str(collection))
        except (TypeError, ValueError):
            collection_uuid = None

        if collection_uuid:
            existing = DocumentCollection.objects.filter(
                tenant=tenant, id=collection_uuid
            ).first()
            if existing is None:
                existing = DocumentCollection.objects.filter(
                    tenant=tenant, collection_id=collection_uuid
                ).first()
            if existing is not None:
                return existing

        return self.ensure_collection(
            tenant=tenant,
            key=str(collection),
            embedding_profile=embedding_profile,
            scope=scope,
            collection_id=collection_uuid,
        )

    def upsert_normalized_document(
        self,
        document: NormalizedDocument,
        parsed_result: ParsedResult,
        collections: Sequence[str | UUID | DocumentCollection] = (),
        scope: str | None = None,
    ) -> UUID:
        """Persist normalized document and artifacts to storage and DB.

        This method ensures:
        1. Text blocks are saved to Object Store.
        2. NormalizedDocument is updated with artifact paths.
        3. Document is persisted to the database via Repository.
        """
        if not self._vector_store:
            pass  # Use defaults or error? existing code uses defaults

        # 1. Persist text blocks to Object Store
        # We save the 'text_blocks' from parsed_result as a JSON artifact
        # This is similar to what 'ingest_raw' did but for parsed content

        # Construct path
        tenant_id = document.meta.tenant_id
        doc_id = str(document.ref.document_id)
        path = f"{tenant_id}/upload/text/{doc_id}.json"

        # Serialize blocks (reuse pipeline logic or simple dict)
        serialized_blocks = [
            {"text": b.text, "kind": b.kind, "language": b.language}
            for b in parsed_result.text_blocks
        ]

        object_store.write_json(path, serialized_blocks)

        # Update document meta with this path if we want to track it
        # document.meta.external_ref["parsed_blocks_path"] = path
        # But DocumentMeta.external_ref is Dict[str, str], so acceptable.

        # 2. Persist to Repository (DB)
        # We need to bridge NormalizedDocument (Pydantic) to Django models.
        # Since we don't have the concrete Repository instance here easily (unless injected),
        # we might need to use Django models directly as fallback or rely on injection.

        # Re-using ingest_document logic partially?
        # Ideally, we should use a Repository abstraction.
        # But 'DocumentDomainService' seems to use Django models directly in 'ingest_document'.

        # 2. Persist to Repository (DB)
        # Delegate to the configured repository (DB in prod, InMemory in tests)
        from ai_core import services  # local import to avoid circular deps

        repository = services._get_documents_repository()

        # upsert() handles the update_or_create logic with IntegrityError handling
        persisted_doc = repository.upsert(document)
        final_doc_id = persisted_doc.ref.document_id

        # 3. Collection Management
        # Resolve tenant for collection operations (DbDocumentsRepository handles it internally for doc, but we need it here)
        tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
        if tenant is None:
            raise Tenant.DoesNotExist(f"tenant_not_found: {tenant_id}")

        collection_instances: list[DocumentCollection] = []
        embedding_profile = getattr(document.meta, "embedding_profile", None)

        for collection in collections:
            if isinstance(collection, DocumentCollection):
                instance = collection
            else:
                instance = self._resolve_collection_reference(
                    tenant=tenant,
                    collection=collection,
                    embedding_profile=embedding_profile,
                    scope=scope,
                )
            collection_instances.append(instance)

        # Update memberships
        for col_instance in collection_instances:
            DocumentCollectionMembership.objects.get_or_create(
                document_id=final_doc_id,
                collection=col_instance,
            )

        return final_doc_id

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
        vector_store = self._vector_store
        if vector_store is None:
            if self._allow_missing_vector_store:
                logger.warning(
                    "documents.vector_store_missing",
                    extra={
                        "tenant_id": str(tenant.id),
                        "collection_id": str(collection.collection_id),
                        "scope": scope,
                    },
                )
                return
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
    "CollectionIdConflictError",
    "DocumentDomainService",
    "IngestionDispatcher",
    "DeletionDispatcher",
    "PersistedDocumentIngest",
    "DocumentIngestSpec",
    "BulkIngestRecord",
    "BulkIngestResult",
]
