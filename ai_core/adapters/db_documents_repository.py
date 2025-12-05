from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from django.apps import apps
from django.db import IntegrityError, models, transaction

from documents.contracts import DocumentRef, NormalizedDocument
from documents.repository import (
    DocumentsRepository,
    _workflow_storage_key,
)
from documents.storage import ObjectStoreStorage, Storage


class DbDocumentsRepository(DocumentsRepository):
    """Database-backed repository using Django models for metadata."""

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self._storage = storage or ObjectStoreStorage()

    def upsert(
        self, doc: NormalizedDocument, workflow_id: Optional[str] = None
    ) -> NormalizedDocument:
        doc_copy = doc.model_copy(deep=True)
        doc_copy = self._materialize_document(doc_copy)

        workflow = workflow_id or doc_copy.ref.workflow_id
        if workflow != doc_copy.ref.workflow_id:
            raise ValueError("workflow_mismatch")

        workflow_key = _workflow_storage_key(workflow)

        Document = apps.get_model("documents", "Document")
        DocumentCollectionMembership = apps.get_model(
            "documents", "DocumentCollectionMembership"
        )
        DocumentLifecycleState = apps.get_model("documents", "DocumentLifecycleState")
        Tenant = apps.get_model("customers", "Tenant")

        metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
        collection_id = doc_copy.ref.collection_id or doc_copy.meta.document_collection_id

        with transaction.atomic():
            try:
                tenant = Tenant.objects.get(schema_name=doc_copy.ref.tenant_id)
            except Tenant.DoesNotExist as exc:  # pragma: no cover - safety net
                raise ValueError("tenant_missing") from exc

            try:
                document, _ = Document.objects.update_or_create(
                    id=doc_copy.ref.document_id,
                    defaults={
                        "tenant": tenant,
                        "hash": doc_copy.checksum,
                        "source": doc_copy.source or "",
                        "metadata": metadata,
                        "lifecycle_state": doc_copy.lifecycle_state,
                        "lifecycle_updated_at": doc_copy.created_at,
                    },
                )
            except IntegrityError as exc:
                if "document_unique_source_hash" not in str(exc):
                    raise
                document = Document.objects.get(
                    tenant=tenant,
                    source=doc_copy.source or "",
                    hash=doc_copy.checksum,
                )

            if collection_id:
                DocumentCollectionMembership.objects.get_or_create(
                    document=document,
                    collection_id=collection_id,
                    defaults={"added_by": "system"},
                )

            DocumentLifecycleState.objects.update_or_create(
                tenant_id=doc_copy.ref.tenant_id,
                document_id=document.id,
                workflow_id=workflow_key,
                defaults={
                    "state": doc_copy.lifecycle_state,
                    "changed_at": doc_copy.created_at,
                    "reason": "",
                    "policy_events": [],
                },
            )

        return self.get(
            doc_copy.ref.tenant_id,
            document.id,
            doc_copy.ref.version,
            workflow_id=workflow,
            prefer_latest=True,
        ) or doc_copy

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
        workflow_id: Optional[str] = None,
    ) -> Optional[NormalizedDocument]:
        Document = apps.get_model("documents", "Document")
        lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")

        document = Document.objects.filter(
            tenant__schema_name=tenant_id, id=document_id
        ).first()
        if document is None:
            return None

        lifecycle = _select_lifecycle_state(
            lifecycle_model, tenant_id, document_id, workflow_id
        )

        normalized = _build_document_from_metadata(document)
        if normalized is None:
            return None

        if lifecycle is not None:
            normalized.lifecycle_state = lifecycle.state

        return normalized

    def list_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = False,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        if latest_only:
            return self.list_latest_by_collection(
                tenant_id, collection_id, limit, cursor, workflow_id=workflow_id
            )

        memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
        memberships = _apply_cursor_filter(memberships, cursor)

        entries: list[tuple[tuple, NormalizedDocument]] = []
        for membership in memberships[: limit + 1]:
            document = membership.document
            normalized = _build_document_from_metadata(document)
            if normalized is None:
                continue
            entries.append(self._document_entry(normalized))

        entries.sort(key=lambda entry: entry[0])
        refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
        next_cursor = self._next_cursor(entries, 0, limit)
        return refs, next_cursor

    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
        memberships = _apply_cursor_filter(memberships, cursor)

        entries: list[tuple[tuple, NormalizedDocument]] = []
        for membership in memberships[: limit + 1]:
            document = membership.document
            normalized = _build_document_from_metadata(document)
            if normalized is None:
                continue
            doc_ref = normalized.ref
            if workflow_id and doc_ref.workflow_id != workflow_id:
                continue
            entries.append(self._document_entry(normalized))

        entries.sort(key=lambda entry: entry[0])
        refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
        next_cursor = self._next_cursor(entries, 0, limit)
        return refs, next_cursor


def _build_document_from_metadata(document) -> Optional[NormalizedDocument]:
    payload = document.metadata or {}
    normalized_payload = payload.get("normalized_document") or {}
    if not normalized_payload:
        return None
    normalized = NormalizedDocument.model_validate(normalized_payload)
    # Align timestamps with the persisted row to support cursor pagination.
    normalized.created_at = document.created_at
    return normalized


def _collection_queryset(tenant_id: str, collection_id: UUID, workflow_id: Optional[str]):
    DocumentCollectionMembership = apps.get_model(
        "documents", "DocumentCollectionMembership"
    )
    lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")
    workflow_key = _workflow_storage_key(workflow_id)

    queryset = DocumentCollectionMembership.objects.select_related("document").filter(
        collection_id=collection_id, document__tenant__schema_name=tenant_id
    )

    if workflow_id is not None:
        lifecycle_exists = models.Exists(
            lifecycle_model.objects.filter(
                tenant_id=tenant_id,
                workflow_id=workflow_key,
                document_id=models.OuterRef("document__id"),
            )
        )
        queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(
            has_lifecycle=True
        )

    return queryset.order_by("-document__created_at", "document__id")


def _select_lifecycle_state(model, tenant_id: str, document_id: UUID, workflow_id: Optional[str]):
    workflow_key = _workflow_storage_key(workflow_id)
    filters = {"tenant_id": tenant_id, "document_id": document_id}
    if workflow_id is not None:
        filters["workflow_id"] = workflow_key
    qs = model.objects.filter(**filters).order_by("-changed_at")
    return qs.first()


def _apply_cursor_filter(queryset, cursor: Optional[str]):
    if not cursor:
        return queryset

    parts = DocumentsRepository._decode_cursor(cursor)
    if len(parts) < 2:
        raise ValueError("cursor_invalid")

    try:
        timestamp = datetime.fromisoformat(parts[0])
        document_id = UUID(parts[1])
    except (ValueError, TypeError) as exc:
        raise ValueError("cursor_invalid") from exc

    return queryset.filter(
        models.Q(document__created_at__lt=timestamp)
        | (
            models.Q(document__created_at=timestamp)
            & models.Q(document__id__gt=document_id)
        )
    )


__all__ = ["DbDocumentsRepository"]
