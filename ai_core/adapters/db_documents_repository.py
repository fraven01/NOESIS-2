from __future__ import annotations
import base64
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID, uuid4
import logging

from django.apps import apps
from django.db import IntegrityError, models, transaction

from ai_core.contracts.scope import ScopeContext
from documents.contracts import (
    DocumentRef,
    FileBlob,
    InlineBlob,
    NormalizedDocument,
    Asset,
    AssetRef,
)
from documents.repository import (
    DocumentsRepository,
    _workflow_storage_key,
)
from documents.storage import ObjectStoreStorage, Storage


logger = logging.getLogger(__name__)


class DbDocumentsRepository(DocumentsRepository):
    """Database-backed repository using Django models for metadata."""

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self._storage = storage or ObjectStoreStorage()

    def _materialize_document(self, doc: NormalizedDocument) -> NormalizedDocument:
        """Persist inline blobs to storage and return document with file references."""
        if isinstance(doc.blob, InlineBlob):
            data = base64.b64decode(doc.blob.base64)
            uri, _, _ = self._storage.put(data)
            doc.blob = FileBlob(
                type="file",
                uri=uri,
                sha256=doc.blob.sha256,
                size=doc.blob.size,
            )
        return doc

    def _resolve_tenant(self, tenant_id: str) -> models.Model:
        """Resolve tenant by schema_name only, enforcing strict string policy."""
        Tenant = apps.get_model("customers", "Tenant")
        try:
            return Tenant.objects.get(schema_name=tenant_id)
        except Tenant.DoesNotExist:
            raise ValueError(f"tenant_not_found: {tenant_id}")

    def upsert(
        self,
        doc: NormalizedDocument,
        workflow_id: Optional[str] = None,
        scope: Optional[ScopeContext] = None,
    ) -> NormalizedDocument:
        # 1. Handle Frozen Models & Materialization safely
        # We cannot mutate 'doc' if it is frozen. We create a modified copy.
        doc_copy = self._materialize_document_safe(doc)

        workflow = workflow_id or doc_copy.ref.workflow_id
        if workflow != doc_copy.ref.workflow_id:
            raise ValueError("workflow_mismatch")

        workflow_key = _workflow_storage_key(workflow)

        Document = apps.get_model("documents", "Document")
        DocumentCollection = apps.get_model("documents", "DocumentCollection")
        DocumentCollectionMembership = apps.get_model(
            "documents", "DocumentCollectionMembership"
        )
        DocumentLifecycleState = apps.get_model("documents", "DocumentLifecycleState")

        metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
        collection_id = (
            doc_copy.ref.collection_id or doc_copy.meta.document_collection_id
        )

        # 2. Robust Upsert Strategy
        # "Get-Modify-Update" is safer than "Insert-Fail-Recover" for heavy collision rates
        # and avoids aborted transaction noise.

        # Resolve Tenant ID strictly
        tenant = self._resolve_tenant(doc_copy.ref.tenant_id)

        # Fail fast if collection is specified but missing (Logic Change)
        coll = None
        if collection_id:
            try:
                coll = DocumentCollection.objects.get(
                    tenant=tenant, collection_id=collection_id
                )
            except DocumentCollection.DoesNotExist:
                # Specific error as requested
                raise ValueError(f"Collection not found: {collection_id}")

        document = None

        # Strategy: Try finding existing document by strictly unique business key
        try:
            document = Document.objects.get(
                tenant=tenant, source=doc_copy.source or "", hash=doc_copy.checksum
            )
        except Document.DoesNotExist:
            pass

        if document:
            # Update existing
            document = self._update_document_instance(document, doc_copy, metadata)
        else:
            # Create new - handle race condition
            try:
                with transaction.atomic():
                    document = Document.objects.create(
                        id=doc_copy.ref.document_id,  # Honor ID if creating new
                        tenant=tenant,
                        hash=doc_copy.checksum,
                        source=doc_copy.source or "",
                        metadata=metadata,
                        lifecycle_state=doc_copy.lifecycle_state,
                        lifecycle_updated_at=doc_copy.created_at,
                    )
            except IntegrityError:
                # Race condition or ID collision
                # 1. Try finding by ID (if we forced one)
                try:
                    document = Document.objects.get(
                        id=doc_copy.ref.document_id, tenant=tenant
                    )
                except Document.DoesNotExist:
                    # 2. Try finding by business key (source + hash)
                    try:
                        document = Document.objects.get(
                            tenant=tenant,
                            source=doc_copy.source or "",
                            hash=doc_copy.checksum,
                        )
                    except Document.DoesNotExist:
                        # Should not happen if IntegrityError was genuine, unless race led to delete?
                        raise
                document = self._update_document_instance(document, doc_copy, metadata)

        # 3. Memberships & Lifecycle (Side Effects)
        try:
            if coll:
                DocumentCollectionMembership.objects.get_or_create(
                    document=document,
                    collection=coll,
                    defaults={"added_by": "system"},
                )

            trace_id = scope.trace_id if scope else uuid4().hex
            run_id_value = ""
            ingestion_run_id_value = ""
            if scope:
                run_id_value = scope.run_id or ""
                ingestion_run_id_value = scope.ingestion_run_id or ""
            else:
                run_id_value = "manual"

            DocumentLifecycleState.objects.update_or_create(
                tenant_id=tenant,
                document_id=document.id,
                workflow_id=workflow_key,
                defaults={
                    "state": doc_copy.lifecycle_state,
                    "changed_at": doc_copy.created_at,
                    "reason": "",
                    "policy_events": [],
                    "trace_id": trace_id,
                    "run_id": run_id_value,
                    "ingestion_run_id": ingestion_run_id_value,
                },
            )
        except Exception:
            # Don't fail the main upsert if side effects flake, but log?
            # For now, let it bubble as it signifies a DB consistency issue
            raise

        return (
            self.get(
                doc_copy.ref.tenant_id,
                document.id,
                doc_copy.ref.version,
                workflow_id=workflow,
                prefer_latest=True,
            )
            or doc_copy
        )

    def _materialize_document_safe(self, doc: NormalizedDocument) -> NormalizedDocument:
        """Return a new NormalizedDocument with materialized blobs, handling immutability."""
        if not isinstance(doc.blob, InlineBlob):
            return doc  # No change needed

        # Prepare new blob
        data = base64.b64decode(doc.blob.base64)
        uri, _, _ = self._storage.put(data)
        new_blob = FileBlob(
            type="file",
            uri=uri,
            sha256=doc.blob.sha256,
            size=doc.blob.size,
        )

        # Pydantic v2 copy with update
        return doc.model_copy(update={"blob": new_blob}, deep=True)

    def _update_document_instance(self, document, doc_copy, metadata):
        document.metadata = metadata
        document.lifecycle_state = doc_copy.lifecycle_state
        document.lifecycle_updated_at = doc_copy.created_at
        document.save(
            update_fields=["metadata", "lifecycle_state", "lifecycle_updated_at"]
        )
        return document

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

        tenant = self._resolve_tenant(tenant_id)
        document = Document.objects.filter(
            tenant=tenant,
            id=document_id,
        ).first()
        if document is None:
            return None

        lifecycle = _select_lifecycle_state(
            lifecycle_model, tenant, document_id, workflow_id
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

        memberships = self._collection_queryset(tenant_id, collection_id, workflow_id)
        memberships = _apply_cursor_filter(memberships, cursor)

        entries: list[tuple[tuple, NormalizedDocument]] = []
        for membership in memberships[: limit + 1]:
            try:
                document = membership.document
                normalized = _build_document_from_metadata(document)
                if normalized is None:
                    continue
                entries.append(self._document_entry(normalized))
            except Exception:
                logger.warning(
                    "documents.list_by_collection.entry_failed",
                    extra={"membership_id": str(membership.id)},
                    exc_info=True,
                )
                continue

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
        memberships = self._collection_queryset(tenant_id, collection_id, workflow_id)
        # Fetch more than limit to allow python-side deduplication
        # We might need a lot more candidates if there are many versions per doc.
        # This is a naive implementation but safer for consistency.
        candidate_limit = limit * 5
        memberships = _apply_cursor_filter(memberships, cursor)

        entries: list[tuple[tuple, NormalizedDocument]] = []
        seen_docs: dict[UUID, NormalizedDocument] = {}

        # Iterate through candidates
        # Note: memberships are ordered by -added_at (roughly creation time).
        # We need to find the LATEST version per document_id.
        for membership in memberships[:candidate_limit]:
            try:
                document = membership.document
                normalized = _build_document_from_metadata(document)
                if normalized is None:
                    continue

                doc_ref = normalized.ref
                if workflow_id and doc_ref.workflow_id != workflow_id:
                    continue

                # Deduplication logic: Keep the "newest" one.
                # Since we accept arbitrary versions, we need a way to compare.
                # Common pattern: created_at timestamp.

                existing = seen_docs.get(doc_ref.document_id)
                if existing is None:
                    seen_docs[doc_ref.document_id] = normalized
                else:
                    # If current is newer than existing, replace it
                    if normalized.created_at > existing.created_at:
                        seen_docs[doc_ref.document_id] = normalized
                    elif normalized.created_at == existing.created_at:
                        if (normalized.source or "") > (
                            existing.source or ""
                        ):  # Tie-break
                            seen_docs[doc_ref.document_id] = normalized

            except Exception:
                logger.warning(
                    "documents.list_latest_by_collection.entry_failed",
                    extra={"membership_id": str(membership.id)},
                    exc_info=True,
                )
                continue

        # Convert map back to list
        for doc in seen_docs.values():
            entries.append(self._document_entry(doc))

        entries.sort(key=lambda entry: entry[0])
        refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
        next_cursor = self._next_cursor(entries, 0, limit)
        return refs, next_cursor

    def _document_entry(
        self, doc: NormalizedDocument
    ) -> tuple[tuple, NormalizedDocument]:
        version_key = doc.ref.version or ""
        key = (
            -doc.created_at.timestamp(),
            str(doc.ref.document_id),
            doc.ref.workflow_id or "",
            version_key,
        )
        return key, doc

    def _asset_entry(self, asset: Asset) -> tuple[tuple, Asset]:
        key = (
            -asset.created_at.timestamp(),
            str(asset.ref.asset_id),
            asset.ref.workflow_id or "",
            "",
        )
        return key, asset

    def _next_cursor(self, entries: list, start: int, limit: int) -> Optional[str]:
        end = start + limit
        if end >= len(entries):
            return None
        key, obj = entries[end - 1]

        if isinstance(obj, NormalizedDocument):
            parts = [
                obj.created_at.isoformat(),
                str(obj.ref.document_id),
                obj.ref.workflow_id or "",
                obj.ref.version or "",
            ]
        elif isinstance(obj, Asset):
            parts = [
                obj.created_at.isoformat(),
                str(obj.ref.asset_id),
                obj.ref.workflow_id or "",
                "",
            ]
        else:
            return None

        return _encode_cursor(parts)

    def add_asset(self, asset: Asset, workflow_id: Optional[str] = None) -> Asset:
        DocumentAsset = apps.get_model("documents", "DocumentAsset")
        Document = apps.get_model("documents", "Document")

        asset_copy = self._materialize_asset(asset)

        # Resolve Tenant
        tenant = self._resolve_tenant(asset_copy.ref.tenant_id)

        # Resolve Parent Document (Latest/Any matching)
        try:
            document_row = Document.objects.get(
                id=asset_copy.ref.document_id, tenant=tenant
            )
        except Document.DoesNotExist:
            raise ValueError(f"Parent document not found: {asset_copy.ref.document_id}")

        blob_meta = {}
        if hasattr(asset_copy.blob, "uri"):
            blob_meta = {"uri": asset_copy.blob.uri, "type": "file"}

        # Create
        DocumentAsset.objects.update_or_create(
            tenant=tenant,
            asset_id=asset_copy.ref.asset_id,
            workflow_id=workflow_id or asset_copy.ref.workflow_id,
            defaults={
                "document": document_row,
                "collection_id": asset_copy.ref.collection_id,
                "media_type": asset_copy.media_type,
                "blob_metadata": blob_meta,
                # Store content if meaningful
                "content": asset_copy.text_description or asset_copy.ocr_text,
                "metadata": asset_copy.model_dump(mode="json", exclude={"blob", "ref"}),
            },
        )
        return asset_copy

    def get_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
    ) -> Optional[Asset]:
        DocumentAsset = apps.get_model("documents", "DocumentAsset")

        tenant = self._resolve_tenant(tenant_id)
        qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
        if workflow_id:
            qs = qs.filter(workflow_id=workflow_id)

        instance = qs.first()
        if not instance:
            return None

        return self._build_asset_from_model(instance)

    def list_assets_by_document(
        self,
        tenant_id: str,
        document_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
    ) -> Tuple[List[AssetRef], Optional[str]]:
        DocumentAsset = apps.get_model("documents", "DocumentAsset")

        tenant = self._resolve_tenant(tenant_id)
        qs = DocumentAsset.objects.filter(tenant=tenant, document__id=document_id)
        if workflow_id:
            qs = qs.filter(workflow_id=workflow_id)

        qs = qs.order_by("-created_at")

        # Basic cursor support (DB based)
        entries = []
        # Fetch Limit + 1
        for instance in qs[: limit + 1]:
            asset = self._build_asset_from_model(instance)
            if asset:
                entries.append(self._asset_entry(asset))

        entries.sort(key=lambda entry: entry[0])
        refs = [asset.ref for _, asset in entries[:limit]]
        next_cursor = self._next_cursor(entries, 0, limit)
        return refs, next_cursor

    def delete_asset(
        self,
        tenant_id: str,
        asset_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        DocumentAsset = apps.get_model("documents", "DocumentAsset")
        tenant = self._resolve_tenant(tenant_id)
        qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
        if workflow_id:
            qs = qs.filter(workflow_id=workflow_id)

        count, _ = qs.delete()  # Hard delete as per instructions
        return count > 0

    def _collection_queryset(
        self, tenant_id: str, collection_id: UUID, workflow_id: Optional[str]
    ):
        DocumentCollectionMembership = apps.get_model(
            "documents", "DocumentCollectionMembership"
        )
        lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")
        workflow_key = _workflow_storage_key(workflow_id)
        tenant = self._resolve_tenant(tenant_id)

        # Secure filter: Ensure collection belongs to tenant
        queryset = DocumentCollectionMembership.objects.filter(
            collection__tenant=tenant, collection__collection_id=collection_id
        )

        if workflow_id is not None:
            lifecycle_exists = models.Exists(
                lifecycle_model.objects.filter(
                    tenant_id=tenant,
                    workflow_id=workflow_key,
                    document_id=models.OuterRef("document__id"),
                )
            )
            queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(
                has_lifecycle=True
            )

        return queryset.order_by("-added_at", "document_id")

    def _materialize_asset(self, asset: Asset) -> Asset:
        """Ensure asset blobs are stored."""
        if isinstance(asset.blob, InlineBlob):
            data = base64.b64decode(asset.blob.base64)
            uri, _, _ = self._storage.put(data)
            asset.blob = FileBlob(
                type="file",
                uri=uri,
                sha256=asset.blob.sha256,
                size=asset.blob.size,
            )
        return asset

    def _build_asset_from_model(self, instance) -> Optional[Asset]:
        # Reconstruct Asset Pydantic model
        # Metadata field stores the bulk of it.
        meta = instance.metadata or {}

        # Blob logic
        b_meta = instance.blob_metadata or {}
        # Reconstruct file blob from metadata
        blob = None
        if b_meta.get("uri"):
            # Assuming we only store FileBlobs or similar references
            blob = FileBlob(
                type="file", uri=b_meta["uri"], sha256=meta.get("checksum", ""), size=0
            )
        else:
            # Fallback dummy blob if missing? Or fail?
            blob = InlineBlob(
                type="inline",
                media_type="application/octet-stream",
                base64="",
                sha256="",
                size=0,
            )

        ref = AssetRef(
            tenant_id=instance.tenant.schema_name,
            workflow_id=instance.workflow_id,
            asset_id=instance.asset_id,
            document_id=instance.document.id,
            collection_id=instance.collection_id,
        )

        try:
            return Asset(
                ref=ref,
                media_type=instance.media_type,
                blob=blob,
                origin_uri=meta.get("origin_uri"),
                page_index=meta.get("page_index"),
                bbox=meta.get("bbox"),
                context_before=meta.get("context_before"),
                context_after=meta.get("context_after"),
                ocr_text=meta.get("ocr_text") or instance.content,
                text_description=meta.get("text_description"),
                caption_source=meta.get("caption_source", "none"),
                caption_method=meta.get("caption_method", "none"),
                caption_model=meta.get("caption_model"),
                caption_confidence=meta.get("caption_confidence"),
                created_at=instance.created_at,
                checksum=meta.get("checksum", ""),
            )
        except Exception:
            logger.warning("asset_reconstruct_failed", exc_info=True)
            return None


def _encode_cursor(parts: list[str]) -> str:
    payload = "|".join(parts)
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
    return encoded.decode("ascii")


def _decode_cursor(cursor: str) -> list[str]:
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
        text = decoded.decode("utf-8")
        return text.split("|")
    except Exception as exc:
        raise ValueError("cursor_invalid") from exc


def _build_document_from_metadata(document) -> Optional[NormalizedDocument]:
    payload = document.metadata or {}
    normalized_payload = payload.get("normalized_document")

    if normalized_payload:
        normalized = NormalizedDocument.model_validate(normalized_payload)
    else:
        # Removal of lazy shim logic as per requirements.
        return None

    if normalized is None:
        return None

    # Align timestamps with the persisted row to support cursor pagination.
    normalized.created_at = document.created_at
    return normalized


def _select_lifecycle_state(
    model, tenant, document_id: UUID, workflow_id: Optional[str]
):
    workflow_key = _workflow_storage_key(workflow_id)
    filters = {"tenant_id": tenant, "document_id": document_id}
    if workflow_id is not None:
        filters["workflow_id"] = workflow_key
    qs = model.objects.filter(**filters).order_by("-changed_at")
    return qs.first()


def _apply_cursor_filter(queryset, cursor: Optional[str]):
    if not cursor:
        return queryset

    parts = _decode_cursor(cursor)
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
