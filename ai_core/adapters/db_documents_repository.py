from __future__ import annotations
import base64
from contextlib import nullcontext
from datetime import datetime
from typing import List, Mapping, Optional, Tuple
from uuid import UUID
import logging

from django.apps import apps
from django.db import IntegrityError, models, transaction
from django.utils import timezone
from django_tenants.utils import schema_context

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
                media_type=doc.blob.media_type,  # Preserve media_type
            )
        return doc

    def _resolve_tenant(self, tenant_id: str) -> models.Model:
        """Resolve tenant by schema_name only, enforcing strict string policy."""
        Tenant = apps.get_model("customers", "Tenant")
        try:
            return Tenant.objects.get(schema_name=tenant_id)
        except Tenant.DoesNotExist:
            raise ValueError(f"tenant_not_found: {tenant_id}")

    def _schema_ctx(
        self, scope: Optional[ScopeContext], tenant: models.Model
    ):  # pragma: no cover - trivial helper
        schema_name = None
        if scope and getattr(scope, "tenant_schema", None):
            schema_name = scope.tenant_schema
        else:
            schema_name = getattr(tenant, "schema_name", None)
        return schema_context(schema_name) if schema_name else nullcontext()

    def upsert(
        self,
        doc: NormalizedDocument,
        workflow_id: Optional[str] = None,
        scope: Optional[ScopeContext] = None,
        audit_meta: Optional[Mapping[str, object]] = None,
    ) -> NormalizedDocument:
        # 1. Handle Frozen Models & Materialization safely
        # We cannot mutate 'doc' if it is frozen. We create a modified copy.
        doc_copy = self._materialize_document_safe(doc)

        workflow = workflow_id or doc_copy.ref.workflow_id
        if workflow != doc_copy.ref.workflow_id:
            raise ValueError("workflow_mismatch")

        collection_id = (
            doc_copy.ref.collection_id or doc_copy.meta.document_collection_id
        )

        # 2. Robust Upsert Strategy
        # "Get-Modify-Update" is safer than "Insert-Fail-Recover" for heavy collision rates
        # and avoids aborted transaction noise.

        # Resolve Tenant ID strictly
        tenant = self._resolve_tenant(doc_copy.ref.tenant_id)

        Document = apps.get_model("documents", "Document")
        DocumentCollection = apps.get_model("documents", "DocumentCollection")
        DocumentCollectionMembership = apps.get_model(
            "documents", "DocumentCollectionMembership"
        )
        User = apps.get_model("users", "User")

        with self._schema_ctx(scope, tenant):
            created_by_user = None
            created_by_user_id = (audit_meta or {}).get("created_by_user_id")
            last_hop_service_id = (audit_meta or {}).get("last_hop_service_id")
            if created_by_user_id:
                try:
                    created_by_user = User.objects.get(pk=created_by_user_id)
                except Exception:
                    logger.warning(
                        "documents.created_by_user_missing",
                        extra={
                            "user_id": created_by_user_id,
                            "tenant_id": str(tenant.id),
                        },
                    )

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

            # Strategy 1: Find by business key (tenant, source, hash)
            try:
                document = Document.objects.get(
                    tenant=tenant, source=doc_copy.source or "", hash=doc_copy.checksum
                )
            except Document.DoesNotExist:
                pass

            # Strategy 2: If not found by hash, try finding by document_id (for re-uploads)
            if document is None and doc_copy.ref.document_id:
                try:
                    document = Document.objects.get(
                        id=doc_copy.ref.document_id, tenant=tenant
                    )
                except Document.DoesNotExist:
                    pass

            # Materialize blob to FileBlob before persistence
            doc_copy = self._materialize_document_safe(doc_copy)

            if document:
                # Align ref/document_id to the existing row to avoid mismatches
                doc_copy = doc_copy.model_copy(
                    update={
                        "ref": doc_copy.ref.model_copy(
                            update={"document_id": document.id}, deep=True
                        )
                    },
                    deep=True,
                )
                metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
                document = self._update_document_instance(
                    document, doc_copy, metadata, scope=scope, workflow_id=workflow
                )
            else:
                metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
                # Create new - handle race condition
                try:
                    with transaction.atomic():
                        # Extract context IDs from scope or doc metadata
                        ctx_workflow_id = workflow or ""
                        ctx_trace_id = scope.trace_id if scope else ""
                        # BREAKING CHANGE (Option A): case_id no longer in ScopeContext
                        # TODO: Accept BusinessContext parameter to extract case_id
                        ctx_case_id = None  # Was: scope.case_id if scope else None

                        document = Document.objects.create(
                            id=doc_copy.ref.document_id,  # Honor ID if creating new
                            tenant=tenant,
                            hash=doc_copy.checksum,
                            source=doc_copy.source or "",
                            metadata=metadata,
                            workflow_id=ctx_workflow_id,
                            trace_id=ctx_trace_id,
                            case_id=ctx_case_id,
                            lifecycle_state=doc_copy.lifecycle_state,
                            lifecycle_updated_at=doc_copy.created_at,
                            created_by=created_by_user,
                            updated_by=created_by_user,
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
                            # Cannot find document by ID (globally) OR by Hash.
                            # Yet create() failed with IntegrityError.
                            logger.error(
                                "upsert_integrity_failure_analysis",
                                extra={
                                    "document_id": str(doc_copy.ref.document_id),
                                    "tenant_id": str(tenant.id),
                                    "source": doc_copy.source or "",
                                    "hash": doc_copy.checksum,
                                    "schema": getattr(tenant, "schema_name", "unknown"),
                                },
                            )
                            raise

                    # 3. Post-recovery Validation
                    if document and document.tenant_id != tenant.id:
                        # Found by ID but tenant mismatch
                        logger.error(
                            "upsert_id_collision_cross_tenant",
                            extra={
                                "document_id": str(document.id),
                                "existing_tenant": str(document.tenant_id),
                                "target_tenant": str(tenant.id),
                            },
                        )
                        # Start fallback: Create new ID for this document?
                        # Currently we just fail, but this explains WHY.
                        raise ValueError("Document ID collision across tenants")
                    # Align ref/metadata to the persisted row
                    doc_copy = doc_copy.model_copy(
                        update={
                            "ref": doc_copy.ref.model_copy(
                                update={"document_id": document.id}, deep=True
                            )
                        },
                        deep=True,
                    )
                    metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
                    document = self._update_document_instance(
                        document, doc_copy, metadata, scope=scope, workflow_id=workflow
                    )

            if document and created_by_user and not getattr(document, "created_by_id"):
                document.created_by = created_by_user
                document.save(update_fields=["created_by", "updated_at"])

            if document and created_by_user:
                document.updated_by = created_by_user
                document.save(update_fields=["updated_by", "updated_at"])

            # 3. Memberships (Side Effects)
            if coll:
                try:
                    membership_defaults: dict[str, object] = {}
                    if created_by_user:
                        membership_defaults["added_by_user"] = created_by_user
                    elif last_hop_service_id:
                        membership_defaults["added_by_service_id"] = last_hop_service_id
                    DocumentCollectionMembership.objects.get_or_create(
                        document=document,
                        collection=coll,
                        defaults=membership_defaults or None,
                    )
                except Exception:
                    # Don't fail the main upsert if side effects flake, but log
                    logger.warning(
                        "collection_membership_failed",
                        exc_info=True,
                        extra={
                            "document_id": str(document.id),
                            "collection_id": str(collection_id),
                        },
                    )

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
        """Materialize transient blobs (InlineBlob, LocalFileBlob) to permanent FileBlob.

        Returns a new NormalizedDocument with the blob persisted to object storage.
        Idempotent: FileBlob and ExternalBlob pass through unchanged.
        """
        from pathlib import Path
        from documents.contracts import LocalFileBlob
        from common.assets.hashing import sha256_bytes

        if isinstance(doc.blob, InlineBlob):
            data = base64.b64decode(doc.blob.base64)
            uri, _, _ = self._storage.put(data)
            new_blob = FileBlob(
                type="file",
                uri=uri,
                sha256=doc.blob.sha256,
                size=doc.blob.size,
                media_type=doc.blob.media_type,  # Preserve media_type
            )
            return doc.model_copy(update={"blob": new_blob}, deep=True)

        if isinstance(doc.blob, LocalFileBlob):
            local_path = Path(doc.blob.path)
            if not local_path.exists():
                raise ValueError(f"local_blob_missing: {doc.blob.path}")
            data = local_path.read_bytes()
            checksum = sha256_bytes(data)
            uri, _, _ = self._storage.put(data)
            new_blob = FileBlob(
                type="file",
                uri=uri,
                sha256=checksum,
                size=len(data),
                media_type=doc.blob.media_type,  # Preserve media_type from LocalFileBlob
            )
            return doc.model_copy(update={"blob": new_blob}, deep=True)

        return doc  # FileBlob, ExternalBlob: no change needed

    def _update_document_instance(
        self, document, doc_copy, metadata, scope=None, workflow_id=None
    ):
        """Update document instance with metadata and context fields."""
        document.metadata = metadata
        document.lifecycle_state = doc_copy.lifecycle_state
        document.lifecycle_updated_at = doc_copy.created_at

        # Update context fields if provided
        update_fields = ["metadata", "lifecycle_state", "lifecycle_updated_at"]
        if workflow_id:
            document.workflow_id = workflow_id
            update_fields.append("workflow_id")
        if scope and scope.trace_id:
            document.trace_id = scope.trace_id
            update_fields.append("trace_id")
        # BREAKING CHANGE (Option A): case_id no longer in ScopeContext
        # TODO: Accept BusinessContext parameter to extract case_id
        # Removed: if scope and scope.case_id: document.case_id = scope.case_id

        document.save(update_fields=update_fields)
        return document

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
        workflow_id: Optional[str] = None,
        include_retired: bool = False,
    ) -> Optional[NormalizedDocument]:
        Document = apps.get_model("documents", "Document")

        tenant = self._resolve_tenant(tenant_id)
        with self._schema_ctx(None, tenant):
            document = Document.objects.filter(
                tenant=tenant,
                id=document_id,
            ).first()
            if document is None:
                logger.info(
                    "db_documents_repository_get_missing reason=not_found document_id=%s tenant=%s",
                    document_id,
                    tenant_id,
                    extra={
                        "event": "db_documents_repository_get_missing",
                        "tenant_schema": getattr(tenant, "schema_name", None),
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "reason": "not_found",
                    },
                )
                return None
            lifecycle_state = (document.lifecycle_state or "").strip().lower()
            # Only filter out terminal states when not explicitly requesting retired docs
            # Ingestion expects pending/ingesting documents to be readable.
            if not include_retired and lifecycle_state in {"deleted", "retired"}:
                logger.info(
                    "db_documents_repository_get_missing reason=filtered_state:%s document_id=%s tenant=%s",
                    lifecycle_state,
                    document_id,
                    tenant_id,
                    extra={
                        "event": "db_documents_repository_get_missing",
                        "tenant_schema": getattr(tenant, "schema_name", None),
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "reason": f"filtered_state:{lifecycle_state}",
                    },
                )
                return None

            normalized = _build_document_from_metadata(document)
            if normalized is None:
                logger.info(
                    "db_documents_repository_get_missing reason=metadata_missing document_id=%s tenant=%s metadata_keys=%s",
                    document_id,
                    tenant_id,
                    list((document.metadata or {}).keys()),
                    extra={
                        "event": "db_documents_repository_get_missing",
                        "tenant_schema": getattr(tenant, "schema_name", None),
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "reason": "metadata_missing",
                        "metadata_keys": list((document.metadata or {}).keys()),
                    },
                )
                return None

            # Use the document's lifecycle_state field directly
            normalized.lifecycle_state = document.lifecycle_state

            logger.info(
                "db_documents_repository_get_hit document_id=%s tenant=%s lifecycle=%s metadata_keys=%s",
                document_id,
                tenant_id,
                document.lifecycle_state,
                list((document.metadata or {}).keys()),
                extra={
                    "event": "db_documents_repository_get_hit",
                    "tenant_schema": getattr(tenant, "schema_name", None),
                    "tenant_id": tenant_id,
                    "document_id": str(document_id),
                    "lifecycle_state": document.lifecycle_state,
                    "metadata_keys": list((document.metadata or {}).keys()),
                },
            )

            # Attach persisted assets to mirror in-memory behaviour.
            # Only override if we find assets in the DB; otherwise keep the ones from metadata.
            try:
                db_assets = []
                for asset_row in document.assets.all():
                    asset = self._build_asset_from_model(asset_row)
                    if asset:
                        db_assets.append(asset)
                # Only override if DB has assets; metadata may already have them serialized
                if db_assets:
                    normalized.assets = db_assets
            except Exception:
                logger.warning(
                    "document_asset_attach_failed",
                    exc_info=True,
                    extra={"document_id": str(document_id), "tenant_id": tenant_id},
                )

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

        tenant = self._resolve_tenant(tenant_id)
        with self._schema_ctx(None, tenant):
            memberships = self._collection_queryset(tenant, collection_id, workflow_id)
            memberships = _apply_cursor_filter(memberships, cursor)

            entries: list[tuple[tuple, NormalizedDocument]] = []
            for membership in memberships[: limit + 1]:
                try:
                    document = membership.document
                    normalized = _build_document_from_metadata(document)
                    if normalized is None:
                        continue
                    try:
                        assets = []
                        for asset_row in document.assets.all():
                            asset = self._build_asset_from_model(asset_row)
                            if asset:
                                assets.append(asset)
                        normalized.assets = assets
                    except Exception:
                        logger.warning(
                            "documents.list_by_collection.asset_attach_failed",
                            extra={
                                "membership_id": str(membership.id),
                                "document_id": str(document.id),
                                "tenant_id": tenant_id,
                            },
                            exc_info=True,
                        )
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
        tenant = self._resolve_tenant(tenant_id)
        with self._schema_ctx(None, tenant):
            memberships = self._collection_queryset(tenant, collection_id, workflow_id)
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
            logger.info(
                f"DEBUG: list_latest_by_collection: tenant={tenant_id} collection={collection_id}"
            )
            logger.info(f"DEBUG: memberships_count: {memberships.count()}")
            for membership in memberships[:candidate_limit]:
                try:
                    document = membership.document
                    normalized = _build_document_from_metadata(document)
                    if normalized is None:
                        continue
                    try:
                        assets = []
                        for asset_row in document.assets.all():
                            asset = self._build_asset_from_model(asset_row)
                            if asset:
                                assets.append(asset)
                        normalized.assets = assets
                    except Exception:
                        logger.warning(
                            "documents.list_latest_by_collection.asset_attach_failed",
                            extra={
                                "membership_id": str(membership.id),
                                "document_id": str(document.id),
                                "tenant_id": tenant_id,
                            },
                            exc_info=True,
                        )

                    doc_ref = normalized.ref
                    if workflow_id and doc_ref.workflow_id != workflow_id:
                        continue

                    existing = seen_docs.get(doc_ref.document_id)
                    if existing is None:
                        seen_docs[doc_ref.document_id] = normalized
                    else:
                        if normalized.created_at > existing.created_at:
                            seen_docs[doc_ref.document_id] = normalized
                        elif normalized.created_at == existing.created_at:
                            if (normalized.source or "") > (existing.source or ""):
                                seen_docs[doc_ref.document_id] = normalized

                except Exception:
                    logger.warning(
                        "documents.list_latest_by_collection.entry_failed",
                        extra={"membership_id": str(membership.id)},
                        exc_info=True,
                    )
                    continue

            # Convert map back to list
            logger.info(f"DEBUG: seen_docs count: {len(seen_docs)}")
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

        with self._schema_ctx(None, tenant):
            # Resolve Parent Document (Latest/Any matching)
            try:
                document_row = Document.objects.get(
                    id=asset_copy.ref.document_id, tenant=tenant
                )
            except Document.DoesNotExist:
                raise ValueError(
                    f"Parent document not found: {asset_copy.ref.document_id}"
                )

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
                    "metadata": asset_copy.model_dump(
                        mode="json", exclude={"blob", "ref"}
                    ),
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
        with self._schema_ctx(None, tenant):
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
        with self._schema_ctx(None, tenant):
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
        with self._schema_ctx(None, tenant):
            qs = DocumentAsset.objects.filter(tenant=tenant, asset_id=asset_id)
            if workflow_id:
                qs = qs.filter(workflow_id=workflow_id)

            count, _ = qs.delete()  # Hard delete as per instructions
            return count > 0

    def _collection_queryset(
        self, tenant: models.Model, collection_id: UUID, workflow_id: Optional[str]
    ):
        DocumentCollectionMembership = apps.get_model(
            "documents", "DocumentCollectionMembership"
        )
        # Secure filter: Ensure collection belongs to tenant
        queryset = DocumentCollectionMembership.objects.filter(
            collection__tenant=tenant, collection__collection_id=collection_id
        )

        if workflow_id:
            queryset = queryset.filter(
                document__metadata__normalized_document__ref__workflow_id=workflow_id
            )

        # Allow documents in all lifecycle states for Document Explorer visibility
        return queryset.order_by("-added_at", "document_id")

    def delete(
        self,
        tenant_id: str,
        document_id: UUID,
        *,
        workflow_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        Document = apps.get_model("documents", "Document")
        tenant = self._resolve_tenant(tenant_id)

        with self._schema_ctx(None, tenant):
            document = Document.objects.filter(tenant=tenant, id=document_id).first()
            if document is None:
                return False

            if hard:
                # Cascades will remove memberships/assets automatically
                document.delete()
                return True

            if document.lifecycle_state == "retired":
                return False

            document.lifecycle_state = "retired"
            document.lifecycle_updated_at = timezone.now()
            document.save(update_fields=["lifecycle_state", "lifecycle_updated_at"])
            return True

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
                media_type=asset.blob.media_type,  # Preserve media_type
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
