"""Repository abstractions for storing normalized documents and assets."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import UUID

from common.logging import log_context

from .contracts import (
    Asset,
    AssetRef,
    BlobLocator,
    DocumentRef,
    NormalizedDocument,
    InlineBlob,
    FileBlob,
)
from .storage import InMemoryStorage, Storage
from .logging_utils import (
    asset_log_fields,
    document_log_fields,
    log_call,
    log_extra_entry,
    log_extra_exit,
)


class DocumentsRepository:
    """Abstract persistence interface for normalized documents."""

    def upsert(self, doc: NormalizedDocument) -> NormalizedDocument:
        """Create or replace a document instance."""

        raise NotImplementedError

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
    ) -> Optional[NormalizedDocument]:
        """Fetch a document by identifiers, returning ``None`` if missing."""

        raise NotImplementedError

    def list_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = False,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List document references for a collection ordered by recency.

        The returned cursor is a best-effort marker derived from created_at and
        document identifiers and may change when records are reordered.
        """

        raise NotImplementedError

    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List newest document versions per document identifier.

        The returned cursor is a best-effort marker derived from created_at and
        document identifiers and may change when records are reordered.
        """

        raise NotImplementedError

    def delete(
        self, tenant_id: str, document_id: UUID, hard: bool = False
    ) -> bool:
        """Soft or hard delete a document across all versions."""

        raise NotImplementedError

    def add_asset(self, asset: Asset) -> Asset:
        """Persist an asset for a previously stored document."""

        raise NotImplementedError

    def get_asset(self, tenant_id: str, asset_id: UUID) -> Optional[Asset]:
        """Fetch an asset by its identifier."""

        raise NotImplementedError

    def list_assets_by_document(
        self,
        tenant_id: str,
        document_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[AssetRef], Optional[str]]:
        """List asset references for a document ordered by recency.

        The returned cursor is a best-effort marker derived from created_at and
        asset identifiers and may change when records are reordered.
        """

        raise NotImplementedError

    def delete_asset(
        self, tenant_id: str, asset_id: UUID, hard: bool = False
    ) -> bool:
        """Soft or hard delete an asset."""

        raise NotImplementedError


@dataclass
class _StoredDocument:
    value: NormalizedDocument
    deleted: bool = False


@dataclass
class _StoredAsset:
    value: Asset
    deleted: bool = False


class InMemoryDocumentsRepository(DocumentsRepository):
    """Thread-safe in-memory repository implementation."""

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self._lock = RLock()
        self._storage = storage or InMemoryStorage()
        self._documents: Dict[Tuple[str, UUID, Optional[str]], _StoredDocument] = {}
        self._assets: Dict[Tuple[str, UUID], _StoredAsset] = {}
        self._asset_index: Dict[Tuple[str, UUID], set[UUID]] = {}

    @log_call("docs.upsert")
    def upsert(self, doc: NormalizedDocument) -> NormalizedDocument:
        doc_copy = doc.model_copy(deep=True)
        doc_copy = self._materialize_document(doc_copy)
        ref = doc_copy.ref
        with log_context(tenant=ref.tenant_id, collection_id=str(ref.collection_id) if ref.collection_id else None):
            log_extra_entry(**document_log_fields(doc_copy))
            key = (ref.tenant_id, ref.document_id, ref.version)

            with self._lock:
                self._documents[key] = _StoredDocument(value=doc_copy, deleted=False)

                if doc_copy.assets:
                    for asset in doc_copy.assets:
                        self._store_asset_locked(asset)

                self._refresh_document_assets_locked(ref.tenant_id, ref.document_id)

                stored = self.get(ref.tenant_id, ref.document_id, ref.version)
                log_extra_exit(asset_count=len(doc_copy.assets))
                return stored

    @log_call("docs.get")
    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
    ) -> Optional[NormalizedDocument]:
        with log_context(tenant=tenant_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                version=version,
                prefer_latest=prefer_latest,
            )
            with self._lock:
                if prefer_latest and version is None:
                    latest = self._latest_document_locked(tenant_id, document_id)
                    if latest is None:
                        log_extra_exit(found=False)
                        return None
                    key = (tenant_id, document_id, latest.ref.version)
                else:
                    key = (tenant_id, document_id, version)

                stored = self._documents.get(key)
                if not stored or stored.deleted:
                    log_extra_exit(found=False)
                    return None

                doc_copy = stored.value.model_copy(deep=True)
                doc_copy.assets = self._collect_assets_for_document_locked(
                    tenant_id, document_id
                )
                log_extra_exit(found=True, asset_count=len(doc_copy.assets))
                return doc_copy

    @log_call("docs.list")
    def list_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = False,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List document references for a collection ordered by recency.

        Pagination cursors are best-effort markers derived from ``created_at``
        and document IDs. When ``latest_only`` is ``True`` the method delegates
        to :meth:`list_latest_by_collection` to surface only the most recent
        version per ``document_id``.
        """
        with log_context(tenant=tenant_id, collection_id=str(collection_id)):
            log_extra_entry(
                tenant_id=tenant_id,
                collection_id=collection_id,
                limit=limit,
                cursor_present=bool(cursor),
                latest_only=latest_only,
            )
            if latest_only:
                refs, next_cursor = self.list_latest_by_collection(
                    tenant_id, collection_id, limit=limit, cursor=cursor
                )
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

            with self._lock:
                entries = [
                    self._document_entry(doc)
                    for doc in self._iter_documents_locked(tenant_id, collection_id)
                ]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

    @log_call("docs.list_latest")
    def list_latest_by_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[DocumentRef], Optional[str]]:
        """List the newest document reference per ``document_id``.

        Pagination cursors remain best-effort markers derived from ``created_at``
        and document IDs. Ties on ``created_at`` timestamps are resolved via
        lexicographic version comparison to ensure deterministic ordering.
        """
        with log_context(tenant=tenant_id, collection_id=str(collection_id)):
            log_extra_entry(
                tenant_id=tenant_id,
                collection_id=collection_id,
                limit=limit,
                cursor_present=bool(cursor),
            )
            with self._lock:
                latest: Dict[UUID, NormalizedDocument] = {}
                for doc in self._iter_documents_locked(tenant_id, collection_id):
                    current = latest.get(doc.ref.document_id)
                    if current is None or self._newer(doc, current):
                        latest[doc.ref.document_id] = doc

                entries = [self._document_entry(doc) for doc in latest.values()]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(
                    item_count=len(refs), next_cursor_present=bool(next_cursor)
                )
                return refs, next_cursor

    @log_call("docs.delete")
    def delete(
        self, tenant_id: str, document_id: UUID, hard: bool = False
    ) -> bool:
        with log_context(tenant=tenant_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                hard=hard,
            )
            doc_keys = []
            with self._lock:
                for key, stored in self._documents.items():
                    if key[0] == tenant_id and key[1] == document_id:
                        doc_keys.append(key)

                if not doc_keys:
                    log_extra_exit(found=False)
                    return False

                changed = False
                for key in doc_keys:
                    stored = self._documents.get(key)
                    if stored is None:
                        continue
                    if hard:
                        del self._documents[key]
                        changed = True
                        continue
                    if not stored.deleted:
                        stored.deleted = True
                        changed = True

                self._mark_assets_for_document_locked(tenant_id, document_id, hard=hard)
                log_extra_exit(found=True, deleted=changed)
                return changed

    @log_call("assets.add")
    def add_asset(self, asset: Asset) -> Asset:
        asset_copy = self._materialize_asset(asset.model_copy(deep=True))
        tenant_id = asset_copy.ref.tenant_id
        document_id = asset_copy.ref.document_id

        with log_context(tenant=tenant_id):
            log_extra_entry(**asset_log_fields(asset_copy))
            with self._lock:
                if not self._has_active_document_locked(tenant_id, document_id):
                    raise ValueError("document_missing")

                self._store_asset_locked(asset_copy)
                self._refresh_document_assets_locked(tenant_id, document_id)
                stored = self._snapshot_asset_locked(tenant_id, asset_copy.ref.asset_id)
                log_extra_exit(**asset_log_fields(stored))
                return stored

    @log_call("assets.get")
    def get_asset(self, tenant_id: str, asset_id: UUID) -> Optional[Asset]:
        with log_context(tenant=tenant_id):
            log_extra_entry(tenant_id=tenant_id, asset_id=asset_id)
            with self._lock:
                stored = self._assets.get((tenant_id, asset_id))
                if not stored or stored.deleted:
                    log_extra_exit(found=False)
                    return None
                asset_copy = stored.value.model_copy(deep=True)
                log_extra_exit(found=True, **asset_log_fields(asset_copy))
                return asset_copy

    @log_call("assets.list")
    def list_assets_by_document(
        self,
        tenant_id: str,
        document_id: UUID,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[AssetRef], Optional[str]]:
        with log_context(tenant=tenant_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                limit=limit,
                cursor_present=bool(cursor),
            )
            with self._lock:
                assets = self._collect_assets_for_document_locked(tenant_id, document_id)
                entries = [self._asset_entry(asset) for asset in assets]
                entries.sort(key=lambda entry: entry[0])
                start = self._cursor_start(entries, cursor)
                sliced = entries[start : start + limit]
                refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
                next_cursor = self._next_cursor(entries, start, limit)
                log_extra_exit(item_count=len(refs), next_cursor_present=bool(next_cursor))
                return refs, next_cursor

    @log_call("assets.delete")
    def delete_asset(
        self, tenant_id: str, asset_id: UUID, hard: bool = False
    ) -> bool:
        key = (tenant_id, asset_id)
        with log_context(tenant=tenant_id):
            log_extra_entry(tenant_id=tenant_id, asset_id=asset_id, hard=hard)
            with self._lock:
                stored = self._assets.get(key)
                if not stored:
                    log_extra_exit(found=False)
                    return False

                doc_id = stored.value.ref.document_id
                if hard:
                    del self._assets[key]
                    index = self._asset_index.get((tenant_id, doc_id))
                    if index and asset_id in index:
                        index.remove(asset_id)
                        if not index:
                            del self._asset_index[(tenant_id, doc_id)]
                    self._refresh_document_assets_locked(tenant_id, doc_id)
                    log_extra_exit(found=True, deleted=True)
                    return True

                if stored.deleted:
                    log_extra_exit(found=True, deleted=False)
                    return False

                stored.deleted = True
                self._refresh_document_assets_locked(tenant_id, doc_id)
                log_extra_exit(found=True, deleted=True)
                return True

    # Internal helpers -------------------------------------------------

    def _store_asset_locked(self, asset: Asset) -> None:
        key = (asset.ref.tenant_id, asset.ref.asset_id)
        existing = self._assets.get(key)
        if existing and existing.value.ref.document_id != asset.ref.document_id:
            raise ValueError("asset_conflict")

        asset_copy = asset.model_copy(deep=True)
        self._assets[key] = _StoredAsset(value=asset_copy, deleted=False)

        index_key = (asset.ref.tenant_id, asset.ref.document_id)
        bucket = self._asset_index.setdefault(index_key, set())
        bucket.add(asset.ref.asset_id)

    def _collect_assets_for_document_locked(
        self, tenant_id: str, document_id: UUID
    ) -> List[Asset]:
        asset_ids = self._asset_index.get((tenant_id, document_id), set())
        assets: List[Asset] = []
        for asset_id in asset_ids:
            stored = self._assets.get((tenant_id, asset_id))
            if not stored or stored.deleted:
                continue
            assets.append(stored.value.model_copy(deep=True))

        assets.sort(key=lambda asset: self._asset_entry(asset)[0])
        return assets

    def _mark_assets_for_document_locked(
        self, tenant_id: str, document_id: UUID, hard: bool
    ) -> None:
        asset_ids = list(self._asset_index.get((tenant_id, document_id), set()))
        for asset_id in asset_ids:
            stored = self._assets.get((tenant_id, asset_id))
            if not stored:
                continue
            if hard:
                del self._assets[(tenant_id, asset_id)]
            else:
                stored.deleted = True

        if hard and (tenant_id, document_id) in self._asset_index:
            del self._asset_index[(tenant_id, document_id)]

        self._refresh_document_assets_locked(tenant_id, document_id)

    def _has_active_document_locked(self, tenant_id: str, document_id: UUID) -> bool:
        for key, stored in self._documents.items():
            if key[0] != tenant_id or key[1] != document_id:
                continue
            if not stored.deleted:
                return True
        return False

    def _refresh_document_assets_locked(self, tenant_id: str, document_id: UUID) -> None:
        assets = self._collect_assets_for_document_locked(tenant_id, document_id)
        for key, stored in self._documents.items():
            if key[0] == tenant_id and key[1] == document_id:
                stored.value.assets = [asset.model_copy(deep=True) for asset in assets]

    def _snapshot_asset_locked(self, tenant_id: str, asset_id: UUID) -> Asset:
        stored = self._assets[(tenant_id, asset_id)]
        return stored.value.model_copy(deep=True)

    def _materialize_document(self, doc: NormalizedDocument) -> NormalizedDocument:
        doc.blob = self._materialize_blob(
            doc.blob,
            owner_checksum=doc.checksum,
            checksum_error="document_checksum_mismatch",
        )
        doc.assets = [self._materialize_asset(asset) for asset in doc.assets]
        return doc

    def _materialize_asset(self, asset: Asset) -> Asset:
        asset.blob = self._materialize_blob(
            asset.blob,
            owner_checksum=asset.checksum,
            checksum_error="asset_checksum_mismatch",
        )
        return asset

    def _materialize_blob(
        self,
        blob: BlobLocator,
        *,
        owner_checksum: Optional[str],
        checksum_error: str,
    ) -> BlobLocator:
        if isinstance(blob, InlineBlob):
            payload = blob.decoded_payload()
            uri, sha256, size = self._storage.put(payload)
            if blob.sha256 != sha256:
                raise ValueError("inline_checksum_mismatch")
            if owner_checksum is not None and owner_checksum != sha256:
                raise ValueError(checksum_error)
            return FileBlob(type="file", uri=uri, sha256=sha256, size=size)

        blob_sha = getattr(blob, "sha256", None)
        if owner_checksum is not None and blob_sha is not None and blob_sha != owner_checksum:
            raise ValueError(checksum_error)
        return blob

    # Ordering helpers -------------------------------------------------

    @staticmethod
    def _encode_cursor(parts: Iterable[str]) -> str:
        """Encode a best-effort cursor based on ``created_at`` and object IDs."""
        payload = "|".join(parts)
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
        return encoded.decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str) -> List[str]:
        """Decode cursors produced by :meth:`_encode_cursor`."""
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
            text = decoded.decode("utf-8")
            return text.split("|")
        except Exception as exc:  # pragma: no cover
            raise ValueError("cursor_invalid") from exc

    def _document_entry(
        self, doc: NormalizedDocument
    ) -> Tuple[Tuple[float, str, str], NormalizedDocument]:
        version_key = doc.ref.version or ""
        key = (-doc.created_at.timestamp(), str(doc.ref.document_id), version_key)
        return key, doc

    def _asset_entry(self, asset: Asset) -> Tuple[Tuple[float, str], Asset]:
        key = (-asset.created_at.timestamp(), str(asset.ref.asset_id))
        return key, asset

    def _cursor_start(
        self,
        entries: List[Tuple[Tuple, object]],
        cursor: Optional[str],
    ) -> int:
        if not cursor:
            return 0
        parts = self._decode_cursor(cursor)
        if not parts:
            raise ValueError("cursor_invalid")
        expected_length = len(entries[0][0]) if entries else len(parts)
        if len(parts) != expected_length:
            raise ValueError("cursor_invalid")

        if len(parts) == 3:
            try:
                timestamp = datetime.fromisoformat(parts[0])
                UUID(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("cursor_invalid") from exc
            cursor_key = (-timestamp.timestamp(), parts[1], parts[2])
        elif len(parts) == 2:
            try:
                timestamp = datetime.fromisoformat(parts[0])
                UUID(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("cursor_invalid") from exc
            cursor_key = (-timestamp.timestamp(), parts[1])
        else:  # pragma: no cover - defensive branch
            raise ValueError("cursor_invalid")

        index = 0
        for idx, (key, _) in enumerate(entries):
            if key <= cursor_key:
                index = idx + 1
        return index

    def _next_cursor(
        self,
        entries: List[Tuple[Tuple, object]],
        start: int,
        limit: int,
    ) -> Optional[str]:
        """Return a best-effort pagination cursor for the last returned entry."""
        end = start + limit
        if end >= len(entries):
            return None
        key, obj = entries[end - 1]
        if len(key) == 3:
            doc: NormalizedDocument = obj  # type: ignore[assignment]
            parts = [
                doc.created_at.isoformat(),
                str(doc.ref.document_id),
                doc.ref.version or "",
            ]
        else:
            asset: Asset = obj  # type: ignore[assignment]
            parts = [asset.created_at.isoformat(), str(asset.ref.asset_id)]
        return self._encode_cursor(parts)

    def _iter_documents_locked(
        self, tenant_id: str, collection_id: Optional[UUID]
    ) -> Iterable[NormalizedDocument]:
        for stored in self._documents.values():
            if stored.deleted:
                continue
            doc = stored.value
            if doc.ref.tenant_id != tenant_id:
                continue
            if collection_id is not None and doc.ref.collection_id != collection_id:
                continue
            yield doc

    @staticmethod
    def _newer(left: NormalizedDocument, right: NormalizedDocument) -> bool:
        """Return ``True`` when ``left`` is considered more recent than ``right``.

        ``created_at`` takes precedence; equal timestamps fall back to comparing
        the version strings lexicographically (empty string for ``None``) so tie
        breaks stay deterministic without assuming semantic versioning.
        """
        if left.created_at > right.created_at:
            return True
        if left.created_at < right.created_at:
            return False
        left_version = left.ref.version or ""
        right_version = right.ref.version or ""
        return left_version > right_version

    def _latest_document_locked(
        self, tenant_id: str, document_id: UUID
    ) -> Optional[NormalizedDocument]:
        latest: Optional[NormalizedDocument] = None
        for stored in self._documents.values():
            if stored.deleted:
                continue
            doc = stored.value
            if doc.ref.tenant_id != tenant_id or doc.ref.document_id != document_id:
                continue
            if latest is None or self._newer(doc, latest):
                latest = doc
        return latest


__all__ = [
    "DocumentsRepository",
    "InMemoryDocumentsRepository",
]

