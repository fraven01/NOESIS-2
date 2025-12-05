from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from uuid import UUID

from ai_core.infra import object_store
from documents.contracts import (
    DocumentRef,
    DocumentMeta,
    NormalizedDocument,
    InlineBlob,
)
from documents.repository import DocumentsRepository


class ObjectStoreDocumentsRepository(DocumentsRepository):
    """
    File-backed repository that persists uploads and minimal metadata
    under `.ai_core_store/{tenant}/{workflow}/uploads/`.

    This is intended for development and ingestion workers to share
    a common document space across processes without a DB backend.
    Only `upsert` and `get` are implemented for current ingestion needs.
    """

    def upsert(
        self, doc: NormalizedDocument, workflow_id: Optional[str] = None
    ) -> NormalizedDocument:
        tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
        workflow = workflow_id or doc.ref.workflow_id or "upload"
        workflow_segment = object_store.sanitize_identifier(workflow)
        uploads_prefix = f"{tenant_segment}/{workflow_segment}/uploads"

        payload = _extract_payload(doc)
        checksum = hashlib.sha256(payload).hexdigest()
        filename_base = str(doc.ref.document_id)

        # Persist raw upload
        object_store.write_bytes(
            f"{uploads_prefix}/{filename_base}_upload.bin", payload
        )

        # Persist minimal metadata used by ingestion
        metadata = _build_metadata_snapshot(doc, checksum)
        object_store.write_json(f"{uploads_prefix}/{filename_base}.meta.json", metadata)

        # Return the original model (no mutation)
        return doc

    def get(
        self,
        tenant_id: str,
        document_id: UUID,
        version: Optional[str] = None,
        *,
        prefer_latest: bool = False,
        workflow_id: Optional[str] = None,
    ) -> Optional[NormalizedDocument]:
        tenant_segment = object_store.sanitize_identifier(tenant_id)
        base = object_store.BASE_PATH / tenant_segment

        if not base.exists():
            return None

        candidates: List[Tuple[dict, Path]] = []

        # Walk workflows and collect matching meta files
        for wf_dir in base.iterdir():
            if not wf_dir.is_dir():
                continue
            if workflow_id and wf_dir.name != object_store.sanitize_identifier(
                workflow_id
            ):
                continue
            meta_path = wf_dir / "uploads" / f"{document_id}.meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidates.append((meta, meta_path))

        if not candidates:
            return None

        if version is not None:
            # Select exact version if available
            for meta, path in candidates:
                if str(meta.get("version") or "") == version:
                    return _rebuild_document_from_meta(
                        tenant_id, document_id, meta, path
                    )
            return None

        # Choose latest by created_at in meta, fallback to file mtime
        def _candidate_key(item: Tuple[dict, Path]):
            meta, path = item
            created_at = meta.get("created_at")
            try:
                ts = (
                    datetime.fromisoformat(created_at).timestamp()
                    if isinstance(created_at, str) and created_at
                    else path.stat().st_mtime
                )
            except Exception:
                ts = path.stat().st_mtime
            # Higher timestamp = newer
            return (ts, str(meta.get("version") or ""))

        meta, path = max(candidates, key=_candidate_key)
        return _rebuild_document_from_meta(tenant_id, document_id, meta, path)

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

        tenant_segment = object_store.sanitize_identifier(tenant_id)
        base = object_store.BASE_PATH / tenant_segment

        if not base.exists():
            return [], None

        entries: List[Tuple[Tuple, NormalizedDocument]] = []

        # Walk workflows and collect matching meta files
        for wf_dir in base.iterdir():
            if not wf_dir.is_dir():
                continue
            if workflow_id and wf_dir.name != object_store.sanitize_identifier(
                workflow_id
            ):
                continue

            uploads_dir = wf_dir / "uploads"
            if not uploads_dir.exists():
                continue

            for meta_path in uploads_dir.glob("*.meta.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta_collection_id = meta.get("collection_id")
                    if meta_collection_id and str(meta_collection_id) == str(
                        collection_id
                    ):
                        doc_id_str = meta_path.name.replace(".meta.json", "")
                        try:
                            doc_id = UUID(doc_id_str)
                        except ValueError:
                            continue

                        doc = _rebuild_document_from_meta(
                            tenant_id, doc_id, meta, meta_path
                        )
                        entries.append(self._document_entry(doc))
                except Exception:
                    continue

        entries.sort(key=lambda entry: entry[0])
        start = self._cursor_start(entries, cursor)
        sliced = entries[start : start + limit]
        refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
        next_cursor = self._next_cursor(entries, start, limit)

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

        tenant_segment = object_store.sanitize_identifier(tenant_id)
        base = object_store.BASE_PATH / tenant_segment

        if not base.exists():
            return [], None

        latest: Dict[UUID, NormalizedDocument] = {}

        # Walk workflows and collect matching meta files
        for wf_dir in base.iterdir():
            if not wf_dir.is_dir():
                continue
            if workflow_id and wf_dir.name != object_store.sanitize_identifier(
                workflow_id
            ):
                continue

            uploads_dir = wf_dir / "uploads"
            if not uploads_dir.exists():
                continue

            for meta_path in uploads_dir.glob("*.meta.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta_collection_id = meta.get("collection_id")
                    if meta_collection_id and str(meta_collection_id) == str(
                        collection_id
                    ):
                        doc_id_str = meta_path.name.replace(".meta.json", "")
                        try:
                            doc_id = UUID(doc_id_str)
                        except ValueError:
                            continue

                        doc = _rebuild_document_from_meta(
                            tenant_id, doc_id, meta, meta_path
                        )

                        current = latest.get(doc.ref.document_id)
                        if current is None or self._newer(doc, current):
                            latest[doc.ref.document_id] = doc
                except Exception:
                    continue

        entries = [self._document_entry(doc) for doc in latest.values()]
        entries.sort(key=lambda entry: entry[0])
        start = self._cursor_start(entries, cursor)
        sliced = entries[start : start + limit]
        refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
        next_cursor = self._next_cursor(entries, start, limit)

        return refs, next_cursor

    # Helpers copied from InMemoryDocumentsRepository to support listing

    @staticmethod
    def _encode_cursor(parts: List[str]) -> str:
        payload = "|".join(parts)
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
        return encoded.decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str) -> List[str]:
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
            text = decoded.decode("utf-8")
            return text.split("|")
        except Exception:
            raise ValueError("cursor_invalid")

    def _document_entry(
        self, doc: NormalizedDocument
    ) -> Tuple[Tuple[float, str, str, str], NormalizedDocument]:
        version_key = doc.ref.version or ""
        # Handle potential None created_at (though NormalizedDocument usually has it)
        ts = doc.created_at.timestamp() if doc.created_at else 0.0
        key = (
            -ts,
            str(doc.ref.document_id),
            doc.ref.workflow_id or "",
            version_key,
        )
        return key, doc

    def _cursor_start(
        self,
        entries: List[Tuple[Tuple, object]],
        cursor: Optional[str],
    ) -> int:
        if not cursor:
            return 0
        try:
            parts = self._decode_cursor(cursor)
        except ValueError:
            return 0

        if not parts:
            return 0

        # Best-effort matching based on parts length
        cursor_key: Tuple = ()
        try:
            if len(parts) == 4:
                timestamp = datetime.fromisoformat(parts[0])
                cursor_key = (-timestamp.timestamp(), parts[1], parts[2], parts[3])
            elif len(parts) == 3:
                timestamp = datetime.fromisoformat(parts[0])
                cursor_key = (-timestamp.timestamp(), parts[1], parts[2])
            elif len(parts) == 2:
                timestamp = datetime.fromisoformat(parts[0])
                cursor_key = (-timestamp.timestamp(), parts[1])
        except (ValueError, TypeError):
            return 0

        if not cursor_key:
            return 0

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
        end = start + limit
        if end >= len(entries):
            return None
        key, obj = entries[end - 1]

        doc: NormalizedDocument = obj  # type: ignore
        parts = [
            doc.created_at.isoformat() if doc.created_at else datetime.min.isoformat(),
            str(doc.ref.document_id),
            doc.ref.workflow_id or "",
            doc.ref.version or "",
        ]
        return self._encode_cursor(parts)

    @staticmethod
    def _newer(left: NormalizedDocument, right: NormalizedDocument) -> bool:
        if left.created_at and right.created_at:
            if left.created_at > right.created_at:
                return True
            if left.created_at < right.created_at:
                return False
        left_version = left.ref.version or ""
        right_version = right.ref.version or ""
        return left_version > right_version


def _extract_payload(doc: NormalizedDocument) -> bytes:
    blob = doc.blob
    # Inline preferred for uploads
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()
    base64_value = getattr(blob, "base64", None)
    if isinstance(base64_value, str):
        return base64.b64decode(base64_value)
    # As a fallback, try a payload() accessor if present
    if hasattr(blob, "decoded_payload"):
        try:
            return blob.decoded_payload()
        except Exception:
            pass
    raise TypeError("unsupported_blob_type")


def _build_metadata_snapshot(doc: NormalizedDocument, checksum: str) -> dict:
    payload: dict = {
        "workflow_id": doc.ref.workflow_id,
        "document_id": str(doc.ref.document_id),
        "created_at": (doc.created_at.isoformat() if doc.created_at else None),
        "checksum": checksum,
        "media_type": getattr(doc.blob, "media_type", None),
        "source": doc.source,
    }

    if doc.ref.collection_id is not None:
        payload["collection_id"] = str(doc.ref.collection_id)
    if doc.ref.version is not None:
        payload["version"] = doc.ref.version

    # External metadata frequently used downstream
    external_ref = getattr(doc.meta, "external_ref", None) or {}
    if isinstance(external_ref, dict):
        external_id = external_ref.get("external_id")
        if external_id:
            payload["external_id"] = external_id

    title = getattr(doc.meta, "title", None)
    if title:
        payload["title"] = title
    language = getattr(doc.meta, "language", None)
    if language:
        payload["language"] = language
    pipeline_config = getattr(doc.meta, "pipeline_config", None)
    if isinstance(pipeline_config, dict):
        payload["pipeline_config"] = dict(pipeline_config)

    return payload


def _rebuild_document_from_meta(
    tenant_id: str, document_id: UUID, meta: dict, meta_path: Path
) -> NormalizedDocument:
    # Inputs
    workflow_id = str(meta.get("workflow_id") or "upload")
    collection_id = meta.get("collection_id")
    version = meta.get("version")
    created_at_raw = meta.get("created_at")
    media_type = meta.get("media_type")
    checksum = str(meta.get("checksum") or "")
    external_id = meta.get("external_id")

    # Blob
    upload_path = meta_path.with_name(f"{document_id}_upload.bin")
    payload = upload_path.read_bytes()
    computed_sha = hashlib.sha256(payload).hexdigest()
    # Prefer recorded checksum if present and correct; otherwise use computed
    sha = checksum if checksum and checksum == computed_sha else computed_sha
    inline_blob = InlineBlob(
        type="inline",
        media_type=media_type or "application/octet-stream",
        base64=base64.b64encode(payload).decode("ascii"),
        sha256=sha,
        size=len(payload),
    )

    # References & meta
    doc_ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=UUID(collection_id) if _maybe_uuid(collection_id) else None,
        version=str(version) if version is not None else None,
    )

    ext_ref = {}
    if external_id:
        ext_ref["external_id"] = str(external_id)
    if media_type:
        ext_ref["media_type"] = str(media_type)

    pipeline_config = meta.get("pipeline_config")
    doc_meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        external_ref=ext_ref or None,
        title=str(meta.get("title")) if meta.get("title") else None,
        language=str(meta.get("language")) if meta.get("language") else None,
        pipeline_config=pipeline_config if isinstance(pipeline_config, dict) else None,
    )

    created_at = None
    if isinstance(created_at_raw, str):
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError:
            created_at = None

    if created_at is None:
        created_at = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc)

    return NormalizedDocument(
        ref=doc_ref,
        meta=doc_meta,
        blob=inline_blob,
        checksum=sha,
        created_at=created_at,
        source=str(meta.get("source") or "upload"),
    )


def _maybe_uuid(value: object) -> bool:
    try:
        return bool(value) and UUID(str(value)) is not None
    except Exception:
        return False


__all__ = ["ObjectStoreDocumentsRepository"]
