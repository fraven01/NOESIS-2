from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from uuid import UUID, uuid4

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
        filename_base = f"{doc.ref.document_id.hex}"

        # Persist raw upload
        object_store.write_bytes(
            f"{uploads_prefix}/{filename_base}_upload.bin", payload
        )

        # Persist minimal metadata used by ingestion
        metadata = _build_metadata_snapshot(doc, checksum)
        object_store.write_json(
            f"{uploads_prefix}/{filename_base}.meta.json", metadata
        )

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
            meta_path = wf_dir / "uploads" / f"{document_id.hex}.meta.json"
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
                    return _rebuild_document_from_meta(tenant_id, document_id, meta, path)
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
        "document_id": doc.ref.document_id.hex,
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
    upload_path = meta_path.with_name(f"{document_id.hex}_upload.bin")
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

    doc_meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        external_ref=ext_ref or None,
        title=str(meta.get("title")) if meta.get("title") else None,
        language=str(meta.get("language")) if meta.get("language") else None,
    )

    created_at = None
    if isinstance(created_at_raw, str):
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError:
            created_at = None

    return NormalizedDocument(
        ref=doc_ref,
        meta=doc_meta,
        blob=inline_blob,
        checksum=sha,
        created_at=created_at or datetime.fromtimestamp(meta_path.stat().st_mtime),
        source=str(meta.get("source") or "upload"),
    )


def _maybe_uuid(value: object) -> bool:
    try:
        return bool(value) and UUID(str(value)) is not None
    except Exception:
        return False


__all__ = ["ObjectStoreDocumentsRepository"]

