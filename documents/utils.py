"""Utilities for document handling."""

from pathlib import Path
from structlog.stdlib import get_logger
from urllib.parse import quote

logger = get_logger(__name__)


def get_upload_file_path(tenant_id: str, workflow_id: str | None, document_id: str) -> Path:
    """
    Construct path to uploaded file in ObjectStore.

    Returns: .ai_core_store/{tenant}/{workflow}/uploads/{document_id}_upload.bin
    """
    from ai_core.infra import object_store

    tenant_segment = object_store.sanitize_identifier(tenant_id)
    workflow_segment = object_store.sanitize_identifier(workflow_id or "default")
    filename = f"{document_id}_upload.bin"

    return (
        object_store.BASE_PATH
        / tenant_segment
        / workflow_segment
        / "uploads"
        / filename
    )


def detect_content_type(blob, blob_path: Path) -> str:
    """
    Detect Content-Type: blob.media_type → Magic → Fallback.
    """
    # 1. Try blob metadata
    if hasattr(blob, "media_type") and blob.media_type:
        logger.debug("content_type.from_blob", mime=blob.media_type)
        return blob.media_type

    # 2. Fallback: Magic number detection
    try:
        import magic

        mime = magic.from_file(str(blob_path), mime=True)
        logger.info("content_type.magic_detected", mime=mime)
        return mime
    except Exception as e:
        logger.warning("content_type.magic_failed", error=str(e))
        return "application/octet-stream"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename: Remove CRLF, path traversal, control chars."""
    import re

    safe = re.sub(r"[\r\n\x00-\x1f\x7f]", "", filename)
    safe = re.sub(r'[\\/:*?"<>|]', "_", safe)
    return safe[:255] or "download"


def content_disposition(filename: str) -> str:
    """RFC 5987 Content-Disposition with ASCII fallback."""
    ascii_fallback = filename.encode("ascii", "ignore").decode() or "download"
    utf8_encoded = quote(filename)
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{utf8_encoded}"


def get_extension_from_mime(mime_type: str) -> str:
    """Get file extension from MIME type."""
    mime_to_ext = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "text/plain": ".txt",
        "text/html": ".html",
        "text/csv": ".csv",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "application/json": ".json",
        "application/xml": ".xml",
        "application/zip": ".zip",
        "application/x-rar-compressed": ".rar",
        "application/x-7z-compressed": ".7z",
        "application/octet-stream": ".bin",
    }
    return mime_to_ext.get(mime_type, ".bin")


def extract_filename(doc, document_id: str, content_type: str) -> str:
    """
    Extract filename from document with fallbacks.

    Priority:
    1. doc.meta.title (original filename from upload)
    2. doc.meta.external_ref["filename"] (if available)
    3. document_id + extension from content_type
    """
    # 1. Try title field
    if doc.meta.title and doc.meta.title.strip():
        return doc.meta.title.strip()

    # 2. Try external_ref filename
    if doc.meta.external_ref and "filename" in doc.meta.external_ref:
        filename = doc.meta.external_ref.get("filename", "").strip()
        if filename:
            return filename

    # 3. Fallback: document_id + extension from content-type
    ext = get_extension_from_mime(content_type)
    return f"{document_id}{ext}"
