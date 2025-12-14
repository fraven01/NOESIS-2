"""Normalization helpers for document payload access and text processing."""

from __future__ import annotations

from typing import Any, Optional


def document_payload_bytes(document: Any, storage: Optional[Any] = None) -> bytes:
    """Decode payload from any blob type.

    Supports InlineBlob (embedded), FileBlob (storage), LocalFileBlob (filesystem),
    ExternalBlob (external storage), and duck-typed blobs with decoded_payload().

    Args:
        document: NormalizedDocument or dict containing blob locator
        storage: Optional storage service for FileBlob/ExternalBlob retrieval.
                 Required for FileBlob and ExternalBlob, ignored for InlineBlob.

    Returns:
        Decoded bytes from blob payload

    Raises:
        ValueError: If blob type is unsupported or storage is missing when required
    """
    from documents.contracts import InlineBlob, FileBlob, ExternalBlob, LocalFileBlob

    # Support both dict and object access patterns
    if isinstance(document, dict):
        blob = document.get("blob")
    else:
        blob = getattr(document, "blob", None)

    if blob is None:
        raise ValueError("blob_missing: document has no blob field")

    # InlineBlob: payload embedded in base64
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()

    # LocalFileBlob: payload in local filesystem
    if isinstance(blob, LocalFileBlob):
        with open(blob.path, "rb") as f:
            return f.read()

    # FileBlob: payload in object storage
    if isinstance(blob, FileBlob):
        if storage is None:
            raise ValueError(
                f"storage_required_for_file_blob: "
                f"FileBlob with uri='{blob.uri}' requires storage service parameter"
            )
        return storage.get(blob.uri)

    # ExternalBlob: payload in external storage (S3, GCS, HTTP)
    if isinstance(blob, ExternalBlob):
        if storage is None:
            raise ValueError(
                f"storage_required_for_external_blob: "
                f"ExternalBlob with kind='{blob.kind}' uri='{blob.uri}' requires storage service parameter"
            )
        return storage.get(blob.uri)

    # Duck-typed fallback: any object with decoded_payload() method (for testing)
    if callable(getattr(blob, "decoded_payload", None)):
        return blob.decoded_payload()

    # Unknown blob type
    raise ValueError(
        f"unsupported_blob_type: {type(blob).__name__} "
        f"(expected InlineBlob, LocalFileBlob, FileBlob, ExternalBlob, or duck-typed with decoded_payload())"
    )


def normalized_primary_text(text: Optional[str]) -> str:
    """Return whitespace-normalised primary text or an empty string."""

    raw = (text or "").strip()
    if not raw:
        return ""
    return " ".join(raw.split())


__all__ = [
    "document_payload_bytes",
    "normalized_primary_text",
]
