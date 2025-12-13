"""Normalization helpers for document payload access and text processing."""

from __future__ import annotations

from typing import Any, Optional


def document_payload_bytes(
    document: Any, storage: Optional[Any] = None
) -> bytes:
    """Decode payload from any blob type.

    Supports InlineBlob (embedded), FileBlob (storage), LocalFileBlob (filesystem),
    and ExternalBlob (external storage).

    Args:
        document: NormalizedDocument containing blob locator
        storage: Optional storage service for FileBlob/ExternalBlob retrieval.
                 Required for FileBlob and ExternalBlob, ignored for InlineBlob.

    Returns:
        Decoded bytes from blob payload

    Raises:
        ValueError: If blob type is unsupported or storage is missing when required
    """
    from documents.contracts import InlineBlob, FileBlob, ExternalBlob, LocalFileBlob

    blob = document.blob

    # InlineBlob: payload embedded in base64
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()

    # LocalFileBlob: payload in local filesystem
    elif isinstance(blob, LocalFileBlob):
        with open(blob.path, "rb") as f:
            return f.read()

    # FileBlob: payload in object storage
    elif isinstance(blob, FileBlob):
        if storage is None:
            raise ValueError(
                f"storage_required_for_file_blob: "
                f"FileBlob with uri='{blob.uri}' requires storage service parameter"
            )
        return storage.get(blob.uri)

    # ExternalBlob: payload in external storage (S3, GCS, HTTP)
    elif isinstance(blob, ExternalBlob):
        if storage is None:
            raise ValueError(
                f"storage_required_for_external_blob: "
                f"ExternalBlob with kind='{blob.kind}' uri='{blob.uri}' requires storage service parameter"
            )
        return storage.get(blob.uri)

    # Unknown blob type
    else:
        raise ValueError(
            f"unsupported_blob_type: {type(blob).__name__} "
            f"(expected InlineBlob, LocalFileBlob, FileBlob, or ExternalBlob)"
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
