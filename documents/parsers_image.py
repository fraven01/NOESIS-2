"""Parser for image documents returning raw bytes as assets."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from documents.contract_utils import normalize_media_type
from documents.parsers import (
    ParsedResult,
    build_parsed_asset_with_meta,
    build_parsed_result,
    build_parsed_text_block,
)


_SUPPORTED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/tiff",
    "image/bmp",
}


def _normalise_media_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return normalize_media_type(value)
    except ValueError:
        return None


def _extract_candidate_media_type(source: Any) -> Optional[str]:
    media_type = getattr(source, "media_type", None)
    if media_type is None and isinstance(source, Mapping):
        media_type = source.get("media_type")
    return _normalise_media_type(media_type)


def _extract_media_type(document: Any) -> Optional[str]:
    return _extract_candidate_media_type(document)


def _extract_blob(document: Any) -> Any:
    blob = getattr(document, "blob", None)
    if blob is None and isinstance(document, Mapping):
        blob = document.get("blob")
    return blob


def _extract_blob_media_type(blob: Any) -> Optional[str]:
    return _extract_candidate_media_type(blob)


def _blob_payload(blob: Any) -> bytes:
    if blob is None:
        return b""
    if hasattr(blob, "decoded_payload"):
        payload = blob.decoded_payload()
        if isinstance(payload, bytes):
            return payload
    data = getattr(blob, "content", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(blob, Mapping):
        inline = blob.get("content") or blob.get("payload")
        if isinstance(inline, (bytes, bytearray)):
            return bytes(inline)
        base64_value = blob.get("base64")
        if isinstance(base64_value, (bytes, bytearray)):
            return bytes(base64_value)
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return b""


class ImageDocumentParser:
    """Document parser for common raster image formats."""

    def _resolve_media_type(self, document: Any) -> Optional[str]:
        media_type = _extract_media_type(document)
        if media_type:
            return media_type
        blob = _extract_blob(document)
        return _extract_blob_media_type(blob)

    def can_handle(self, document: Any) -> bool:
        media_type = self._resolve_media_type(document)
        return bool(media_type and media_type in _SUPPORTED_MEDIA_TYPES)

    def parse(self, document: Any, config: Any) -> ParsedResult:  # noqa: ARG002
        blob = _extract_blob(document)
        media_type = self._resolve_media_type(document) or "application/octet-stream"
        payload = _blob_payload(blob)
        asset = build_parsed_asset_with_meta(
            media_type=media_type,
            content=payload,
            metadata={"asset_kind": "document_content", "locator": "image-body"},
        )
        placeholder_text = build_parsed_text_block(text="[Bilddatei]", kind="other")
        statistics = {
            "parser.kind": "image",
            "parser.bytes": len(payload),
            "parser.assets": 1,
        }
        return build_parsed_result(
            text_blocks=(placeholder_text,), assets=(asset,), statistics=statistics
        )


__all__ = ["ImageDocumentParser"]
