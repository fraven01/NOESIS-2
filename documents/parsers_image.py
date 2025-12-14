"""Parser for image documents returning raw bytes as assets."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from documents.contract_utils import normalize_media_type
from documents.normalization import document_payload_bytes
from documents.parsers import (
    DocumentParser,
    ParsedAsset,
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


class ImageDocumentParser(DocumentParser):
    """Document parser for common raster image formats."""

    def _resolve_media_type(self, document: Any) -> Optional[str]:
        # Check document media type
        media_type = getattr(document, "media_type", None)
        if isinstance(media_type, str):
            return _normalise_media_type(media_type)
        if isinstance(document, Mapping):
             media_type = document.get("media_type")
             if isinstance(media_type, str):
                 return _normalise_media_type(media_type)

        # Check blob media type
        blob = getattr(document, "blob", None)
        if blob is None and isinstance(document, Mapping):
            blob = document.get("blob")
            
        media_type = getattr(blob, "media_type", None)
        if isinstance(media_type, str):
            return _normalise_media_type(media_type)
        if isinstance(blob, Mapping):
            media_type = blob.get("media_type")
            if isinstance(media_type, str):
                 return _normalise_media_type(media_type)
                 
        return None

    def can_handle(self, document: Any) -> bool:
        media_type = self._resolve_media_type(document)
        return bool(media_type and media_type in _SUPPORTED_MEDIA_TYPES)

    def parse(self, document: Any, config: Any) -> ParsedResult:  # noqa: ARG002
        try:
            payload = document_payload_bytes(document)
        except ValueError as exc:
             raise ValueError("image_blob_missing") from exc
             
        if not payload:
             # If payload is empty but passed can_handle, it might be an issue.
             # However, image parser expects bytes.
             raise ValueError("image_blob_missing")
             
        media_type = self._resolve_media_type(document) or "application/octet-stream"
        
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
