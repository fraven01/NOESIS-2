"""Plain text parser used as fallback for ingestion."""

from __future__ import annotations

import re
from typing import List, Mapping, Optional

from documents.parsers import (
    ParsedResult,
    ParsedTextBlock,
    build_parsed_result,
    build_parsed_text_block,
)
from documents.payloads import extract_payload


class TextDocumentParser:
    """Fallback parser decoding UTF-8 payloads for plain text blobs."""

    _SUPPORTED_MEDIA_TYPE = "text/plain"

    @staticmethod
    def _normalized_media_type(value: object) -> Optional[str]:
        if not isinstance(value, str):
            return None
        candidate = value.split(";")[0].strip().lower()
        return candidate or None

    def _infer_media_type(self, document: object) -> Optional[str]:
        blob = getattr(document, "blob", None)
        media_type = self._normalized_media_type(getattr(blob, "media_type", None))
        if media_type:
            return media_type
        meta = getattr(document, "meta", None)
        external_ref = getattr(meta, "external_ref", None)
        if isinstance(external_ref, Mapping):
            media_type = self._normalized_media_type(external_ref.get("media_type"))
            if media_type:
                return media_type
        candidate = self._normalized_media_type(getattr(document, "media_type", None))
        if candidate:
            return candidate
        return None

    def can_handle(self, document: object) -> bool:
        media_type = self._infer_media_type(document)
        if media_type:
            return media_type == self._SUPPORTED_MEDIA_TYPE
        blob = getattr(document, "blob", None)
        blob_media = self._normalized_media_type(getattr(blob, "media_type", None))
        if blob_media:
            return blob_media == self._SUPPORTED_MEDIA_TYPE
        # Fall back to payload sniffing for small inline text blobs.
        try:
            payload = extract_payload(blob)
        except Exception:  # pragma: no cover - guard against storage errors
            return False
        if not payload:
            return False
        # Heuristic: treat as text if it decodes cleanly as UTF-8.
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    def parse(self, document: object, config: object) -> ParsedResult:  # noqa: ARG002
        blob = getattr(document, "blob", None)
        encoding = None
        if hasattr(blob, "content_encoding"):
            candidate = getattr(blob, "content_encoding")
            if isinstance(candidate, str):
                encoding = candidate
        payload = extract_payload(blob, content_encoding=encoding)
        text = ""
        if payload:
            text = payload.decode("utf-8", errors="replace")
        text = text.lstrip("\ufeff")
        normalised = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = [
            part.strip() for part in re.split(r"\n\s*\n", normalised) if part.strip()
        ]
        blocks: List[ParsedTextBlock] = []
        if parts:
            if len(parts) == 1:
                blocks.append(
                    build_parsed_text_block(text=parts[0], kind="paragraph")
                )
            else:
                blocks.extend(
                    build_parsed_text_block(text=part, kind="paragraph")
                    for part in parts
                )
        elif normalised.strip():
            blocks.append(
                build_parsed_text_block(text=normalised.strip(), kind="paragraph")
            )
        statistics = {
            "parser.kind": "text/plain",
            "parser.characters": len(normalised),
        }
        return build_parsed_result(text_blocks=tuple(blocks), statistics=statistics)


__all__ = ["TextDocumentParser"]
