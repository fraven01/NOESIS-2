"""Normalization contract mapping parser output onto document metadata."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from documents.contract_utils import normalize_string
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument as ContractsNormalizedDocument,
)
from documents.parsers import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
)
from documents.providers import (
    ProviderReference,
    build_external_reference,
    parse_provider_reference,
)

from .contracts import NormalizedSource

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .fetcher import FetchResult


def build_normalized_document(
    *,
    parse_result: ParseResult,
    source: NormalizedSource,
    tenant_id: str,
    workflow_id: str,
    document_id: str,
    tags: Optional[Sequence[str]] = None,
) -> ContractsNormalizedDocument:
    """Compose a :class:`documents.contracts.NormalizedDocument` instance."""

    if parse_result.status is not ParseStatus.PARSED:
        raise ValueError("parse_result_not_parsed")
    content = _require_content(parse_result.content)
    stats = _require_stats(parse_result.stats)
    request_source = parse_result.fetch.request.canonical_source
    canonical_source = _require_identifier(source.canonical_source, "canonical_source")
    if (
        _require_identifier(request_source, "fetch_canonical_source")
        != canonical_source
    ):
        raise ValueError("canonical_source_mismatch")

    normalized_tags = tuple(tags) if tags else ()
    parser_stats = dict(_parser_stats_mapping(stats))
    normalized_text = normalized_primary_text(content.primary_text)
    if normalized_text:
        hash_value = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if _HASH_HEX_RE.fullmatch(hash_value):
            parser_stats["crawler.primary_text_hash_sha256"] = hash_value
    bytes_in = _normalizer_bytes_in(parse_result.fetch)
    if bytes_in is not None:
        parser_stats["normalizer.bytes_in"] = bytes_in

    meta = DocumentMeta(
        tenant_id=_require_identifier(tenant_id, "tenant_id"),
        workflow_id=_require_identifier(workflow_id, "workflow_id"),
        title=content.title,
        language=content.content_language,
        tags=list(normalized_tags),
        origin_uri=canonical_source,
        parse_stats=parser_stats,
        external_ref=build_external_reference(
            provider=source.provider,
            external_id=source.external_id,
            provider_tags=source.provider_tags,
        ),
    )

    payload_bytes = parse_result.fetch.payload
    if payload_bytes is None and content.primary_text is not None:
        payload_bytes = content.primary_text.encode("utf-8")
    if payload_bytes is None:
        raise ValueError("normalizer_payload_missing")
    blob = _build_inline_blob(content.media_type, payload_bytes)

    document_ref = DocumentRef(
        tenant_id=meta.tenant_id,
        workflow_id=meta.workflow_id,
        document_id=document_id,
    )

    normalized = ContractsNormalizedDocument(
        ref=document_ref,
        meta=meta,
        blob=blob,
        checksum=blob.sha256,
        created_at=datetime.now(timezone.utc),
        source="crawler",
    )

    return normalized


def resolve_provider_reference(
    document: ContractsNormalizedDocument,
) -> ProviderReference:
    """Return the provider reference derived from the document metadata."""

    return parse_provider_reference(document.meta)


def document_parser_stats(
    document: ContractsNormalizedDocument,
) -> Mapping[str, object]:
    """Expose parser statistics as an immutable mapping."""

    return MappingProxyType(dict(document.meta.parse_stats or {}))


def document_payload_bytes(document: ContractsNormalizedDocument) -> bytes:
    """Decode the inline payload embedded in the normalized document."""

    blob = document.blob
    if isinstance(blob, InlineBlob):
        return blob.decoded_payload()
    raise ValueError("unsupported_blob_type")


def normalized_primary_text(text: Optional[str]) -> str:
    """Return whitespace-normalized primary text or an empty string."""

    raw = (text or "").strip()
    if not raw:
        return ""
    return " ".join(raw.split())


_HASH_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_identifier(value: Optional[str], field: str) -> str:
    candidate = normalize_string(value or "")
    if not candidate:
        raise ValueError(f"{field}_required")
    return candidate


def _require_content(content: Optional[ParserContent]) -> ParserContent:
    if content is None:
        raise ValueError("parser_content_missing")
    if not isinstance(content, ParserContent):
        raise TypeError("parser_content_invalid")
    return content


def _require_stats(stats: Optional[ParserStats]) -> ParserStats:
    if stats is None:
        raise ValueError("parser_stats_missing")
    if not isinstance(stats, ParserStats):
        raise TypeError("parser_stats_invalid")
    return stats


def _parser_stats_mapping(stats: ParserStats) -> Mapping[str, object]:
    data: dict[str, object] = {
        "parser.token_count": stats.token_count,
        "parser.character_count": stats.character_count,
        "parser.error_fraction": stats.error_fraction,
        "parser.extraction_path": stats.extraction_path,
    }
    if stats.warnings:
        data["parser.warnings"] = list(stats.warnings)
    if stats.boilerplate_reduction is not None:
        data["parser.boilerplate_reduction"] = stats.boilerplate_reduction
    return data


def _normalizer_bytes_in(fetch_result: "FetchResult") -> Optional[int]:
    payload = fetch_result.payload
    if payload is not None:
        return len(payload)
    bytes_downloaded = fetch_result.telemetry.bytes_downloaded
    return bytes_downloaded if bytes_downloaded > 0 else None


def _build_inline_blob(media_type: str, payload: bytes) -> InlineBlob:
    encoded = base64.b64encode(payload).decode("ascii")
    checksum = hashlib.sha256(payload).hexdigest()
    return InlineBlob(
        type="inline",
        media_type=media_type,
        base64=encoded,
        sha256=checksum,
        size=len(payload),
    )
