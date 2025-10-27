"""Normalization contract mapping parser output onto document metadata."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument as ContractsNormalizedDocument,
)
from .contracts import NormalizedSource
from .parser import ParseResult, ParseStatus, ParserContent, ParserStats

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .fetcher import FetchResult


@dataclass(frozen=True)
class ProviderReference:
    """Adapter exposing crawler provider metadata from document contracts."""

    provider: str
    external_id: str
    canonical_source: str
    provider_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _require_identifier(self.provider, "provider")
        external_id = _require_identifier(self.external_id, "external_id")
        canonical_source = _require_identifier(
            self.canonical_source, "canonical_source"
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "external_id", external_id)
        object.__setattr__(self, "canonical_source", canonical_source)
        normalized = _normalize_str_mapping(self.provider_tags)
        object.__setattr__(self, "provider_tags", normalized)


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

    normalized_tags = _normalize_tags(tags)
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
        external_ref=_build_external_ref(source, canonical_source),
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
        document_id=_coerce_uuid(document_id),
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

    return _parse_external_ref(document.meta)


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


def normalize_diagnostics(entries: Iterable[str]) -> Tuple[str, ...]:
    """Normalize diagnostic messages by stripping whitespace and dropping empties."""

    return _normalize_diagnostics(tuple(entries))


def _require_identifier(value: Optional[str], field: str) -> str:
    candidate = (value or "").strip()
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


def _normalize_tags(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    seen = set()
    for raw in values:
        text = (raw or "").strip()
        if not text:
            continue
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def _normalize_str_mapping(mapping: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    if not mapping:
        return MappingProxyType({})
    normalized: dict[str, str] = {}
    for key, value in mapping.items():
        normalized_key = _require_identifier(str(key), "provider_tag_key")
        normalized_val = _require_identifier(str(value), "provider_tag_value")
        normalized[normalized_key] = normalized_val
    return MappingProxyType(normalized)
def _normalize_diagnostics(entries: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        normalized
        for normalized in ((entry or "").strip() for entry in entries)
        if normalized
    )


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


def _build_external_ref(
    source: NormalizedSource, canonical_source: str
) -> Mapping[str, str]:
    data: dict[str, str] = {
        "provider": _sanitize_external_value(source.provider, limit=128),
        "external_id": _sanitize_external_value(source.external_id),
    }
    for key, value in (source.provider_tags or {}).items():
        normalized_key = _sanitize_tag_key(str(key))
        normalized_val = _sanitize_external_value(str(value))
        data[f"provider_tag:{normalized_key}"] = normalized_val
    return data


def _parse_external_ref(meta: DocumentMeta) -> ProviderReference:
    external = meta.external_ref or {}
    provider = external.get("provider", "")
    external_id = external.get("external_id", "")
    canonical_source = meta.origin_uri or ""
    tags = {
        key.split(":", 1)[1]: value
        for key, value in external.items()
        if key.startswith("provider_tag:")
    }
    return ProviderReference(
        provider=provider,
        external_id=external_id,
        canonical_source=canonical_source,
        provider_tags=tags,
    )


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


def _coerce_uuid(value: str) -> UUID:
    try:
        return UUID(_require_identifier(value, "document_id"))
    except (ValueError, AttributeError) as exc:  # pragma: no cover - invalid uuid
        raise ValueError("document_id_invalid") from exc


def _sanitize_external_value(value: str, *, limit: int = 512) -> str:
    normalized = _require_identifier(value, "external_ref_value")
    if len(normalized) > limit:
        return normalized[:limit]
    return normalized


def _sanitize_tag_key(key: str) -> str:
    normalized = _require_identifier(key, "provider_tag_key")
    max_length = 128 - len("provider_tag:")
    if len(normalized) > max_length:
        return normalized[:max_length]
    return normalized
