"""Normalization contract mapping parser output onto document metadata."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import UUID

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
    normalize_diagnostics,
)
from documents.providers import (
    ProviderReference,
    build_external_reference,
    parse_provider_reference,
)

from .contracts import NormalizedSource


MAX_TAG_LENGTH = 64
_PROVIDER_TAG_PREFIX = "provider."

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .fetcher import FetchResult


def build_normalized_document(
    *,
    parse_result: ParseResult,
    source: NormalizedSource,
    tenant_id: str,
    workflow_id: str,
    document_id: UUID | str,
    tags: Optional[Sequence[str]] = None,
) -> ContractsNormalizedDocument:
    """Compose a :class:`documents.contracts.NormalizedDocument` instance."""

    if parse_result.status is not ParseStatus.PARSED:
        raise ValueError("parse_result_not_parsed")
    content = _require_content(parse_result.content)
    stats = _require_stats(parse_result.stats)
    request_source = parse_result.fetch.request.canonical_source
    canonical_source = _require_identifier(source.canonical_source, "canonical_source")
    if _require_identifier(request_source, "fetch_canonical_source") != canonical_source:
        raise ValueError("canonical_source_mismatch")

    normalized_tags = tuple(tags) if tags else ()
    document_tags = _compose_document_tags(
        normalized_tags,
        source.provider_tags,
        parse_result.fetch,
    )
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
        tags=list(document_tags),
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
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def _compose_document_tags(
    base_tags: Sequence[str],
    provider_tags: Mapping[str, str],
    fetch_result: "FetchResult",
) -> Tuple[str, ...]:
    tags: list[str] = list(base_tags)
    tags.extend(_provider_tag_aliases(provider_tags))
    tags.extend(_robots_tags(fetch_result))
    return tuple(tags)


def _provider_tag_aliases(provider_tags: Mapping[str, str]) -> Tuple[str, ...]:
    aliases: list[str] = []
    for raw_key, raw_value in provider_tags.items():
        key = _sanitize_tag_component(raw_key)
        value = _sanitize_tag_component(raw_value)
        if not (key and value):
            continue
        alias = f"{_PROVIDER_TAG_PREFIX}{key}.{value}"
        if len(alias) > MAX_TAG_LENGTH:
            max_value_length = MAX_TAG_LENGTH - len(_PROVIDER_TAG_PREFIX) - len(key) - 1
            if max_value_length <= 0:
                continue
            truncated_value = value[:max_value_length]
            truncated_value = _sanitize_tag_component(truncated_value)
            if not truncated_value:
                continue
            alias = f"{_PROVIDER_TAG_PREFIX}{key}.{truncated_value}"
            if len(alias) > MAX_TAG_LENGTH:
                continue
        aliases.append(alias)
    return tuple(aliases)


def _robots_tags(fetch_result: "FetchResult") -> Tuple[str, ...]:
    tags: list[str] = []
    for event in getattr(fetch_result, "policy_events", ()):
        if event.startswith("robots_"):
            suffix = event[len("robots_") :]
            suffix_tag = _sanitize_tag_component(suffix)
            if suffix_tag:
                tags.append(f"robots.{suffix_tag}")
    request_metadata = getattr(fetch_result.request, "metadata", {})
    if isinstance(request_metadata, Mapping):
        tags.extend(_robots_metadata_tags(request_metadata.get("robots")))
    return tuple(tags)


def _robots_metadata_tags(raw_value: Optional[object]) -> Tuple[str, ...]:
    if raw_value is None:
        return ()
    tags: list[str] = []
    if isinstance(raw_value, Mapping):
        for raw_key, raw_hint in raw_value.items():
            key = _sanitize_tag_component(raw_key)
            hint = _sanitize_tag_component(raw_hint)
            if key and hint:
                tags.append(f"robots.{key}.{hint}")
            elif key:
                tags.append(f"robots.{key}")
    elif isinstance(raw_value, str):
        hint = _sanitize_tag_component(raw_value)
        if hint:
            tags.append(f"robots.{hint}")
    elif isinstance(raw_value, Iterable) and not isinstance(raw_value, (bytes, bytearray)):
        for entry in raw_value:
            hint = _sanitize_tag_component(entry)
            if hint:
                tags.append(f"robots.{hint}")
    else:
        hint = _sanitize_tag_component(raw_value)
        if hint:
            tags.append(f"robots.{hint}")
    return tuple(tags)


def _sanitize_tag_component(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = normalize_string(str(value))
    if not normalized:
        return None
    sanitized = _TAG_SANITIZE_RE.sub("-", normalized.lower())
    sanitized = sanitized.strip("-._")
    if not sanitized:
        return None
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized


