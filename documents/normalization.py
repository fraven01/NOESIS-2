"""Normalization helpers bridging parser output to stored documents."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from documents.contract_utils import normalize_string
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.parsers import ParseResult, ParseStatus, ParserContent, ParserStats
from documents.providers import build_external_reference, parse_provider_reference


MAX_TAG_LENGTH = 64
_PROVIDER_TAG_PREFIX = "provider."
_HASH_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_from_parse(
    *,
    parse_result: ParseResult,
    tenant_id: str,
    workflow_id: str,
    document_id: UUID | str,
    canonical_source: str,
    provider: str,
    external_id: str,
    provider_tags: Optional[Mapping[str, str]] = None,
    tags: Optional[Sequence[str]] = None,
    fetch_result: Optional[Any] = None,
    ingest_source: Optional[str] = "crawler",
    created_at: Optional[datetime] = None,
) -> NormalizedDocument:
    """Return a :class:`NormalizedDocument` from parsed content and fetch metadata."""

    if parse_result.status is not ParseStatus.PARSED:
        raise ValueError("parse_result_not_parsed")

    content = _require_content(parse_result.content)
    stats = _require_stats(parse_result.stats)
    fetch = fetch_result or parse_result.fetch
    if fetch is None:
        raise ValueError("fetch_result_missing")

    request_source = getattr(getattr(fetch, "request", None), "canonical_source", None)
    canonical = _require_identifier(canonical_source, "canonical_source")
    if _require_identifier(request_source, "fetch_canonical_source") != canonical:
        raise ValueError("canonical_source_mismatch")

    normalized_tags = tuple(tags) if tags else ()
    composed_tags = _compose_document_tags(
        normalized_tags,
        provider_tags or {},
        fetch,
    )
    parser_stats = dict(_parser_stats_mapping(stats))

    normalized_text = normalized_primary_text(content.primary_text)
    if normalized_text:
        hash_value = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if _HASH_HEX_RE.fullmatch(hash_value):
            parser_stats["crawler.primary_text_hash_sha256"] = hash_value

    bytes_in = _normalizer_bytes_in(fetch)
    if bytes_in is not None:
        parser_stats["normalizer.bytes_in"] = bytes_in

    meta = DocumentMeta(
        tenant_id=_require_identifier(tenant_id, "tenant_id"),
        workflow_id=_require_identifier(workflow_id, "workflow_id"),
        title=content.title,
        language=content.content_language,
        tags=list(composed_tags),
        origin_uri=canonical,
        parse_stats=parser_stats,
        external_ref=build_external_reference(
            provider=_require_identifier(provider, "provider"),
            external_id=_require_identifier(external_id, "external_id"),
            provider_tags=provider_tags or {},
        ),
    )

    payload_bytes = getattr(fetch, "payload", None)
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

    created_ts = (
        created_at.astimezone(timezone.utc)
        if created_at is not None
        else datetime.now(timezone.utc)
    )

    return NormalizedDocument(
        ref=document_ref,
        meta=meta,
        blob=blob,
        checksum=blob.sha256,
        created_at=created_ts,
        source=ingest_source,
        lifecycle_state="active",
    )


def resolve_provider_reference(document: NormalizedDocument):
    """Return the provider reference derived from normalized document metadata."""

    return parse_provider_reference(document.meta)


def document_parser_stats(document: NormalizedDocument) -> Mapping[str, object]:
    """Expose parser statistics from a :class:`NormalizedDocument`."""

    stats = getattr(document.meta, "parse_stats", {}) or {}
    if not isinstance(stats, Mapping):
        return {}
    return dict(stats)


def document_payload_bytes(
    document: NormalizedDocument, storage: Optional[Any] = None
) -> bytes:
    """Decode payload from any blob type.

    Supports InlineBlob (embedded), FileBlob (storage), and ExternalBlob (external storage).

    Args:
        document: NormalizedDocument containing blob locator
        storage: Optional storage service for FileBlob/ExternalBlob retrieval.
                 Required for FileBlob and ExternalBlob, ignored for InlineBlob.

    Returns:
        Decoded bytes from blob payload

    Raises:
        ValueError: If blob type is unsupported or storage is missing when required

    Example:
        # InlineBlob - no storage needed
        payload = document_payload_bytes(doc_with_inline)

        # FileBlob - storage required
        payload = document_payload_bytes(doc_with_file, storage=storage_service)
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
        # Storage interface: get(uri: str) -> bytes
        return storage.get(blob.uri)

    # ExternalBlob: payload in external storage (S3, GCS, HTTP)
    elif isinstance(blob, ExternalBlob):
        if storage is None:
            raise ValueError(
                f"storage_required_for_external_blob: "
                f"ExternalBlob with kind='{blob.kind}' uri='{blob.uri}' requires storage service parameter"
            )
        # Storage interface handles different external kinds
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


def _normalizer_bytes_in(fetch_result: Any) -> Optional[int]:
    payload = getattr(fetch_result, "payload", None)
    if payload is not None:
        return len(payload)
    telemetry = getattr(fetch_result, "telemetry", None)
    bytes_downloaded = getattr(telemetry, "bytes_downloaded", 0)
    if isinstance(bytes_downloaded, int) and bytes_downloaded > 0:
        return bytes_downloaded
    return None


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
    fetch_result: Any,
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
            if not truncated_value:
                continue
            alias = f"{_PROVIDER_TAG_PREFIX}{key}.{truncated_value}"
            if len(alias) > MAX_TAG_LENGTH:
                continue
        aliases.append(alias)
    return tuple(aliases)


def _robots_tags(fetch_result: Any) -> Tuple[str, ...]:
    tags: list[str] = []
    for event in getattr(fetch_result, "policy_events", ()):
        if isinstance(event, str) and event.startswith("robots_"):
            suffix = event[len("robots_") :]
            suffix_tag = _sanitize_tag_component(suffix)
            if suffix_tag:
                tags.append(f"robots.{suffix_tag}")
    request_metadata = getattr(getattr(fetch_result, "request", None), "metadata", {})
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
    elif isinstance(raw_value, Iterable) and not isinstance(
        raw_value, (bytes, bytearray)
    ):
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


__all__ = [
    "MAX_TAG_LENGTH",
    "document_parser_stats",
    "document_payload_bytes",
    "normalize_from_parse",
    "normalized_primary_text",
    "resolve_provider_reference",
]
