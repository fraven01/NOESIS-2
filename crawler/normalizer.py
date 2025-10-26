"""Normalization contract mapping parser output onto document metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, TypeVar

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .fetcher import FetchResult

from .contracts import NormalizedSource
from .parser import (
    ParseResult,
    ParseStatus,
    ParsedEntity,
    ParserContent,
    ParserStats,
    StructuralElement,
)


@dataclass(frozen=True)
class ExternalDocumentReference:
    """External document reference derived from crawler source data."""

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
        normalized_tags = _normalize_str_mapping(self.provider_tags)
        object.__setattr__(self, "provider_tags", normalized_tags)


@dataclass(frozen=True)
class NormalizedDocumentMeta:
    """Metadata describing document provenance and parser statistics."""

    title: Optional[str]
    language: Optional[str]
    tags: Tuple[str, ...]
    origin_uri: str
    media_type: str
    parser_stats: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.title is not None:
            object.__setattr__(self, "title", self.title.strip())
        if self.language is not None:
            object.__setattr__(self, "language", self.language.strip())
        origin_uri = _require_identifier(self.origin_uri, "origin_uri")
        object.__setattr__(self, "origin_uri", origin_uri)
        media_type = _require_identifier(self.media_type, "media_type")
        object.__setattr__(self, "media_type", media_type.lower())
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        if not isinstance(self.parser_stats, Mapping):
            raise TypeError("parser_stats_must_be_mapping")
        object.__setattr__(
            self, "parser_stats", MappingProxyType(dict(self.parser_stats))
        )


@dataclass(frozen=True)
class NormalizedDocumentContent:
    """Content payload produced by the parser."""

    primary_text: Optional[str]
    binary_payload_ref: Optional[str]
    structural_elements: Tuple[StructuralElement, ...]
    entities: Tuple[ParsedEntity, ...]

    def __post_init__(self) -> None:
        if self.primary_text is not None:
            object.__setattr__(self, "primary_text", self.primary_text.strip())
        if self.binary_payload_ref is not None:
            object.__setattr__(
                self, "binary_payload_ref", self.binary_payload_ref.strip()
            )
        object.__setattr__(
            self,
            "structural_elements",
            _ensure_tuple(self.structural_elements, StructuralElement),
        )
        object.__setattr__(self, "entities", _ensure_tuple(self.entities, ParsedEntity))
        if not (self.primary_text or self.binary_payload_ref):
            raise ValueError("content_payload_missing")


@dataclass(frozen=True)
class NormalizedDocument:
    """Normalized crawler document linking metadata, content and provenance."""

    tenant_id: str
    workflow_id: str
    document_id: str
    meta: NormalizedDocumentMeta
    content: NormalizedDocumentContent
    external_ref: ExternalDocumentReference
    stats: ParserStats
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tenant = _require_identifier(self.tenant_id, "tenant_id")
        workflow = _require_identifier(self.workflow_id, "workflow_id")
        document = _require_identifier(self.document_id, "document_id")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workflow_id", workflow)
        object.__setattr__(self, "document_id", document)
        if not isinstance(self.stats, ParserStats):
            raise TypeError("stats_must_be_parser_stats")
        object.__setattr__(
            self, "diagnostics", _normalize_diagnostics(self.diagnostics)
        )


def build_normalized_document(
    *,
    parse_result: ParseResult,
    source: NormalizedSource,
    tenant_id: str,
    workflow_id: str,
    document_id: str,
    tags: Optional[Sequence[str]] = None,
) -> NormalizedDocument:
    """Compose a :class:`NormalizedDocument` from parser output and source data."""

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
    bytes_in = _normalizer_bytes_in(parse_result.fetch)
    if bytes_in is not None:
        parser_stats["normalizer.bytes_in"] = bytes_in

    meta = NormalizedDocumentMeta(
        title=content.title,
        language=content.content_language,
        tags=normalized_tags,
        origin_uri=canonical_source,
        media_type=content.media_type,
        parser_stats=parser_stats,
    )

    doc_content = NormalizedDocumentContent(
        primary_text=content.primary_text,
        binary_payload_ref=content.binary_payload_ref,
        structural_elements=content.structural_elements,
        entities=content.entities,
    )

    external_ref = ExternalDocumentReference(
        provider=source.provider,
        external_id=source.external_id,
        canonical_source=canonical_source,
        provider_tags=source.provider_tags,
    )

    return NormalizedDocument(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        meta=meta,
        content=doc_content,
        external_ref=external_ref,
        stats=stats,
        diagnostics=parse_result.diagnostics,
    )


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


T = TypeVar("T")


def _ensure_tuple(values: Optional[Sequence[T]], expected_type: type) -> Tuple[T, ...]:
    if values is None:
        return ()
    tupled: Tuple[T, ...] = tuple(values)  # type: ignore[var-annotated]
    for entry in tupled:
        if not isinstance(entry, expected_type):
            raise TypeError("unexpected_entry_type")
    return tupled


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
        # Keep warnings as a list to match telemetry payload expectations downstream
        # (e.g. serializer targets) even though the underlying stats store tuples.
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
