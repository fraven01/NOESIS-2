"""Parser contracts for normalized ingestion outputs across media types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple, TypeVar

from .errors import CrawlerError, ErrorClass
from .fetcher import FetchResult


_LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")


class ParseStatus(str, Enum):
    """Semantic parser outcomes exposed to downstream processing."""

    PARSED = "parsed"
    UNSUPPORTED_MEDIA = "unsupported_media"
    PARSER_FAILURE = "parser_failure"


@dataclass(frozen=True)
class StructuralElement:
    """Structured block extracted from the source document."""

    kind: str
    text: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_kind = (self.kind or "").strip().lower()
        if not normalized_kind:
            raise ValueError("kind_required")
        object.__setattr__(self, "kind", normalized_kind)
        if self.text is not None:
            object.__setattr__(self, "text", self.text.strip())
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata_must_be_mapping")


@dataclass(frozen=True)
class ParsedEntity:
    """Entity recognized during parsing.

    Offsets follow the ``[start, end)`` convention with respect to the
    ``primary_text`` characters.
    """

    label: str
    value: str
    confidence: Optional[float] = None
    offsets: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        label = (self.label or "").strip()
        value = (self.value or "").strip()
        if not label or not value:
            raise ValueError("label_and_value_required")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "value", value)
        if self.confidence is not None:
            if self.confidence < 0 or self.confidence > 1:
                raise ValueError("confidence_out_of_bounds")
        if self.offsets is not None:
            start, end = self.offsets
            if start < 0 or end < start:
                raise ValueError("invalid_offsets")


@dataclass(frozen=True)
class ParserContent:
    """Primary parser payload returned to the ingestion pipeline.

    ``content_language`` must be provided as a BCP 47 / ISO 639-1 language tag.
    """

    media_type: str
    primary_text: Optional[str] = None
    binary_payload_ref: Optional[str] = None
    title: Optional[str] = None
    content_language: Optional[str] = None
    structural_elements: Tuple[StructuralElement, ...] = ()
    entities: Tuple[ParsedEntity, ...] = ()

    def __post_init__(self) -> None:
        media_type = (self.media_type or "").strip().lower()
        if not media_type:
            raise ValueError("media_type_required")
        object.__setattr__(self, "media_type", media_type)

        has_text = bool(self.primary_text and self.primary_text.strip())
        has_binary = bool(self.binary_payload_ref and self.binary_payload_ref.strip())
        if not (has_text or has_binary):
            raise ValueError("primary_text_or_binary_required")
        if self.primary_text is not None:
            object.__setattr__(self, "primary_text", self.primary_text.strip())
        if self.binary_payload_ref is not None:
            object.__setattr__(
                self, "binary_payload_ref", self.binary_payload_ref.strip()
            )
        if self.title is not None:
            object.__setattr__(self, "title", self.title.strip())
        if self.content_language is not None:
            language = self.content_language.strip()
            if not language:
                raise ValueError("content_language_invalid")
            normalized_language = language.lower()
            if not _LANGUAGE_TAG_RE.fullmatch(normalized_language):
                raise ValueError("content_language_invalid")
            object.__setattr__(self, "content_language", normalized_language)
        object.__setattr__(
            self,
            "structural_elements",
            _tupled(self.structural_elements, StructuralElement),
        )
        object.__setattr__(self, "entities", _tupled(self.entities, ParsedEntity))


@dataclass(frozen=True)
class ParserStats:
    """Observability metrics emitted by parsers.

    ``extraction_path`` should follow dotted component conventions (e.g.
    ``html.body`` or ``pdf.text``) to ease downstream telemetry analysis.
    """

    token_count: int
    character_count: int
    extraction_path: str
    error_fraction: float = 0.0
    warnings: Tuple[str, ...] = ()
    boilerplate_reduction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count_negative")
        if self.character_count < 0:
            raise ValueError("character_count_negative")
        extraction_path = (self.extraction_path or "").strip()
        if not extraction_path:
            raise ValueError("extraction_path_required")
        object.__setattr__(self, "extraction_path", extraction_path)
        if self.error_fraction < 0 or self.error_fraction > 1:
            raise ValueError("error_fraction_bounds")
        if self.boilerplate_reduction is not None:
            if self.boilerplate_reduction < 0 or self.boilerplate_reduction > 1:
                raise ValueError("boilerplate_reduction_bounds")
        sanitized = tuple(filter(None, (w.strip() for w in self.warnings)))
        object.__setattr__(self, "warnings", sanitized)


@dataclass(frozen=True)
class ParseResult:
    """Parser result shared with downstream enrichment steps."""

    status: ParseStatus
    fetch: FetchResult
    content: Optional[ParserContent]
    stats: Optional[ParserStats]
    diagnostics: Tuple[str, ...] = ()
    error: Optional[CrawlerError] = None

    def __post_init__(self) -> None:
        if self.status is ParseStatus.PARSED:
            if self.content is None:
                raise ValueError("content_required_for_parsed")
            if self.stats is None:
                raise ValueError("stats_required_for_parsed")
        if self.status in {ParseStatus.UNSUPPORTED_MEDIA, ParseStatus.PARSER_FAILURE}:
            if self.content is not None:
                raise ValueError("error_states_must_not_include_content")
        object.__setattr__(
            self, "diagnostics", _normalize_diagnostics(self.diagnostics)
        )
        if self.status is ParseStatus.PARSED:
            object.__setattr__(self, "error", None)


def build_parse_result(
    fetch_result: FetchResult,
    *,
    status: ParseStatus,
    content: Optional[ParserContent] = None,
    stats: Optional[ParserStats] = None,
    diagnostics: Optional[Sequence[str]] = None,
) -> ParseResult:
    """Compose a :class:`ParseResult` ensuring contract invariants."""

    diag_tuple = tuple(diagnostics) if diagnostics else ()
    if status is ParseStatus.PARSED and (content is None or stats is None):
        raise ValueError("parsed_requires_content_and_stats")
    if status is not ParseStatus.PARSED:
        content = None
    return ParseResult(
        status=status,
        fetch=fetch_result,
        content=content,
        stats=stats,
        diagnostics=diag_tuple,
        error=_build_parse_error(fetch_result, status, diag_tuple),
    )


def compute_parser_stats(
    *,
    primary_text: Optional[str],
    extraction_path: str,
    warnings: Optional[Sequence[str]] = None,
    error_fraction: float = 0.0,
    raw_token_count: Optional[int] = None,
) -> ParserStats:
    """Estimate parser stats from normalized text and optional raw metrics."""

    text = (primary_text or "").strip()
    token_count = len(text.split()) if text else 0
    character_count = len(text)
    warnings_tuple = tuple(warnings) if warnings else ()
    boilerplate_reduction = None
    if raw_token_count is not None and raw_token_count > 0:
        ratio = 1 - (token_count / raw_token_count)
        boilerplate_reduction = max(0.0, min(1.0, ratio))
    return ParserStats(
        token_count=token_count,
        character_count=character_count,
        extraction_path=extraction_path,
        error_fraction=error_fraction,
        warnings=warnings_tuple,
        boilerplate_reduction=boilerplate_reduction,
    )


T = TypeVar("T")


def _tupled(values: Sequence[T], expected_type: type) -> Tuple[T, ...]:
    if values is None:
        return ()
    tupled: Tuple[T, ...] = tuple(values)
    for value in tupled:
        if not isinstance(value, expected_type):
            raise TypeError("unexpected_sequence_entry")
    return tupled


def _normalize_diagnostics(entries: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        normalized
        for normalized in ((entry or "").strip() for entry in entries)
        if normalized
    )


def _build_parse_error(
    fetch_result: FetchResult,
    status: ParseStatus,
    diagnostics: Tuple[str, ...],
) -> Optional[CrawlerError]:
    if status is ParseStatus.PARSED:
        return None

    if status is ParseStatus.UNSUPPORTED_MEDIA:
        error_class = ErrorClass.UNSUPPORTED_MEDIA
    else:
        error_class = ErrorClass.PARSER_FAILURE

    reason = diagnostics[0].strip() if diagnostics else status.value
    attributes: Dict[str, object] = {}
    if diagnostics:
        attributes["diagnostics"] = diagnostics

    provider = _provider_from_fetch(fetch_result)

    return CrawlerError(
        error_class=error_class,
        reason=reason,
        source=fetch_result.request.canonical_source,
        provider=provider,
        status_code=fetch_result.metadata.status_code,
        attributes=attributes,
    )


def _provider_from_fetch(fetch_result: FetchResult) -> Optional[str]:
    provider = fetch_result.request.metadata.get("provider")
    if isinstance(provider, str):
        cleaned = provider.strip()
        return cleaned or None
    return None
