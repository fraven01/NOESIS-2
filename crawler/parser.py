"""Parser result helpers bridging crawler fetches and document contracts."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from documents.parsers import (
    ParseResult,
    ParseStatus,
    ParsedEntity,
    ParsedResult,
    ParsedTextBlock,
    ParserContent,
    ParserStats,
    compute_parser_stats,
    normalize_diagnostics,
)

from .errors import CrawlerError, ErrorClass
from .fetcher import FetchResult


def build_parse_result(
    fetch_result: FetchResult,
    *,
    status: ParseStatus,
    content: Optional[ParserContent] = None,
    stats: Optional[ParserStats] = None,
    diagnostics: Optional[Sequence[str]] = None,
) -> ParseResult:
    """Compose a :class:`ParseResult` ensuring contract invariants."""

    normalized_diagnostics = normalize_diagnostics(diagnostics or ())
    if status is ParseStatus.PARSED and (content is None or stats is None):
        raise ValueError("parsed_requires_content_and_stats")
    if status is not ParseStatus.PARSED:
        content = None
    return ParseResult(
        status=status,
        fetch=fetch_result,
        content=content,
        stats=stats,
        diagnostics=normalized_diagnostics,
        error=_build_parse_error(fetch_result, status, normalized_diagnostics),
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


__all__ = [
    "ParseResult",
    "ParseStatus",
    "ParsedEntity",
    "ParsedResult",
    "ParsedTextBlock",
    "ParserContent",
    "ParserStats",
    "build_parse_result",
    "compute_parser_stats",
]
