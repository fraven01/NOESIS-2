from typing import Tuple

import pytest

from crawler.errors import ErrorClass
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from pydantic import ValidationError

from documents.parsers import ParsedTextBlock

from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParsedEntity,
    ParserContent,
    ParserStats,
    build_parse_result,
    compute_parser_stats,
)


def make_fetch_result(
    status: FetchStatus = FetchStatus.FETCHED, payload: bytes | None = b"data"
) -> FetchResult:
    request = FetchRequest(
        canonical_source="https://example.com/article",
        politeness=PolitenessContext(host="example.com"),
        metadata={"provider": "web"},
    )
    metadata = FetchMetadata(
        status_code=200,
        content_type="text/html",
        etag="abc",
        last_modified="Wed, 01 Jan 2020 00:00:00 GMT",
        content_length=len(payload or b""),
    )
    telemetry = FetchTelemetry(latency=0.12, bytes_downloaded=len(payload or b""))
    return FetchResult(
        status=status,
        request=request,
        payload=payload,
        metadata=metadata,
        telemetry=telemetry,
    )


def test_parser_content_requires_primary_text_or_binary():
    with pytest.raises(ValueError):
        ParserContent(media_type="text/html")


def test_parsed_text_block_rejects_invalid_kind():
    with pytest.raises(ValueError):
        ParsedTextBlock(text="Example", kind="invalid")  # type: ignore[arg-type]


def test_build_parse_result_requires_content_when_parsed():
    fetch = make_fetch_result()
    stats = ParserStats(
        token_count=10,
        character_count=42,
        extraction_path="html.body",
        error_fraction=0.0,
    )
    with pytest.raises(ValueError):
        build_parse_result(fetch, status=ParseStatus.PARSED, stats=stats)


def test_successful_parse_includes_structures_and_entities():
    fetch = make_fetch_result()
    content = ParserContent(
        media_type="text/html",
        primary_text="Hello world",
        title="Example",
        structural_elements=(ParsedTextBlock(text="Example", kind="heading"),),
        entities=(ParsedEntity(label="ORG", value="Noesis"),),
    )
    parsed_view = content.to_parsed_result()
    stats = compute_parser_stats(
        primary_text=content.primary_text,
        extraction_path="html.body",
        warnings=[" minor-truncation "],
        raw_token_count=40,
    )
    result = build_parse_result(
        fetch,
        status=ParseStatus.PARSED,
        content=content,
        stats=stats,
        diagnostics=[" ok "],
    )

    assert result.status is ParseStatus.PARSED
    assert result.content == content
    assert isinstance(result.stats, ParserStats)
    assert result.stats.token_count == 2
    assert result.stats.boilerplate_reduction and result.stats.boilerplate_reduction > 0
    assert result.stats.warnings == ("minor-truncation",)
    assert result.diagnostics == ("ok",)
    assert result.error is None
    assert parsed_view.text_blocks[0].text == "Example"


def test_compute_parser_stats_handles_empty_text():
    stats = compute_parser_stats(primary_text=None, extraction_path="pdf.extract")
    assert stats.token_count == 0
    assert stats.character_count == 0
    assert stats.boilerplate_reduction is None


def test_unsupported_media_discards_payload_and_sets_diagnostic():
    fetch = make_fetch_result()
    stats = compute_parser_stats(
        primary_text=None, extraction_path="pdf.extract", error_fraction=0.2
    )
    result = build_parse_result(
        fetch,
        status=ParseStatus.UNSUPPORTED_MEDIA,
        stats=stats,
        diagnostics=["Unsupported media"],
    )

    assert result.status is ParseStatus.UNSUPPORTED_MEDIA
    assert result.content is None
    assert result.stats == stats
    assert result.diagnostics == ("Unsupported media",)
    assert result.error is not None
    assert result.error.error_class is ErrorClass.UNSUPPORTED_MEDIA
    assert result.error.reason == "Unsupported media"


def test_parser_failure_prevents_content_in_result():
    fetch = make_fetch_result()
    content = ParserContent(
        media_type="application/pdf", binary_payload_ref="s3://bucket/key"
    )
    with pytest.raises(ValidationError):
        ParseResult(
            status=ParseStatus.PARSER_FAILURE,
            fetch=fetch,
            content=content,
            stats=None,
        )


def test_parser_failure_error_classification():
    fetch = make_fetch_result()
    result = build_parse_result(
        fetch,
        status=ParseStatus.PARSER_FAILURE,
        diagnostics=["extractor_crash"],
    )

    assert result.status is ParseStatus.PARSER_FAILURE
    assert result.content is None
    assert result.error is not None
    assert result.error.error_class is ErrorClass.PARSER_FAILURE
    assert result.error.reason == "extractor_crash"
    assert result.error.provider == "web"


def test_parser_content_validates_language_tag():
    content = ParserContent(
        media_type="text/html",
        primary_text="data",
        content_language="EN-US",
    )

    assert content.content_language == "en-us"


@pytest.mark.parametrize("invalid", ["", "engb", "de_", "123", "zzzzzz"])
def test_parser_content_rejects_invalid_language_tag(invalid: str):
    with pytest.raises(ValueError):
        ParserContent(
            media_type="text/html",
            primary_text="data",
            content_language=invalid,
        )


@pytest.mark.parametrize(
    "offsets",
    [(-1, 3), (5, 4)],
)
def test_parsed_entity_rejects_invalid_offsets(offsets: Tuple[int, int]):
    with pytest.raises(ValueError):
        ParsedEntity(label="PER", value="Alice", offsets=offsets)


def test_parser_content_accepts_binary_only_payload():
    content = ParserContent(
        media_type="application/pdf",
        binary_payload_ref="s3://bucket/key",
    )

    assert content.binary_payload_ref == "s3://bucket/key"
    assert content.primary_text is None
