from datetime import datetime, timezone

import pytest

from crawler.contracts import NormalizedSource
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from uuid import uuid4

from documents.parsers import ParsedTextBlock

from crawler.normalizer import NormalizedDocument, build_normalized_document
from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)


def _make_fetch_result(canonical_source: str) -> FetchResult:
    request = FetchRequest(
        canonical_source=canonical_source,
        politeness=PolitenessContext(host="example.com"),
    )
    metadata = FetchMetadata(
        status_code=200,
        content_type="text/html",
        etag="abc",
        last_modified=datetime.now(tz=timezone.utc).isoformat(),
        content_length=128,
    )
    telemetry = FetchTelemetry(latency=0.1, bytes_downloaded=128)
    return FetchResult(
        status=FetchStatus.FETCHED,
        request=request,
        payload=b"<html></html>",
        metadata=metadata,
        telemetry=telemetry,
    )


def _make_parse_result(canonical_source: str) -> ParseResult:
    fetch = _make_fetch_result(canonical_source)
    content = ParserContent(
        media_type="text/html",
        primary_text="Example text",
        title=" Example Title ",
        content_language="en",
        structural_elements=(ParsedTextBlock(text="Title", kind="heading"),),
    )
    stats = ParserStats(
        token_count=2,
        character_count=12,
        extraction_path="html.body",
        error_fraction=0.0,
        warnings=(" minor ",),
    )
    return build_parse_result(
        fetch,
        status=ParseStatus.PARSED,
        content=content,
        stats=stats,
        diagnostics=[" ok "],
    )


def test_build_normalized_document_merges_metadata_and_content() -> None:
    canonical = "https://example.com/article"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/article",
        provider_tags={"collection": "news"},
    )

    document_id = str(uuid4())
    document = build_normalized_document(
        parse_result=parse_result,
        source=source,
        tenant_id="tenant-1",
        workflow_id="wf-42",
        document_id=document_id,
        tags=[" featured ", "news", "featured"],
    )

    assert isinstance(document, NormalizedDocument)
    assert document.tenant_id == "tenant-1"
    assert document.workflow_id == "wf-42"
    assert document.document_id == document_id
    assert document.meta.origin_uri == canonical
    assert document.media_type == "text/html"
    assert document.meta.title == "Example Title"
    assert document.meta.language == "en"
    assert document.meta.tags == ["featured", "news"]
    assert document.external_ref.external_id == source.external_id
    assert document.external_ref.provider == "web"
    assert document.external_ref.provider_tags["collection"] == "news"
    assert document.primary_text == "Example text"
    assert document.stats.token_count == 2
    assert document.parser_stats["parser.token_count"] == 2
    assert document.parser_stats["parser.warnings"] == ["minor"]
    assert document.parser_stats["normalizer.bytes_in"] == len(b"<html></html>")
    assert document.diagnostics == ("ok",)


def test_build_normalized_document_requires_parsed_status() -> None:
    canonical = "https://example.com/article"
    fetch = _make_fetch_result(canonical)
    stats = ParserStats(
        token_count=0,
        character_count=0,
        extraction_path="html.body",
        error_fraction=0.0,
    )
    result = build_parse_result(
        fetch,
        status=ParseStatus.UNSUPPORTED_MEDIA,
        stats=stats,
        diagnostics=["unsupported"],
    )
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/article",
        provider_tags={},
    )

    with pytest.raises(ValueError):
        build_normalized_document(
            parse_result=result,
            source=source,
            tenant_id="tenant",
            workflow_id="wf",
            document_id="doc",
        )


def test_build_normalized_document_requires_workflow_identifier() -> None:
    canonical = "https://example.com/resource"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/resource",
        provider_tags={},
    )

    with pytest.raises(ValueError):
        build_normalized_document(
            parse_result=parse_result,
            source=source,
            tenant_id="tenant",
            workflow_id=" ",
            document_id="doc",
        )


def test_build_normalized_document_rejects_canonical_mismatch() -> None:
    canonical = "https://example.com/entry"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source="https://other.example.com/entry",
        external_id="web::https://other.example.com/entry",
        provider_tags={},
    )

    with pytest.raises(ValueError):
        build_normalized_document(
            parse_result=parse_result,
            source=source,
            tenant_id="tenant",
            workflow_id="wf",
            document_id="doc",
        )
