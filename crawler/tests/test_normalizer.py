from datetime import datetime, timezone
import hashlib
from typing import Mapping, Optional, Sequence
from uuid import uuid4

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

from documents import normalize_diagnostics
from documents.contracts import InlineBlob, NormalizedDocument
from documents.normalization import (
    MAX_TAG_LENGTH,
    document_parser_stats,
    document_payload_bytes,
    normalize_from_parse,
    resolve_provider_reference,
)
from documents.parsers import ParsedTextBlock

from crawler.normalizer import build_normalized_document
from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)


def _make_fetch_result(
    canonical_source: str,
    *,
    request_metadata: Optional[Mapping[str, object]] = None,
    policy_events: Sequence[str] = (),
) -> FetchResult:
    request = FetchRequest(
        canonical_source=canonical_source,
        politeness=PolitenessContext(host="example.com"),
        metadata=request_metadata or {},
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
        policy_events=tuple(policy_events),
    )


def _make_parse_result(
    canonical_source: str,
    *,
    request_metadata: Optional[Mapping[str, object]] = None,
    policy_events: Sequence[str] = (),
) -> ParseResult:
    fetch = _make_fetch_result(
        canonical_source,
        request_metadata=request_metadata,
        policy_events=policy_events,
    )
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


def test_normalize_from_parse_merges_metadata_and_content() -> None:
    canonical = "https://example.com/article"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/article",
        provider_tags={"collection": "news"},
    )

    document_uuid = uuid4()
    document = normalize_from_parse(
        parse_result=parse_result,
        tenant_id="tenant-1",
        workflow_id="wf-42",
        document_id=document_uuid,
        canonical_source=source.canonical_source,
        provider=source.provider,
        external_id=source.external_id,
        provider_tags=source.provider_tags,
        tags=[" featured ", "news", "featured"],
    )

    assert isinstance(document, NormalizedDocument)
    assert document.ref.tenant_id == "tenant-1"
    assert document.ref.workflow_id == "wf-42"
    assert document.ref.document_id == document_uuid
    assert document.meta.origin_uri == canonical
    assert isinstance(document.blob, InlineBlob)
    assert document.blob.media_type == "text/html"
    assert document.meta.title == "Example Title"
    assert document.meta.language == "en"
    assert document.meta.tags == ["featured", "news", "provider.collection.news"]
    provider = resolve_provider_reference(document)
    assert provider.external_id == source.external_id
    assert provider.provider == "web"
    assert provider.provider_tags["collection"] == "news"
    parser_stats = document_parser_stats(document)
    assert parser_stats["parser.token_count"] == 2
    assert parser_stats["parser.warnings"] == ["minor"]
    assert parser_stats["normalizer.bytes_in"] == len(b"<html></html>")
    expected_text_hash = hashlib.sha256("Example text".encode("utf-8")).hexdigest()
    assert parser_stats["crawler.primary_text_hash_sha256"] == expected_text_hash
    payload = document_payload_bytes(document)
    assert payload == b"<html></html>"
    assert normalize_diagnostics(parse_result.diagnostics) == ("ok",)


def test_build_normalized_document_includes_robot_tags() -> None:
    canonical = "https://example.com/robots"
    parse_result = _make_parse_result(
        canonical,
        request_metadata={"robots": {"index": "noindex", "follow": "nofollow"}},
        policy_events=("robots_allow", "robots_index"),
    )
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/robots",
        provider_tags={"source": canonical},
    )

    document = normalize_from_parse(
        parse_result=parse_result,
        tenant_id="tenant-robots",
        workflow_id="wf-robots",
        document_id=uuid4(),
        canonical_source=source.canonical_source,
        provider=source.provider,
        external_id=source.external_id,
        provider_tags=source.provider_tags,
        tags=["base"],
    )

    assert set(document.meta.tags) == {
        "base",
        "provider.source.https-example.com-robots",
        "robots.allow",
        "robots.index",
        "robots.index.noindex",
        "robots.follow.nofollow",
    }


def test_normalize_from_parse_truncates_long_provider_tags() -> None:
    long_path = "a" * 200
    canonical = f"https://example.com/{long_path}"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id=f"web::{canonical}",
        provider_tags={"source": canonical},
    )

    document = normalize_from_parse(
        parse_result=parse_result,
        tenant_id="tenant-long-tag",
        workflow_id="wf-long-tag",
        document_id=uuid4(),
        canonical_source=source.canonical_source,
        provider=source.provider,
        external_id=source.external_id,
        provider_tags=source.provider_tags,
    )

    provider_tags = [
        tag for tag in document.meta.tags if tag.startswith("provider.source.")
    ]
    assert provider_tags, "expected provider tag alias to be emitted"
    assert all(len(tag) <= MAX_TAG_LENGTH for tag in provider_tags)


def test_normalize_from_parse_requires_parsed_status() -> None:
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
        normalize_from_parse(
            parse_result=result,
            tenant_id="tenant",
            workflow_id="wf",
            document_id="doc",
            canonical_source=source.canonical_source,
            provider=source.provider,
            external_id=source.external_id,
            provider_tags=source.provider_tags,
        )


def test_normalize_from_parse_requires_workflow_identifier() -> None:
    canonical = "https://example.com/resource"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/resource",
        provider_tags={},
    )

    with pytest.raises(ValueError):
        normalize_from_parse(
            parse_result=parse_result,
            tenant_id="tenant",
            workflow_id=" ",
            document_id="doc",
            canonical_source=source.canonical_source,
            provider=source.provider,
            external_id=source.external_id,
            provider_tags=source.provider_tags,
        )


def test_normalize_from_parse_rejects_canonical_mismatch() -> None:
    canonical = "https://example.com/entry"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source="https://other.example.com/entry",
        external_id="web::https://other.example.com/entry",
        provider_tags={},
    )

    with pytest.raises(ValueError):
        normalize_from_parse(
            parse_result=parse_result,
            tenant_id="tenant",
            workflow_id="wf",
            document_id="doc",
            canonical_source=source.canonical_source,
            provider=source.provider,
            external_id=source.external_id,
            provider_tags=source.provider_tags,
        )


def test_build_normalized_document_proxies_facade() -> None:
    canonical = "https://example.com/wrapper"
    parse_result = _make_parse_result(canonical)
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical,
        external_id="web::https://example.com/wrapper",
        provider_tags={"collection": "news"},
    )

    expected = normalize_from_parse(
        parse_result=parse_result,
        tenant_id="tenant",
        workflow_id="workflow",
        document_id=uuid4(),
        canonical_source=source.canonical_source,
        provider=source.provider,
        external_id=source.external_id,
        provider_tags=source.provider_tags,
    )

    actual = build_normalized_document(
        parse_result=parse_result,
        source=source,
        tenant_id="tenant",
        workflow_id="workflow",
        document_id=expected.ref.document_id,
    )

    assert actual.model_dump(exclude={"created_at"}) == expected.model_dump(
        exclude={"created_at"}
    )
