import pytest

from crawler.delta import DeltaStatus, evaluate_delta
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from crawler.normalizer import build_normalized_document
from crawler.parser import (
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)
from crawler.contracts import NormalizedSource
from documents.parsers import ParsedTextBlock
from uuid import uuid4


@pytest.fixture(autouse=True)
def _stub_vector_hashing(monkeypatch):
    from ai_core.rag.vector_client import DedupSignatures

    def _build_signatures(**_: object) -> DedupSignatures:
        return DedupSignatures(content_hash="stub-hash")

    monkeypatch.setattr("crawler.delta.build_dedup_signatures", _build_signatures)
    monkeypatch.setattr(
        "crawler.delta.extract_primary_text_hash", lambda *args, **kwargs: None
    )


def _make_document(text: str = "Body text"):
    fetch = FetchResult(
        status=FetchStatus.FETCHED,
        request=FetchRequest(
            canonical_source="https://example.com/doc",
            politeness=PolitenessContext(host="example.com"),
        ),
        payload=b"<html>Body text</html>",
        metadata=FetchMetadata(
            status_code=200,
            content_type="text/html",
            etag="abc",
            last_modified="2023-01-01T00:00:00+00:00",
            content_length=20,
        ),
        telemetry=FetchTelemetry(latency=0.1, bytes_downloaded=20),
    )
    parse = build_parse_result(
        fetch,
        status=ParseStatus.PARSED,
        content=ParserContent(
            media_type="text/html",
            primary_text=text,
            binary_payload_ref=None,
            title="Title",
            content_language="en",
            structural_elements=(ParsedTextBlock(text=text, kind="paragraph"),),
        ),
        stats=ParserStats(
            token_count=10,
            character_count=len(text),
            extraction_path="html.body",
            error_fraction=0.0,
        ),
    )
    source = NormalizedSource(
        provider="web",
        canonical_source="https://example.com/doc",
        external_id="web::https://example.com/doc",
        provider_tags={},
    )
    return build_normalized_document(
        parse_result=parse,
        source=source,
        tenant_id="tenant-a",
        workflow_id="wf-1",
        document_id=str(uuid4()),
        tags=(),
    )


def test_evaluate_delta_marks_new_when_no_previous_hash() -> None:
    document = _make_document()

    decision = evaluate_delta(document)

    assert decision.status is DeltaStatus.NEW
    assert decision.reason == "no_previous_hash"
    assert decision.signatures.content_hash == "stub-hash"
    assert decision.version == 1


def test_evaluate_delta_detects_unchanged_payload() -> None:
    document = _make_document()

    decision = evaluate_delta(
        document,
        previous_content_hash="stub-hash",
        previous_version=3,
    )

    assert decision.status is DeltaStatus.UNCHANGED
    assert decision.reason == "hash_match"
    assert decision.version == 3


def test_evaluate_delta_detects_changes() -> None:
    document = _make_document()

    decision = evaluate_delta(
        document,
        previous_content_hash="different-hash",
        previous_version=2,
    )

    assert decision.status is DeltaStatus.CHANGED
    assert decision.reason == "hash_mismatch"
    assert decision.version == 3
