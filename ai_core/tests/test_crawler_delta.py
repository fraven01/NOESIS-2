from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from ai_core.rag.delta import DeltaDecision, DeltaStatus, evaluate_delta
from ai_core.rag.vector_client import DedupSignatures
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from documents.normalization import normalize_from_parse
from documents.parsers import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    ParsedTextBlock,
)


@pytest.fixture(autouse=True)
def _stub_vector_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _build_signatures(**_: object) -> DedupSignatures:
        return DedupSignatures(content_hash="stub-hash")

    monkeypatch.setattr(
        "ai_core.rag.delta.build_dedup_signatures",
        _build_signatures,
    )
    monkeypatch.setattr(
        "ai_core.rag.delta.extract_primary_text_hash",
        lambda *args, **kwargs: None,
    )


def _make_fetch_result() -> FetchResult:
    request = FetchRequest(
        canonical_source="https://example.com/doc",
        politeness=PolitenessContext(host="example.com"),
    )
    metadata = FetchMetadata(
        status_code=200,
        content_type="text/html",
        etag="abc",
        last_modified="2023-01-01T00:00:00+00:00",
        content_length=20,
    )
    telemetry = FetchTelemetry(latency=0.1, bytes_downloaded=20)
    return FetchResult(
        status=FetchStatus.FETCHED,
        request=request,
        payload=b"<html>Body text</html>",
        metadata=metadata,
        telemetry=telemetry,
    )


def _make_document(text: str = "Body text"):
    fetch = _make_fetch_result()
    content = ParserContent(
        media_type="text/html",
        primary_text=text,
        binary_payload_ref=None,
        title="Title",
        content_language="en",
        structural_elements=(ParsedTextBlock(text=text, kind="paragraph"),),
    )
    stats = ParserStats(
        token_count=10,
        character_count=len(text),
        extraction_path="html.body",
        error_fraction=0.0,
    )
    parse = ParseResult(
        status=ParseStatus.PARSED,
        fetch=fetch,
        content=content,
        stats=stats,
    )
    return normalize_from_parse(
        parse_result=parse,
        tenant_id="tenant-a",
        workflow_id="wf-1",
        document_id=str(uuid4()),
        canonical_source="https://example.com/doc",
        provider="web",
        external_id="web::https://example.com/doc",
        provider_tags={},
        tags=(),
        fetch_result=fetch,
        ingest_source="crawler",
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


def test_delta_decision_from_legacy_preserves_attributes() -> None:
    decision = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        DedupSignatures(content_hash="hash"),
        1,
        "reason",
        parent_document_id="parent",
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.signatures.content_hash == "hash"
    assert decision.version == 1
    assert decision.parent_document_id == "parent"
