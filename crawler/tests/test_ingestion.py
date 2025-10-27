import pytest

from ai_core.rag.ingestion_contracts import ChunkMeta

from crawler.contracts import MAX_EXTERNAL_ID_LENGTH, normalize_source
from crawler.delta import DeltaDecision, DeltaSignatures, DeltaStatus
from crawler.errors import ErrorClass
from datetime import datetime, timezone
from uuid import uuid4

from documents.parsers import ParsedTextBlock

from crawler.contracts import NormalizedSource
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from crawler.ingestion import (
    IngestionStatus,
    build_ingestion_decision,
    build_ingestion_error,
)
from documents.contracts import NormalizedDocument

from crawler.normalizer import build_normalized_document
from crawler.parser import (
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)
from crawler.retire import LifecycleDecision, LifecycleState


def _make_fetch_result(canonical_source: str, payload: bytes) -> FetchResult:
    request = FetchRequest(
        canonical_source=canonical_source,
        politeness=PolitenessContext(host="example.com"),
    )
    metadata = FetchMetadata(
        status_code=200,
        content_type="text/html",
        etag="abc",
        last_modified=datetime.now(tz=timezone.utc).isoformat(),
        content_length=len(payload),
    )
    telemetry = FetchTelemetry(latency=0.1, bytes_downloaded=len(payload))
    return FetchResult(
        status=FetchStatus.FETCHED,
        request=request,
        payload=payload,
        metadata=metadata,
        telemetry=telemetry,
    )


def _make_document(
    *,
    canonical_source: str = "https://example.com/doc",
    external_id: str = "web::https://example.com/doc",
) -> NormalizedDocument:
    payload = b"<html>Body text</html>"
    fetch = _make_fetch_result(canonical_source, payload)
    content = ParserContent(
        media_type="text/html",
        primary_text="Body text",
        binary_payload_ref=None,
        title=" Example Title ",
        content_language=" en ",
        structural_elements=(
            ParsedTextBlock(text="Body text", kind="paragraph"),
        ),
    )
    stats = ParserStats(
        token_count=42,
        character_count=128,
        extraction_path="html.body",
        error_fraction=0.0,
    )
    parse_result = build_parse_result(
        fetch,
        status=ParseStatus.PARSED,
        content=content,
        stats=stats,
    )
    source = NormalizedSource(
        provider="web",
        canonical_source=canonical_source,
        external_id=external_id,
        provider_tags={"collection": "docs"},
    )
    return build_normalized_document(
        parse_result=parse_result,
        source=source,
        tenant_id="tenant-a",
        workflow_id="workflow-1",
        document_id=str(uuid4()),
        tags=("alpha", "beta"),
    )


def _make_signatures(content_hash: str) -> DeltaSignatures:
    return DeltaSignatures(content_hash=content_hash)


def test_build_ingestion_decision_upsert_includes_metadata() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        _make_signatures("hash123"),
        1,
        "no_previous_hash",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-7")

    assert decision.status is IngestionStatus.UPSERT
    assert decision.reason == "no_previous_hash"
    assert decision.payload is not None
    assert decision.lifecycle_state is LifecycleState.ACTIVE
    assert decision.policy_events == ()
    payload = decision.payload
    assert payload.case_id == "case-7"
    assert payload.content_hash == "hash123"
    assert payload.origin_uri == "https://example.com/doc"
    assert payload.external_id == "web::https://example.com/doc"
    assert payload.source == "crawler"
    assert payload.provider == "web"
    assert payload.tags == ("alpha", "beta")
    assert payload.parser_stats["parser.token_count"] == 42
    assert payload.provider_tags["collection"] == "docs"
    assert payload.language == "en"
    assert payload.title == "Example Title"


def test_build_ingestion_decision_skips_when_unchanged() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.UNCHANGED,
        _make_signatures("hash999"),
        3,
        "hash_match",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    assert decision.status is IngestionStatus.SKIP
    assert decision.payload is None
    assert decision.reason == "hash_match"
    assert decision.lifecycle_state is LifecycleState.ACTIVE


def test_build_ingestion_decision_skips_near_duplicate() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEAR_DUPLICATE,
        _make_signatures("hash777"),
        None,
        "near_duplicate:0.950",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    assert decision.status is IngestionStatus.SKIP
    assert decision.payload is None
    assert decision.reason == "near_duplicate:0.950"
    assert decision.lifecycle_state is LifecycleState.ACTIVE


def test_build_ingestion_decision_requires_case_id() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        _make_signatures("hash000"),
        1,
        "new",
    )

    with pytest.raises(ValueError):
        build_ingestion_decision(document, delta, case_id=" ")


def test_build_ingestion_decision_retire_overrides_delta() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.UNCHANGED,
        _make_signatures("hash111"),
        5,
        "hash_match",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-9", retire=True)

    assert decision.status is IngestionStatus.RETIRE
    assert decision.payload is not None
    assert decision.payload.content_hash == "hash111"
    assert decision.payload.source == "crawler"
    assert decision.reason == "retired"
    assert decision.lifecycle_state is LifecycleState.RETIRED


def test_build_ingestion_decision_respects_lifecycle_decision() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.CHANGED,
        _make_signatures("hash333"),
        2,
        "hash_mismatch",
    )
    lifecycle = LifecycleDecision(
        state=LifecycleState.RETIRED,
        reason="gone_410",
        policy_events=("gone_410",),
    )

    decision = build_ingestion_decision(
        document,
        delta,
        case_id="case-11",
        lifecycle=lifecycle,
    )

    assert decision.status is IngestionStatus.RETIRE
    assert decision.reason == "gone_410"
    assert decision.policy_events == ("gone_410",)
    assert decision.lifecycle_state is LifecycleState.RETIRED
    assert decision.payload is not None


def test_build_ingestion_error_wraps_metadata() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        _make_signatures("hash222"),
        1,
        "new",
    )
    decision = build_ingestion_decision(document, delta, case_id="case-5")
    assert decision.payload is not None

    error = build_ingestion_error(
        payload=decision.payload,
        reason="ingest_failed",
        error_code="INGEST_TIMEOUT",
        status_code=502,
    )

    assert error.error_class is ErrorClass.INGESTION_FAILURE
    assert error.reason == "ingest_failed"
    assert error.provider == "web"
    assert error.source == "https://example.com/doc"
    assert error.status_code == 502
    assert error.attributes["error_code"] == "INGEST_TIMEOUT"


def test_ingestion_payload_hashed_external_id_validates_with_chunk_meta() -> None:
    long_segment = "z" * 1500
    normalized_source = normalize_source(
        "web", f"https://example.com/resource/{long_segment}"
    )
    document = _make_document(
        canonical_source=normalized_source.canonical_source,
        external_id=normalized_source.external_id,
    )
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        _make_signatures("hash555"),
        1,
        "no_previous_hash",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-hash")

    assert decision.payload is not None
    payload = decision.payload
    assert payload.external_id == normalized_source.external_id
    assert len(payload.external_id) <= MAX_EXTERNAL_ID_LENGTH

    chunk_meta = ChunkMeta.model_validate(
        {
            "tenant_id": payload.tenant_id,
            "case_id": payload.case_id,
            "source": payload.source,
            "hash": payload.content_hash,
            "external_id": payload.external_id,
            "content_hash": payload.content_hash,
            "workflow_id": payload.workflow_id,
            "document_id": payload.document_id,
        }
    )

    assert chunk_meta.external_id == payload.external_id
