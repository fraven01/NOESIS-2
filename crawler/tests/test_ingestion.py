import pytest

from ai_core.rag.ingestion_contracts import ChunkMeta

from crawler.contracts import MAX_EXTERNAL_ID_LENGTH, NormalizedSource, normalize_source
from crawler.delta import DeltaDecision, DeltaSignatures, DeltaStatus
from crawler.errors import ErrorClass
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
from crawler.normalizer import build_normalized_document
from crawler.parser import (
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)
from crawler.retire import LifecycleDecision, LifecycleState
from datetime import datetime, timezone
from documents.contracts import NormalizedDocument
from documents.parsers import ParsedTextBlock
from uuid import uuid4


@pytest.fixture(autouse=True)
def _stub_profile_resolution(monkeypatch):
    from types import SimpleNamespace

    fake_resolution = SimpleNamespace(
        vector_space=SimpleNamespace(id="vs-test", schema="public", dimension=1536)
    )
    monkeypatch.setattr(
        "ai_core.rag.ingestion_contracts.resolve_ingestion_profile",
        lambda embedding_profile: SimpleNamespace(
            profile_id="profile-test", resolution=fake_resolution
        ),
    )


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
        structural_elements=(ParsedTextBlock(text="Body text", kind="paragraph"),),
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

    status = IngestionStatus(decision.decision)
    assert status is IngestionStatus.UPSERT
    assert decision.reason == "no_previous_hash"
    attributes = decision.attributes
    assert attributes["lifecycle_state"] is LifecycleState.ACTIVE
    assert attributes["policy_events"] == ()
    assert attributes["case_id"] == "case-7"
    assert attributes["content_hash"] == "hash123"
    chunk_meta = attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.case_id == "case-7"
    assert chunk_meta.external_id == "web::https://example.com/doc"
    assert chunk_meta.source == "crawler"
    assert chunk_meta.workflow_id == "workflow-1"
    assert chunk_meta.document_id is not None
    adapter_metadata = attributes["adapter_metadata"]
    assert adapter_metadata["canonical_source"] == "https://example.com/doc"
    assert adapter_metadata["provider"] == "web"
    provider_tags = adapter_metadata["provider_tags"]
    assert provider_tags["collection"] == "docs"
    parser_stats = adapter_metadata["parser_stats"]
    assert parser_stats["parser.token_count"] == 42
    assert adapter_metadata["language"] == "en"
    assert adapter_metadata["title"] == "Example Title"


def test_build_ingestion_decision_upserts_when_unchanged() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.UNCHANGED,
        _make_signatures("hash999"),
        3,
        "hash_match",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    status = IngestionStatus(decision.decision)
    assert status is IngestionStatus.UPSERT
    assert decision.reason == "hash_match"
    attributes = decision.attributes
    assert attributes["lifecycle_state"] is LifecycleState.ACTIVE
    chunk_meta = attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.lifecycle_state == LifecycleState.ACTIVE.value


def test_build_ingestion_decision_upserts_near_duplicate() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEAR_DUPLICATE,
        _make_signatures("hash777"),
        None,
        "near_duplicate:0.950",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    status = IngestionStatus(decision.decision)
    assert status is IngestionStatus.UPSERT
    assert decision.reason == "near_duplicate:0.950"
    assert decision.attributes["lifecycle_state"] is LifecycleState.ACTIVE
    chunk_meta = decision.attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.lifecycle_state == LifecycleState.ACTIVE.value


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

    status = IngestionStatus(decision.decision)
    assert status is IngestionStatus.RETIRE
    assert decision.reason == "retired"
    assert decision.attributes["lifecycle_state"] is LifecycleState.RETIRED
    chunk_meta = decision.attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.content_hash == "hash111"


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

    status = IngestionStatus(decision.decision)
    assert status is IngestionStatus.RETIRE
    assert decision.reason == "gone_410"
    assert decision.attributes["policy_events"] == ("gone_410",)
    assert decision.attributes["lifecycle_state"] is LifecycleState.RETIRED
    assert isinstance(decision.attributes.get("chunk_meta"), ChunkMeta)


def test_build_ingestion_error_wraps_metadata() -> None:
    document = _make_document()
    delta = DeltaDecision.from_legacy(
        DeltaStatus.NEW,
        _make_signatures("hash222"),
        1,
        "new",
    )
    decision = build_ingestion_decision(document, delta, case_id="case-5")

    error = build_ingestion_error(
        decision=decision,
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

    chunk_meta = decision.attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.external_id == normalized_source.external_id
    assert len(chunk_meta.external_id) <= MAX_EXTERNAL_ID_LENGTH

    validated = ChunkMeta.model_validate(chunk_meta.model_dump())
    assert validated.external_id == chunk_meta.external_id
