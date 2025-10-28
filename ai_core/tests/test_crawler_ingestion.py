import pytest

from crawler.delta import DeltaDecision, DeltaStatus
from crawler.fetcher import (
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from crawler.ingestion import IngestionStatus, build_ingestion_decision
from crawler.normalizer import build_normalized_document
from crawler.parser import (
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)
from crawler.retire import LifecycleDecision, LifecycleState
from crawler.contracts import NormalizedSource
from documents.contracts import NormalizedDocument
from documents.parsers import ParsedTextBlock
from uuid import uuid4

from ai_core.rag.ingestion_contracts import ChunkMeta


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
        last_modified="2023-01-01T00:00:00+00:00",
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


def _make_delta(status: DeltaStatus, *, reason: str = "reason") -> DeltaDecision:
    from ai_core.rag.vector_client import DedupSignatures

    return DeltaDecision.from_legacy(
        status,
        DedupSignatures(content_hash=f"hash-{status.value}"),
        1,
        reason,
    )


def test_build_ingestion_decision_always_provides_chunk_meta() -> None:
    document = _make_document()
    delta = _make_delta(DeltaStatus.NEW, reason="new")

    decision = build_ingestion_decision(document, delta, case_id="case-7")

    assert decision.decision == IngestionStatus.UPSERT.value
    assert decision.reason == "new"
    attributes = decision.attributes
    assert attributes["lifecycle_state"] == LifecycleState.ACTIVE.value
    assert attributes["policy_events"] == ()
    assert attributes["case_id"] == "case-7"
    assert attributes["content_hash"] == "hash-new"
    chunk_meta = attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.case_id == "case-7"
    assert chunk_meta.external_id == "web::https://example.com/doc"
    assert chunk_meta.lifecycle_state == LifecycleState.ACTIVE.value
    assert attributes["delta_status"] == DeltaStatus.NEW.value


@pytest.mark.parametrize(
    "status, reason",
    [
        (DeltaStatus.UNCHANGED, "hash_match"),
        (DeltaStatus.NEAR_DUPLICATE, "near_duplicate:0.950"),
    ],
)
def test_build_ingestion_decision_upserts_for_duplicates(status, reason) -> None:
    document = _make_document()
    delta = _make_delta(status, reason=reason)

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    assert decision.decision == IngestionStatus.UPSERT.value
    assert decision.reason == reason
    chunk_meta = decision.attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.content_hash == f"hash-{status.value}"
    assert chunk_meta.lifecycle_state == LifecycleState.ACTIVE.value
    assert decision.attributes["delta_status"] == status.value


def test_build_ingestion_decision_retire_includes_metadata() -> None:
    document = _make_document()
    delta = _make_delta(DeltaStatus.UNCHANGED, reason="hash_match")
    lifecycle = LifecycleDecision(
        state=LifecycleState.RETIRED,
        reason="policy_retired",
        policy_events=("retired",),
    )

    decision = build_ingestion_decision(
        document,
        delta,
        case_id="case-9",
        retire=True,
        lifecycle=lifecycle,
    )

    assert decision.decision == IngestionStatus.RETIRE.value
    assert decision.reason == "policy_retired"
    attributes = decision.attributes
    assert attributes["lifecycle_state"] == LifecycleState.RETIRED.value
    assert attributes["policy_events"] == ("retired",)
    assert attributes["chunk_meta"] is not None
    chunk_meta = attributes["chunk_meta"]
    assert isinstance(chunk_meta, ChunkMeta)
    assert chunk_meta.lifecycle_state == LifecycleState.RETIRED.value
    assert attributes["delta_status"] == DeltaStatus.UNCHANGED.value


def test_build_ingestion_decision_requires_case_id() -> None:
    document = _make_document()
    delta = _make_delta(DeltaStatus.NEW)

    with pytest.raises(ValueError):
        build_ingestion_decision(document, delta, case_id=" ")
