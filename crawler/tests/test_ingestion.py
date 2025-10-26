import pytest

from ai_core.rag.ingestion_contracts import ChunkMeta

from crawler.contracts import MAX_EXTERNAL_ID_LENGTH, normalize_source
from crawler.delta import DeltaDecision, DeltaSignatures, DeltaStatus
from crawler.errors import ErrorClass
from crawler.ingestion import (
    IngestionStatus,
    build_ingestion_decision,
    build_ingestion_error,
)
from crawler.retire import LifecycleDecision, LifecycleState
from crawler.normalizer import (
    ExternalDocumentReference,
    NormalizedDocument,
    NormalizedDocumentContent,
    NormalizedDocumentMeta,
)
from crawler.parser import ParserStats


def _make_document(
    *,
    canonical_source: str = "https://example.com/doc",
    external_id: str = "web::https://example.com/doc",
) -> NormalizedDocument:
    stats = ParserStats(
        token_count=42,
        character_count=128,
        extraction_path="html.body",
        error_fraction=0.0,
    )
    meta = NormalizedDocumentMeta(
        title=" Example Title ",
        language=" en ",
        tags=("alpha", "beta"),
        origin_uri=canonical_source,
        media_type="text/html",
        parser_stats={"parser.token_count": stats.token_count},
    )
    content = NormalizedDocumentContent(
        primary_text="Body text",
        binary_payload_ref=None,
        structural_elements=(),
        entities=(),
    )
    external_ref = ExternalDocumentReference(
        provider="web",
        external_id=external_id,
        canonical_source=canonical_source,
        provider_tags={"collection": "docs"},
    )
    return NormalizedDocument(
        tenant_id="tenant-a",
        workflow_id="workflow-1",
        document_id="doc-1",
        meta=meta,
        content=content,
        external_ref=external_ref,
        stats=stats,
    )


def _make_signatures(content_hash: str) -> DeltaSignatures:
    return DeltaSignatures(content_hash=content_hash)


def test_build_ingestion_decision_upsert_includes_metadata() -> None:
    document = _make_document()
    delta = DeltaDecision(
        status=DeltaStatus.NEW,
        signatures=_make_signatures("hash123"),
        version=1,
        reason="no_previous_hash",
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
    delta = DeltaDecision(
        status=DeltaStatus.UNCHANGED,
        signatures=_make_signatures("hash999"),
        version=3,
        reason="hash_match",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    assert decision.status is IngestionStatus.SKIP
    assert decision.payload is None
    assert decision.reason == "hash_match"
    assert decision.lifecycle_state is LifecycleState.ACTIVE


def test_build_ingestion_decision_skips_near_duplicate() -> None:
    document = _make_document()
    delta = DeltaDecision(
        status=DeltaStatus.NEAR_DUPLICATE,
        signatures=_make_signatures("hash777"),
        version=None,
        reason="near_duplicate:0.950",
    )

    decision = build_ingestion_decision(document, delta, case_id="case-1")

    assert decision.status is IngestionStatus.SKIP
    assert decision.payload is None
    assert decision.reason == "near_duplicate:0.950"
    assert decision.lifecycle_state is LifecycleState.ACTIVE


def test_build_ingestion_decision_requires_case_id() -> None:
    document = _make_document()
    delta = DeltaDecision(
        status=DeltaStatus.NEW,
        signatures=_make_signatures("hash000"),
        version=1,
        reason="new",
    )

    with pytest.raises(ValueError):
        build_ingestion_decision(document, delta, case_id=" ")


def test_build_ingestion_decision_retire_overrides_delta() -> None:
    document = _make_document()
    delta = DeltaDecision(
        status=DeltaStatus.UNCHANGED,
        signatures=_make_signatures("hash111"),
        version=5,
        reason="hash_match",
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
    delta = DeltaDecision(
        status=DeltaStatus.CHANGED,
        signatures=_make_signatures("hash333"),
        version=2,
        reason="hash_mismatch",
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
    delta = DeltaDecision(
        status=DeltaStatus.NEW,
        signatures=_make_signatures("hash222"),
        version=1,
        reason="new",
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
    delta = DeltaDecision(
        status=DeltaStatus.NEW,
        signatures=_make_signatures("hash555"),
        version=1,
        reason="no_previous_hash",
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
