import hashlib
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from crawler.contracts import NormalizedSource
from crawler.delta import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DeltaStatus,
    NearDuplicateSignature,
    evaluate_delta,
)
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
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)
from documents.contracts import NormalizedDocument
from documents.parsers import ParsedTextBlock


def _doc_id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, name))


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
    document_id: str,
    text: str | None,
    *,
    binary_ref: str | None = None,
    structural_elements: tuple[ParsedTextBlock, ...] | None = None,
) -> tuple[NormalizedDocument, ParseResult]:
    canonical_source = f"https://example.com/{document_id}"
    payload_bytes = (text.encode("utf-8") if text is not None else b"binary")
    fetch = _make_fetch_result(canonical_source, payload_bytes)
    content = ParserContent(
        media_type="text/html",
        primary_text=text,
        binary_payload_ref=binary_ref,
        title="Example",
        content_language="en",
        structural_elements=structural_elements or (),
    )
    stats = ParserStats(
        token_count=len((text or "").split()),
        character_count=len(text or ""),
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
        external_id=f"web::{document_id}",
        provider_tags={},
    )
    document = build_normalized_document(
        parse_result=parse_result,
        source=source,
        tenant_id="tenant",
        workflow_id="wf",
        document_id=document_id,
    )
    return document, parse_result


def _evaluate(
    document: NormalizedDocument,
    parse_result: ParseResult,
    **kwargs,
):
    primary_text = (
        parse_result.content.primary_text if parse_result.content is not None else None
    )
    return evaluate_delta(document, primary_text=primary_text, **kwargs)


def test_evaluate_delta_marks_new_document() -> None:
    document, parse_result = _make_document(str(uuid4()), "Hello world")

    decision = _evaluate(document, parse_result)

    assert decision.status is DeltaStatus.NEW
    assert decision.version == 1
    assert decision.signatures.content_hash
    assert decision.reason == "no_previous_hash"


def test_evaluate_delta_detects_unchanged_hash() -> None:
    document_id = str(uuid4())
    document, parse_result = _make_document(document_id, "Hello world")
    initial = _evaluate(document, parse_result)

    repeat = _evaluate(
        document,
        parse_result,
        previous_content_hash=initial.signatures.content_hash,
        previous_version=7,
    )

    assert repeat.status is DeltaStatus.UNCHANGED
    assert repeat.version == 7
    assert repeat.signatures.content_hash == initial.signatures.content_hash


def test_evaluate_delta_increments_version_on_change() -> None:
    doc_id = str(uuid4())
    old_document, old_parse = _make_document(doc_id, "Hello world")
    initial = _evaluate(old_document, old_parse)

    updated, updated_parse = _make_document(doc_id, "Hello brave new world")
    decision = _evaluate(
        updated,
        updated_parse,
        previous_content_hash=initial.signatures.content_hash,
        previous_version=2,
    )

    assert decision.status is DeltaStatus.CHANGED
    assert decision.version == 3
    assert decision.signatures.content_hash != initial.signatures.content_hash


def test_evaluate_delta_ignores_whitespace_changes() -> None:
    doc_id = str(uuid4())
    original, original_parse = _make_document(doc_id, "Hello    world\n\n")
    baseline = _evaluate(original, original_parse)

    noisy, noisy_parse = _make_document(doc_id, "  Hello world")
    repeat = _evaluate(
        noisy,
        noisy_parse,
        previous_content_hash=baseline.signatures.content_hash,
        previous_version=baseline.version,
    )

    assert repeat.status is DeltaStatus.UNCHANGED


def test_evaluate_delta_ignores_markup_only_changes() -> None:
    doc_id = str(uuid4())
    baseline, baseline_parse = _make_document(doc_id, "Hello world")
    first = _evaluate(baseline, baseline_parse)

    updated, updated_parse = _make_document(
        doc_id,
        "Hello world",
        structural_elements=(ParsedTextBlock(text="Hello world", kind="paragraph"),),
    )
    repeat = _evaluate(
        updated,
        updated_parse,
        previous_content_hash=first.signatures.content_hash,
        previous_version=first.version,
    )

    assert repeat.status is DeltaStatus.UNCHANGED


def test_evaluate_delta_hashes_binary_payload_from_document() -> None:
    doc_id = str(uuid4())
    document, parse_result = _make_document(doc_id, None, binary_ref="blob://1")

    decision = _evaluate(document, parse_result)

    assert decision.status is DeltaStatus.NEW
    assert decision.signatures.content_hash


def test_evaluate_delta_hashes_binary_payload() -> None:
    doc_id = str(uuid4())
    document, parse_result = _make_document(doc_id, None, binary_ref="blob://1")
    payload = b"binary-data"
    expected_hash = hashlib.sha256(payload).hexdigest()

    decision = _evaluate(document, parse_result, binary_payload=payload)

    assert decision.signatures.content_hash == expected_hash


def test_evaluate_delta_flags_near_duplicate() -> None:
    canonical_tokens = ("example", "hello", "world")
    parent_signature = NearDuplicateSignature(
        fingerprint="abc",
        tokens=canonical_tokens,
    )
    parent_id = str(uuid4())
    document, parse_result = _make_document(str(uuid4()), "world hello example")

    decision = _evaluate(
        document,
        parse_result,
        known_near_duplicates={parent_id: parent_signature},
        near_duplicate_threshold=DEFAULT_NEAR_DUPLICATE_THRESHOLD - 0.1,
    )

    assert decision.status is DeltaStatus.NEAR_DUPLICATE
    assert decision.parent_document_id == parent_id
    assert decision.version is None
    assert decision.reason.startswith("near_duplicate")


def test_evaluate_delta_respects_threshold() -> None:
    parent_signature = NearDuplicateSignature(
        fingerprint="abc",
        tokens=("foo", "bar"),
    )
    document, parse_result = _make_document(
        str(uuid4()), "completely different text"
    )

    parent_id = str(uuid4())
    decision = _evaluate(
        document,
        parse_result,
        known_near_duplicates={parent_id: parent_signature},
        near_duplicate_threshold=0.95,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None


def test_evaluate_delta_checks_near_duplicates_when_changed() -> None:
    base_id = _doc_id("doc-1")
    baseline, baseline_parse = _make_document(
        base_id, "Breaking news across the world"
    )
    first = _evaluate(baseline, baseline_parse)

    updated, updated_parse = _make_document(
        base_id, "Breaking news across the globe"
    )
    competitor_signature = NearDuplicateSignature(
        fingerprint="xyz",
        tokens=("across", "breaking", "globe", "news", "the"),
    )

    decision = _evaluate(
        updated,
        updated_parse,
        previous_content_hash=first.signatures.content_hash,
        previous_version=first.version,
        known_near_duplicates={"competitor": competitor_signature},
        near_duplicate_threshold=DEFAULT_NEAR_DUPLICATE_THRESHOLD - 0.05,
        check_near_duplicates_for_changes=True,
    )

    assert decision.status is DeltaStatus.NEAR_DUPLICATE
    assert decision.parent_document_id == "competitor"
    assert decision.version is None
    assert decision.reason.startswith("near_duplicate")


def test_evaluate_delta_ignores_near_duplicate_for_same_document() -> None:
    duplicate_id = _doc_id("doc-42")
    document, parse_result = _make_document(duplicate_id, "Short headline text")
    signature = NearDuplicateSignature(
        fingerprint="sig", tokens=("headline", "short", "text")
    )

    decision = _evaluate(
        document,
        parse_result,
        known_near_duplicates={duplicate_id: signature},
        near_duplicate_threshold=0.0,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None


def test_evaluate_delta_rejects_unsupported_hash_algorithm() -> None:
    hash_id = _doc_id("doc-hash")
    document, parse_result = _make_document(hash_id, "hash me")

    with pytest.raises(ValueError):
        _evaluate(document, parse_result, hash_algorithm="not-a-real-hash")


def test_evaluate_delta_handles_minimal_text_without_false_positive() -> None:
    document, parse_result = _make_document(_doc_id("doc-1"), "Hi")
    signature = NearDuplicateSignature(fingerprint="sig", tokens=("bye",))

    decision = _evaluate(
        document,
        parse_result,
        known_near_duplicates={_doc_id("doc-other"): signature},
        near_duplicate_threshold=0.5,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None
