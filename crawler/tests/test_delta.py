from __future__ import annotations

import hashlib

import pytest

from crawler.delta import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DeltaStatus,
    NearDuplicateSignature,
    evaluate_delta,
)
from crawler.normalizer import (
    ExternalDocumentReference,
    NormalizedDocument,
    NormalizedDocumentContent,
    NormalizedDocumentMeta,
)
from crawler.parser import ParserStats, StructuralElement


def _make_document(
    document_id: str,
    text: str | None,
    *,
    binary_ref: str | None = None,
    structural_elements: tuple[StructuralElement, ...] | None = None,
) -> NormalizedDocument:
    stats = ParserStats(
        token_count=len((text or "").split()),
        character_count=len(text or ""),
        extraction_path="html.body",
        error_fraction=0.0,
    )
    meta = NormalizedDocumentMeta(
        title="Example",
        language="en",
        tags=(),
        origin_uri="https://example.com/doc",
        media_type="text/html",
        parser_stats={"parser.token_count": stats.token_count},
    )
    content = NormalizedDocumentContent(
        primary_text=text,
        binary_payload_ref=binary_ref,
        structural_elements=structural_elements or (),
        entities=(),
    )
    external_ref = ExternalDocumentReference(
        provider="web",
        external_id=f"web::https://example.com/{document_id}",
        canonical_source="https://example.com/doc",
    )
    return NormalizedDocument(
        tenant_id="tenant",
        workflow_id="wf",
        document_id=document_id,
        meta=meta,
        content=content,
        external_ref=external_ref,
        stats=stats,
    )


def test_evaluate_delta_marks_new_document() -> None:
    document = _make_document("doc-1", "Hello world")

    decision = evaluate_delta(document)

    assert decision.status is DeltaStatus.NEW
    assert decision.version == 1
    assert decision.signatures.content_hash
    assert decision.reason == "no_previous_hash"


def test_evaluate_delta_detects_unchanged_hash() -> None:
    document = _make_document("doc-1", "Hello world")
    initial = evaluate_delta(document)

    repeat = evaluate_delta(
        document,
        previous_content_hash=initial.signatures.content_hash,
        previous_version=7,
    )

    assert repeat.status is DeltaStatus.UNCHANGED
    assert repeat.version == 7
    assert repeat.signatures.content_hash == initial.signatures.content_hash


def test_evaluate_delta_increments_version_on_change() -> None:
    old_document = _make_document("doc-1", "Hello world")
    initial = evaluate_delta(old_document)

    updated = _make_document("doc-1", "Hello brave new world")
    decision = evaluate_delta(
        updated,
        previous_content_hash=initial.signatures.content_hash,
        previous_version=2,
    )

    assert decision.status is DeltaStatus.CHANGED
    assert decision.version == 3
    assert decision.signatures.content_hash != initial.signatures.content_hash


def test_evaluate_delta_ignores_whitespace_changes() -> None:
    original = _make_document("doc-1", "Hello    world\n\n")
    baseline = evaluate_delta(original)

    noisy = _make_document("doc-1", "  Hello world")
    repeat = evaluate_delta(
        noisy,
        previous_content_hash=baseline.signatures.content_hash,
        previous_version=baseline.version,
    )

    assert repeat.status is DeltaStatus.UNCHANGED


def test_evaluate_delta_ignores_markup_only_changes() -> None:
    baseline = _make_document("doc-1", "Hello world")
    first = evaluate_delta(baseline)

    elements = (
        StructuralElement(
            kind="paragraph",
            text="Hello world",
            metadata={"tag": "strong"},
        ),
    )
    updated = _make_document(
        "doc-1",
        "Hello world",
        structural_elements=elements,
    )
    repeat = evaluate_delta(
        updated,
        previous_content_hash=first.signatures.content_hash,
        previous_version=first.version,
    )

    assert repeat.status is DeltaStatus.UNCHANGED


def test_evaluate_delta_requires_binary_payload_when_text_missing() -> None:
    document = _make_document("doc-1", None, binary_ref="blob://1")

    with pytest.raises(ValueError):
        evaluate_delta(document)


def test_evaluate_delta_hashes_binary_payload() -> None:
    document = _make_document("doc-1", None, binary_ref="blob://1")
    payload = b"binary-data"
    expected_hash = hashlib.sha256(payload).hexdigest()

    decision = evaluate_delta(document, binary_payload=payload)

    assert decision.signatures.content_hash == expected_hash
    assert decision.status is DeltaStatus.NEW


def test_evaluate_delta_flags_near_duplicate() -> None:
    canonical_tokens = ("example", "hello", "world")
    parent_signature = NearDuplicateSignature(
        fingerprint="abc",
        tokens=canonical_tokens,
    )
    document = _make_document("doc-2", "world hello example")

    decision = evaluate_delta(
        document,
        known_near_duplicates={"doc-1": parent_signature},
        near_duplicate_threshold=DEFAULT_NEAR_DUPLICATE_THRESHOLD - 0.1,
    )

    assert decision.status is DeltaStatus.NEAR_DUPLICATE
    assert decision.parent_document_id == "doc-1"
    assert decision.version is None
    assert decision.reason.startswith("near_duplicate")


def test_evaluate_delta_respects_threshold() -> None:
    parent_signature = NearDuplicateSignature(
        fingerprint="abc",
        tokens=("foo", "bar"),
    )
    document = _make_document("doc-3", "completely different text")

    decision = evaluate_delta(
        document,
        known_near_duplicates={"doc-1": parent_signature},
        near_duplicate_threshold=0.95,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None


def test_evaluate_delta_checks_near_duplicates_when_changed() -> None:
    baseline = _make_document("doc-1", "Breaking news across the world")
    first = evaluate_delta(baseline)

    updated = _make_document("doc-1", "Breaking news across the globe")
    competitor_signature = NearDuplicateSignature(
        fingerprint="xyz",
        tokens=("across", "breaking", "globe", "news", "the"),
    )

    decision = evaluate_delta(
        updated,
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
    document = _make_document("doc-42", "Short headline text")
    signature = NearDuplicateSignature(
        fingerprint="sig", tokens=("headline", "short", "text")
    )

    decision = evaluate_delta(
        document,
        known_near_duplicates={"doc-42": signature},
        near_duplicate_threshold=0.0,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None


def test_evaluate_delta_rejects_unsupported_hash_algorithm() -> None:
    document = _make_document("doc-1", "hash me")

    with pytest.raises(ValueError):
        evaluate_delta(document, hash_algorithm="not-a-real-hash")


def test_evaluate_delta_handles_minimal_text_without_false_positive() -> None:
    document = _make_document("doc-1", "Hi")
    signature = NearDuplicateSignature(fingerprint="sig", tokens=("bye",))

    decision = evaluate_delta(
        document,
        known_near_duplicates={"doc-other": signature},
        near_duplicate_threshold=0.5,
    )

    assert decision.status is DeltaStatus.NEW
    assert decision.parent_document_id is None
