from __future__ import annotations

import hashlib

import pytest

from ai_core.rag.vector_client import (
    build_dedup_signatures,
    compute_near_duplicate_signature,
    extract_primary_text_hash,
    match_near_duplicate,
)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_build_dedup_signatures_normalizes_primary_text() -> None:
    base_text = "Hello world"
    noisy_text = "  Hello   world\n"

    expected_hash = _hash_text(base_text)
    reference = build_dedup_signatures(
        primary_text=base_text,
        normalized_primary_text=base_text,
        stored_primary_text_hash=None,
        payload_bytes=b"ignored",
    )
    noisy = build_dedup_signatures(
        primary_text=noisy_text,
        normalized_primary_text=base_text,
        stored_primary_text_hash=None,
        payload_bytes=b"ignored",
    )

    assert reference.content_hash == expected_hash
    assert noisy.content_hash == expected_hash


def test_build_dedup_signatures_uses_payload_when_text_missing() -> None:
    payload = b"binary-data"
    expected_hash = hashlib.sha256(payload).hexdigest()

    signatures = build_dedup_signatures(
        primary_text=None,
        normalized_primary_text="",
        stored_primary_text_hash=None,
        payload_bytes=payload,
    )

    assert signatures.content_hash == expected_hash


def test_build_dedup_signatures_prefers_stored_hash() -> None:
    stored_hash = _hash_text("stored")

    signatures = build_dedup_signatures(
        primary_text=None,
        normalized_primary_text="",
        stored_primary_text_hash=stored_hash,
        payload_bytes=b"ignored",
    )

    assert signatures.content_hash == stored_hash


def test_extract_primary_text_hash_validates_stats() -> None:
    stats = {"crawler.primary_text_hash_sha256": "ABCDEF"}

    assert extract_primary_text_hash(stats, "sha256") is None

    stats["crawler.primary_text_hash_sha256"] = _hash_text("payload")
    extracted = extract_primary_text_hash(stats, "sha256")

    assert extracted == _hash_text("payload")


def test_build_dedup_signatures_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported_hash_algorithm"):
        build_dedup_signatures(
            primary_text="hello",
            normalized_primary_text="hello",
            stored_primary_text_hash=None,
            payload_bytes=b"hello",
            algorithm="sha256-unknown",
        )


def test_compute_near_duplicate_signature_tokens() -> None:
    signature = compute_near_duplicate_signature("Alpha beta alpha")

    assert signature is not None
    assert signature.tokens == ("alpha", "beta")
    assert isinstance(signature.fingerprint, str)


def test_match_near_duplicate_prefers_highest_similarity() -> None:
    base = compute_near_duplicate_signature("one two three")
    competitor = compute_near_duplicate_signature("two three four")
    distant = compute_near_duplicate_signature("five six seven")
    assert base is not None and competitor is not None and distant is not None

    known = {
        "competitor": competitor,
        "distant": distant,
    }
    match = match_near_duplicate(base, known, threshold=0.1)

    assert match is not None
    assert match.document_id == "competitor"
    assert match.similarity > 0.4


def test_match_near_duplicate_respects_threshold() -> None:
    base = compute_near_duplicate_signature("one two three")
    competitor = compute_near_duplicate_signature("four five six")
    assert base is not None and competitor is not None

    match = match_near_duplicate(base, {"competitor": competitor}, threshold=0.9)

    assert match is None


def test_match_near_duplicate_excludes_same_document() -> None:
    signature = compute_near_duplicate_signature("one two three")
    assert signature is not None

    match = match_near_duplicate(
        signature,
        {"self": signature},
        threshold=0.1,
        exclude="self",
    )

    assert match is None
