from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ai_core.rag.delta import DeltaDecision, DeltaStatus, evaluate_delta
from ai_core.rag.vector_client import DedupSignatures
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
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


def _make_document(text: str = "Body text") -> NormalizedDocument:
    """Build a minimal NormalizedDocument for delta tests."""
    doc_id = uuid4()
    payload = text.encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    checksum = hashlib.sha256(payload).hexdigest()

    ref = DocumentRef(
        tenant_id="tenant-a",
        workflow_id="wf-1",
        document_id=doc_id,
    )

    meta = DocumentMeta(
        tenant_id="tenant-a",
        workflow_id="wf-1",
        title="Title",
        language="en",
        origin_uri="https://example.com/doc",
        tags=[],
        external_ref={
            "provider": "web",
            "external_id": "web::https://example.com/doc",
        },
    )

    blob = InlineBlob(
        type="inline",
        media_type="text/html",
        base64=encoded,
        sha256=checksum,
        size=len(payload),
    )

    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="crawler",
        lifecycle_state="active",
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
