from __future__ import annotations

from uuid import uuid4

import pytest

from common import logging as common_logging

from ai_core import api as ai_api
from ai_core.rag.delta import DeltaDecision
from ai_core.rag.vector_client import DedupSignatures
from documents.api import normalize_from_raw


class _CapturingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def info(self, event: str, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.records.append({"event": event, "args": args, "kwargs": kwargs})


@pytest.mark.django_db
def test_decide_delta_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    normalized = normalize_from_raw(
        raw_reference={
            "document_id": str(uuid4()),
            "content": "Updated document body",
            "metadata": {
                "source": "crawler",
                "origin_uri": "https://example.com/resource",
                "provider": "web",
                "content_type": "text/plain",
            },
        },
        tenant_id="tenant-xyz",
        case_id="case-123",
        request_id="req-456",
    )

    baseline = {
        "content_hash": "old-hash",
        "primary_text": "Previous text",
        "payload_bytes": b"old-bytes",
        "version": "2",
    }

    delta = DeltaDecision(
        "changed",
        "hash_mismatch",
        {"signatures": DedupSignatures(content_hash="new-hash"), "version": 3},
    )

    monkeypatch.setattr(ai_api, "evaluate_delta", lambda *_, **__: delta)

    capturing_logger = _CapturingLogger()
    monkeypatch.setattr(ai_api, "logger", capturing_logger)

    frontier_state = {"policy_events": ["frontier_event"], "trace_id": "trace-frontier"}

    with common_logging.log_context(trace_id="trace-123", span_id="span-456"):
        result = ai_api.decide_delta(
            normalized_document=normalized,
            baseline=baseline,
            frontier_state=frontier_state,
        )

    assert result.decision == "changed"
    assert capturing_logger.records, "decide_delta should emit a structured log entry"

    record = capturing_logger.records[0]
    assert record["event"] == "crawler.decide_delta"
    extra = record["kwargs"].get("extra")  # type: ignore[assignment]
    assert isinstance(extra, dict)

    assert extra["decision"] == "changed"
    assert extra["reason"] == "hash_mismatch"
    assert extra["tenant_id"] == "tenant-xyz"
    assert extra["document_id"] == normalized.document_id
    assert extra["case_id"] == "case-123"
    assert extra["request_id"] == "req-456"
    assert extra["trace_id"] == "trace-123"
    assert "span_id" not in extra
    assert extra["content_hash"] == "new-hash"
    assert extra["baseline_content_hash"] == "old-hash"
    assert extra["version"] == 3
    assert extra["baseline_version"] == 2
    assert extra["changed_fields"] == (
        "primary_text",
        "payload_bytes",
        "content_hash",
        "version",
    )
    assert extra["policy_events"] == ("frontier_event",)
    assert extra["frontier"]["trace_id"] == "trace-frontier"
