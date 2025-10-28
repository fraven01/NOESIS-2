from __future__ import annotations

import pytest

import pytest

from uuid import uuid4

from ai_core.graphs.crawler_ingestion_graph import CrawlerIngestionGraph
from documents.api import normalize_from_raw


class StubVectorClient:
    def __init__(self) -> None:
        self.upserted = []

    def upsert_chunks(self, chunks):  # type: ignore[no-untyped-def]
        chunk_list = list(chunks)
        self.upserted.extend(chunk_list)
        return len(chunk_list)


def _build_state(content: str = "Example document", **overrides) -> dict[str, object]:
    state: dict[str, object] = {
        "tenant_id": overrides.get("tenant_id", "tenant"),
        "case_id": overrides.get("case_id", "case"),
        "request_id": overrides.get("request_id", "req-1"),
        "raw_document": {
            "document_id": overrides.get("document_id", f"doc-{uuid4()}"),
            "content": content,
            "metadata": {"source": "crawler"},
        },
        "guardrails": overrides.get("guardrails", {"max_document_bytes": 4096}),
        "baseline": overrides.get("baseline", {}),
        "embedding": overrides.get(
            "embedding",
            {
                "profile": overrides.get("embedding_profile", "standard"),
                "client": overrides.get("vector_client"),
            },
        ),
    }
    return state


def test_orchestrates_nominal_flow() -> None:
    graph = CrawlerIngestionGraph()
    client = StubVectorClient()
    state = _build_state(vector_client=client)

    updated_state, result = graph.run(state, {"request_id": "req-1"})

    assert updated_state is not state
    transitions = result["transitions"]
    assert transitions["normalize"]["decision"] == "normalized"
    assert transitions["update_status_normalized"]["decision"] == "status_updated"
    assert transitions["enforce_guardrails"]["decision"] == "allow"
    assert transitions["decide_delta"]["decision"] == "new"
    assert transitions["trigger_embedding"]["decision"] == "embedding_triggered"
    assert transitions["finish"]["decision"] == "new"
    assert result["decision"] == "new"
    summary = updated_state["summary"]
    assert summary["delta"]["decision"] == "new"
    assert summary["embedding"]["status"] == "upserted"
    assert summary["guardrails"]["decision"] == "allow"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "no_previous_hash" for status in statuses)


def test_guardrail_denied_short_circuits() -> None:
    graph = CrawlerIngestionGraph()
    state = _build_state(
        guardrails={"max_document_bytes": 8},
        raw_document={"content": "Very long content"},
    )
    # ensure content longer than limit to trigger denial
    state["raw_document"]["content"] = "deny" * 10  # type: ignore[index]

    updated_state, result = graph.run(state, {})

    transitions = result["transitions"]
    assert "decide_delta" not in transitions
    assert "trigger_embedding" not in transitions
    assert transitions["enforce_guardrails"]["decision"] == "deny"
    assert transitions["finish"]["decision"] == "denied"
    assert result["decision"] == "denied"
    summary = updated_state["summary"]
    assert summary["guardrails"]["decision"] == "deny"
    assert "embedding" not in summary
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "document_too_large" for status in statuses)


def test_delta_unchanged_skips_embedding() -> None:
    graph = CrawlerIngestionGraph()
    baseline_doc = normalize_from_raw(
        raw_reference={"content": "Persistent"}, tenant_id="tenant"
    )
    client = StubVectorClient()
    state = _build_state(
        content="Persistent",
        baseline={"checksum": baseline_doc.checksum},
        vector_client=client,
    )

    updated_state, result = graph.run(state, {})

    summary = updated_state["summary"]
    assert "embedding" not in summary
    assert result["decision"] == "unchanged"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "hash_match" for status in statuses)


@pytest.mark.parametrize(
    "content,term",
    [
        ("Forbidden topic included", "forbidden"),
        ("Another BLOCKED item", "blocked"),
    ],
)
def test_guardrail_banned_terms(content: str, term: str) -> None:
    graph = CrawlerIngestionGraph()
    state = _build_state(content=content, guardrails={"banned_terms": [term]})

    updated_state, result = graph.run(state, {})

    assert result["decision"] == "denied"


def test_embedding_failure_marks_error_and_status() -> None:
    class FailingClient(StubVectorClient):
        def upsert_chunks(self, chunks):  # type: ignore[no-untyped-def]
            raise RuntimeError("vector_down")

    graph = CrawlerIngestionGraph()
    state = _build_state(vector_client=FailingClient())

    updated_state, result = graph.run(state, {})

    assert result["decision"] == "error"
    errors = updated_state["artifacts"].get("errors") or []
    assert errors and errors[0]["node"] == "trigger_embedding"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "trigger_embedding_failed" for status in statuses)
