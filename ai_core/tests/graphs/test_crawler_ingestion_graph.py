from __future__ import annotations

from typing import Mapping
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from ai_core.graphs.crawler_ingestion_graph import CrawlerIngestionGraph

from ai_core.graphs.document_service import DocumentLifecycleService
from ai_core.rag.guardrails import GuardrailLimits, GuardrailSignals
from documents import api as documents_api
from documents.api import normalize_from_raw
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository


pytestmark = pytest.mark.django_db


class StubVectorClient:
    def __init__(self) -> None:
        self.upserted = []

    def upsert_chunks(self, chunks):  # type: ignore[no-untyped-def]
        chunk_list = list(chunks)
        self.upserted.extend(chunk_list)
        return len(chunk_list)


class RecordingRepository(DocumentsRepository):
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []

    def upsert(self, doc, workflow_id=None):  # type: ignore[override]
        self.upserts.append(
            {
                "document_id": str(doc.ref.document_id),
                "workflow_id": workflow_id,
            }
        )
        return doc


class FailingRepository(DocumentsRepository):
    def upsert(self, doc, workflow_id=None):  # type: ignore[override]
        raise RuntimeError("upsert_failed")


def _build_state(content: str = "Example document", **overrides) -> dict[str, object]:
    tenant_id = overrides.get("tenant_id", "tenant")
    case_id = overrides.get("case_id", "case")
    request_id = overrides.get("request_id", "req-1")
    origin_uri = overrides.get("origin_uri", "https://example.com/document")
    provider = overrides.get("provider", "web")
    content_type = overrides.get("content_type", "text/plain")
    document_id = overrides.get("document_id", f"doc-{uuid4()}")

    raw_document_override = overrides.get("raw_document")
    if raw_document_override is not None:
        raw_document = dict(raw_document_override)
        metadata = dict(raw_document.get("metadata") or {})
        metadata.setdefault("source", metadata.get("source", "crawler"))
        metadata.setdefault("origin_uri", origin_uri)
        metadata.setdefault("provider", provider)
        metadata.setdefault("content_type", content_type)
        raw_document["metadata"] = metadata
        raw_document.setdefault("document_id", document_id)
        raw_document.setdefault("content", content)
    else:
        raw_document = {
            "document_id": document_id,
            "content": content,
            "metadata": {
                "source": "crawler",
                "origin_uri": origin_uri,
                "provider": provider,
                "content_type": content_type,
            },
        }

    guardrail_overrides = overrides.get("guardrails") or {}
    if isinstance(guardrail_overrides, Mapping):
        guardrail_overrides = dict(guardrail_overrides)
    else:
        guardrail_overrides = {}

    limits_override = guardrail_overrides.get("limits")
    if not isinstance(limits_override, GuardrailLimits):
        limit_bytes = overrides.get("max_document_bytes")
        if limit_bytes is None:
            limit_bytes = guardrail_overrides.get("max_document_bytes")
        limits_override = GuardrailLimits(max_document_bytes=limit_bytes)

    raw_content = raw_document.get("content")
    if isinstance(raw_content, bytes):
        body_bytes = raw_content
    else:
        body_bytes = str(raw_content or "").encode("utf-8")

    canonical_source = raw_document.get("metadata", {}).get("origin_uri")
    host = urlparse(canonical_source).hostname if canonical_source else None
    signals_override = guardrail_overrides.get("signals")
    if not isinstance(signals_override, GuardrailSignals):
        signals_override = GuardrailSignals(
            tenant_id=tenant_id,
            provider=raw_document.get("metadata", {}).get("provider"),
            canonical_source=canonical_source,
            host=host,
            document_bytes=len(body_bytes),
            mime_type=raw_document.get("metadata", {}).get("content_type"),
        )

    error_builder_override = guardrail_overrides.get("error_builder")
    config_override = {
        key: value
        for key, value in guardrail_overrides.items()
        if key not in {"limits", "signals", "error_builder"}
    }
    if not config_override:
        config_override = None

    guardrail_context: dict[str, object] = {
        "limits": limits_override,
        "signals": signals_override,
    }
    if error_builder_override is not None:
        guardrail_context["error_builder"] = error_builder_override
    if config_override:
        guardrail_context["config"] = config_override

    state: dict[str, object] = {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "request_id": request_id,
        "raw_document": raw_document,
        "guardrails": guardrail_context,
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
    assert transitions["persist_document"]["decision"] == "persisted"
    assert transitions["trigger_embedding"]["decision"] == "embedding_triggered"
    assert transitions["finish"]["decision"] == "new"
    assert result["decision"] == "new"
    summary = updated_state["summary"]
    assert summary["delta"]["decision"] == "new"
    assert summary["embedding"]["status"] == "upserted"
    assert summary["guardrails"]["decision"] == "allow"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(
        status.to_dict().get("reason") == "no_previous_hash" for status in statuses
    )
    assert "persisted_document" in updated_state["artifacts"]


def test_guardrail_denied_short_circuits(monkeypatch) -> None:
    graph = CrawlerIngestionGraph()
    recorded_events: list[tuple[str, Mapping[str, object]]] = []

    def _record_event(name: str, payload: Mapping[str, object]) -> None:
        recorded_events.append((name, dict(payload)))

    monkeypatch.setattr(
        "ai_core.graphs.crawler_ingestion_graph.emit_event",
        _record_event,
    )
    state = _build_state(
        guardrails={"max_document_bytes": 8},
        raw_document={"content": "Very long content"},
    )
    # ensure content longer than limit to trigger denial
    state["raw_document"]["content"] = "deny" * 10  # type: ignore[index]

    updated_state, result = graph.run(state, {})

    transitions = result["transitions"]
    assert "decide_delta" not in transitions
    assert "persist_document" not in transitions
    assert "trigger_embedding" not in transitions
    assert transitions["enforce_guardrails"]["decision"] == "deny"
    assert transitions["finish"]["decision"] == "denied"
    assert result["decision"] == "denied"
    summary = updated_state["summary"]
    assert summary["guardrails"]["decision"] == "deny"
    assert "embedding" not in summary
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(
        status.to_dict().get("reason") == "document_too_large" for status in statuses
    )
    assert recorded_events == [
        (
            "crawler_guardrail_denied",
            {
                "reason": "document_too_large",
                "policy_events": ["max_document_bytes"],
            },
        )
    ]


def test_delta_unchanged_skips_embedding() -> None:
    repository = InMemoryDocumentsRepository()
    baseline_payload = normalize_from_raw(
        raw_reference={"content": "Persistent"}, tenant_id="tenant"
    )
    repository.upsert(
        baseline_payload.document,
        workflow_id=baseline_payload.document.ref.workflow_id,
    )

    graph = CrawlerIngestionGraph(repository=repository)
    client = StubVectorClient()
    state = _build_state(
        content="Persistent",
        vector_client=client,
        document_id=str(baseline_payload.document.ref.document_id),
    )

    updated_state, result = graph.run(state, {})

    summary = updated_state["summary"]
    assert "embedding" not in summary
    assert result["decision"] == "unchanged"
    transitions = result["transitions"]
    assert transitions["persist_document"]["decision"] == "persisted"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "hash_match" for status in statuses)
    baseline_state = updated_state.get("baseline")
    assert isinstance(baseline_state, dict)
    assert baseline_state.get("checksum") == baseline_payload.document.checksum


def test_repository_upsert_invoked() -> None:
    repository = RecordingRepository()
    graph = CrawlerIngestionGraph(repository=repository)
    state = _build_state()

    updated_state, result = graph.run(state, {})

    assert result["transitions"]["persist_document"]["decision"] == "persisted"
    assert repository.upserts
    assert updated_state["artifacts"].get("persisted_document")


def test_repository_upsert_failure_records_error() -> None:
    graph = CrawlerIngestionGraph(repository=FailingRepository())
    state = _build_state()

    updated_state, result = graph.run(state, {})

    assert result["decision"] == "error"
    transitions = result["transitions"]
    assert transitions["persist_document"]["decision"] == "error"
    artifacts = updated_state["artifacts"]
    assert artifacts.get("persistence_failure", {}).get("type") == "RuntimeError"
    assert artifacts.get("persistence_errors")
    statuses = artifacts.get("status_updates", [])
    assert any(
        status.to_dict().get("reason") == "persist_document_failed"
        for status in statuses
    )


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
    assert any(
        status.to_dict().get("reason") == "trigger_embedding_failed"
        for status in statuses
    )


class RecordingDocumentLifecycleService(DocumentLifecycleService):
    def __init__(self) -> None:
        self.normalize_calls: list[dict[str, object]] = []
        self.status_calls: list[dict[str, object]] = []

    def normalize_from_raw(
        self,
        *,
        raw_reference,
        tenant_id: str,
        case_id: str | None = None,
        request_id: str | None = None,
    ):
        self.normalize_calls.append(
            {
                "tenant_id": tenant_id,
                "case_id": case_id,
                "request_id": request_id,
            }
        )
        return documents_api.normalize_from_raw(
            raw_reference=raw_reference,
            tenant_id=tenant_id,
            case_id=case_id,
            request_id=request_id,
        )

    def update_lifecycle_status(
        self,
        *,
        tenant_id: str,
        document_id,
        status: str,
        previous_status: str | None = None,
        workflow_id: str | None = None,
        reason: str | None = None,
        policy_events=None,
    ):
        self.status_calls.append(
            {
                "tenant_id": tenant_id,
                "document_id": str(document_id),
                "status": status,
                "workflow_id": workflow_id,
                "reason": reason,
            }
        )
        return documents_api.update_lifecycle_status(
            tenant_id=tenant_id,
            document_id=document_id,
            status=status,
            previous_status=previous_status,
            workflow_id=workflow_id,
            reason=reason,
            policy_events=policy_events,
        )


def test_document_service_adapter_is_injected() -> None:
    service = RecordingDocumentLifecycleService()
    graph = CrawlerIngestionGraph(document_service=service)
    state = _build_state(request_id=None)

    updated_state, result = graph.run(state, {"request_id": "req-custom"})

    assert result["decision"]
    assert service.normalize_calls == [
        {"tenant_id": "tenant", "case_id": "case", "request_id": "req-custom"}
    ]
    assert service.status_calls  # status transitions flowed through the adapter
    # ensure artifacts originate from adapter results
    statuses = updated_state["artifacts"].get("status_updates", [])
    recorded_ids = {call["reason"] for call in service.status_calls if call["reason"]}
    assert recorded_ids
    assert all(status.to_dict()["reason"] in recorded_ids for status in statuses)
