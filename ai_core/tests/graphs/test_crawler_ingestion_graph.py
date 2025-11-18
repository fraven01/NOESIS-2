from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping
from urllib.parse import urlparse
from uuid import uuid4

import importlib
from typing import Any

import ai_core.api as ai_api
import pytest

import ai_core.graphs.crawler_ingestion_graph as crawler_ingestion_graph
from ai_core.contracts.payloads import CompletionPayload
from ai_core.graphs.crawler_ingestion_graph import CrawlerIngestionGraph

from ai_core.graphs.document_service import DocumentLifecycleService
from ai_core.rag.guardrails import GuardrailLimits, GuardrailSignals
from documents import api as documents_api
from documents import metrics as document_metrics
from documents.api import normalize_from_raw
from documents.contracts import DocumentBlobDescriptorV1, NormalizedDocumentInputV1
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository
from documents.pipeline import (
    DocumentChunkArtifact,
    DocumentParseArtifact,
    DocumentProcessingContext,
)


pytestmark = pytest.mark.django_db


class StubVectorClient:
    def __init__(self) -> None:
        self.upserted = []

    def upsert_chunks(self, chunks):  # type: ignore[no-untyped-def]
        chunk_list = list(chunks)
        self.upserted.extend(chunk_list)
        return len(chunk_list)


class FailingRepository(DocumentsRepository):
    def upsert(self, doc, workflow_id=None):  # type: ignore[override]
        raise RuntimeError("upsert_failed")

    def get(
        self,
        tenant_id,
        document_id,
        version=None,
        *,
        prefer_latest=False,
        workflow_id=None,
    ):
        return None


def _descriptor_from_raw(
    raw_document: Mapping[str, object],
) -> DocumentBlobDescriptorV1:
    if "payload_bytes" in raw_document:
        return DocumentBlobDescriptorV1(payload_bytes=raw_document["payload_bytes"])
    if "payload_base64" in raw_document:
        return DocumentBlobDescriptorV1(payload_base64=raw_document["payload_base64"])
    if "payload_path" in raw_document:
        return DocumentBlobDescriptorV1(
            object_store_path=str(raw_document["payload_path"])
        )
    content = raw_document.get("content")
    if isinstance(content, bytes):
        return DocumentBlobDescriptorV1(payload_bytes=content)
    return DocumentBlobDescriptorV1(inline_text=str(content or ""))


def _build_contract(
    raw_document: Mapping[str, object],
    *,
    tenant_id: str,
    case_id: str | None,
    workflow_id: str | None = None,
    source: str | None = None,
) -> NormalizedDocumentInputV1:
    metadata = dict(raw_document.get("metadata") or {})
    return NormalizedDocumentInputV1(
        tenant_id=tenant_id,
        case_id=case_id,
        workflow_id=workflow_id,
        source=source,
        metadata=metadata,
        document_id=raw_document.get("document_id"),
        blob=_descriptor_from_raw(raw_document),
    )


def _build_state(
    content: str = "Example document",
    *,
    frontier: Mapping[str, object] | None = None,
    **overrides,
) -> dict[str, object]:
    tenant_id = overrides.get("tenant_id", "tenant")
    case_id = overrides.get("case_id", "case")
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

    contract = _build_contract(
        raw_document,
        tenant_id=tenant_id,
        case_id=case_id,
        workflow_id=overrides.get("workflow_id"),
        source=raw_document.get("metadata", {}).get("source"),
    )
    normalized_payload = normalize_from_raw(contract=contract)

    state: dict[str, object] = {
        "tenant_id": tenant_id,
        "case_id": case_id,
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
        "normalized_document_input": normalized_payload.document,
    }
    state.setdefault("document_id", str(normalized_payload.document.ref.document_id))
    if frontier is not None:
        state["frontier"] = dict(frontier)
    return state


def test_orchestrates_nominal_flow() -> None:
    graph = CrawlerIngestionGraph()
    client = StubVectorClient()
    state = _build_state(vector_client=client)

    updated_state, result = graph.run(state, {})

    assert updated_state is not state
    transitions = result["transitions"]
    assert transitions["update_status_normalized"].decision == "status_updated"
    assert transitions["enforce_guardrails"].decision == "allow"
    assert transitions["document_pipeline"].decision == "processed"
    assert transitions["ingest_decision"].decision == "new"
    assert transitions["ingest"].decision == "embedding_triggered"
    assert transitions["finish"].decision == "new"
    assert result["decision"] == "new"
    summary = updated_state["summary"]
    assert isinstance(summary, CompletionPayload)
    assert summary.delta.decision == "new"
    assert summary.embedding is not None
    assert summary.embedding.status == "upserted"
    assert summary.guardrails.decision == "allow"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(
        status.to_dict().get("reason") == "no_previous_hash" for status in statuses
    )
    artifacts = updated_state["artifacts"]
    assert isinstance(
        artifacts["document_processing_context"], DocumentProcessingContext
    )
    assert isinstance(artifacts["parse_artifact"], DocumentParseArtifact)
    assert isinstance(artifacts["chunk_artifact"], DocumentChunkArtifact)
    assert artifacts["document_pipeline_phase"]
    assert artifacts["document_pipeline_run_until"] == "full"
    assert updated_state["ingest_action"] == "upsert"


def test_guardrail_denied_short_circuits(monkeypatch) -> None:
    graph = CrawlerIngestionGraph()
    recorded_events: list[tuple[str, Mapping[str, object]]] = []

    def _record_event(name: str, payload: Mapping[str, object]) -> None:
        recorded_events.append((name, dict(payload)))

    monkeypatch.setattr(
        "ai_core.graphs.crawler_ingestion_graph.emit_event",
        _record_event,
    )
    monkeypatch.setattr(
        document_metrics,
        "GUARDRAIL_DENIAL_REASON_TOTAL",
        document_metrics._FallbackCounterVec(),
    )
    state = _build_state(
        guardrails={"max_document_bytes": 8},
        raw_document={"content": "Very long content"},
    )
    # ensure content longer than limit to trigger denial
    state["raw_document"]["content"] = "deny" * 10  # type: ignore[index]

    updated_state, result = graph.run(state, {})

    transitions = result["transitions"]
    assert "document_pipeline" not in transitions
    assert "ingest_decision" not in transitions
    assert "ingest" not in transitions
    assert transitions["enforce_guardrails"].decision == "deny"
    assert transitions["finish"].decision == "denied"
    assert result["decision"] == "denied"
    summary = updated_state["summary"]
    assert isinstance(summary, CompletionPayload)
    assert summary.guardrails.decision == "deny"
    assert summary.embedding is None
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
    guardrail_counter = document_metrics.GUARDRAIL_DENIAL_REASON_TOTAL
    guardrail_decision = updated_state["artifacts"]["guardrail_decision"]
    normalized_payload = updated_state["artifacts"]["normalized_document"]

    def _label(value: str | None) -> str:
        candidate = (value or "").strip()
        return candidate or "unknown"

    expected_labels = {
        "reason": _label(guardrail_decision.reason),
        "workflow_id": _label(normalized_payload.document.ref.workflow_id),
        "tenant_id": _label(normalized_payload.tenant_id),
        "source": _label(normalized_payload.document.source),
    }
    assert guardrail_counter.value(**expected_labels) == 1.0
    assert "ingest_action" not in updated_state


def test_guardrail_denied_emits_event_callback() -> None:
    recorded_events: list[tuple[str, Mapping[str, object]]] = []

    def _capture_event(name: str, payload: Mapping[str, object]) -> None:
        recorded_events.append((name, dict(payload)))

    graph = CrawlerIngestionGraph(event_emitter=_capture_event)
    state = _build_state(
        guardrails={"max_document_bytes": 8},
        raw_document={"content": "deny" * 10},
    )
    state["raw_document"]["content"] = "deny" * 10  # type: ignore[index]

    updated_state, _ = graph.run(state, {})

    guardrail_events = [
        payload for name, payload in recorded_events if name == "guardrail_denied"
    ]
    assert guardrail_events, "guardrail_denied event not captured"
    payload = guardrail_events[0]
    assert payload["reason"] == "document_too_large"
    assert payload["policy_events"] == ["max_document_bytes"]
    normalized_payload = updated_state["artifacts"]["normalized_document"]
    assert payload["document_id"] == normalized_payload.document_id


def test_delta_unchanged_skips_embedding() -> None:
    repository = InMemoryDocumentsRepository()
    baseline_contract = NormalizedDocumentInputV1(
        tenant_id="tenant",
        metadata={"content_type": "text/plain", "provider": "crawler"},
        blob=DocumentBlobDescriptorV1(inline_text="Persistent"),
    )
    baseline_payload = normalize_from_raw(contract=baseline_contract)
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
    assert transitions["document_pipeline"].decision == "processed"
    assert transitions["ingest_decision"].decision == "unchanged"
    assert transitions["ingest"].decision == "skipped"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "hash_match" for status in statuses)
    assert updated_state["ingest_action"] == "skip"
    baseline_state = updated_state.get("baseline")
    assert isinstance(baseline_state, dict)
    assert baseline_state.get("checksum") == baseline_payload.document.checksum


def test_repository_upsert_invoked() -> None:
    repository = InMemoryDocumentsRepository()
    graph = CrawlerIngestionGraph(repository=repository)
    state = _build_state()

    updated_state, result = graph.run(state, {})

    assert result["transitions"]["document_pipeline"].decision == "processed"
    artifacts = updated_state["artifacts"]
    assert artifacts.get("document_pipeline_phase")
    normalized = artifacts["normalized_document"]
    persisted = repository.get(
        tenant_id=normalized.tenant_id,
        document_id=normalized.document.ref.document_id,
        workflow_id=normalized.document.ref.workflow_id,
    )
    assert persisted is not None


def test_repository_upsert_failure_records_error() -> None:
    graph = CrawlerIngestionGraph(repository=FailingRepository())
    state = _build_state()

    updated_state, result = graph.run(state, {})

    assert result["decision"] == "error"
    transitions = result["transitions"]
    assert transitions["document_pipeline"].decision == "error"
    artifacts = updated_state["artifacts"]
    assert artifacts.get("document_pipeline_error")
    statuses = artifacts.get("status_updates", [])
    assert any(
        status.to_dict().get("reason") == "document_pipeline_failed"
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
    assert errors and errors[0]["node"] == "ingest"
    statuses = updated_state["artifacts"].get("status_updates", [])
    assert any(status.to_dict().get("reason") == "ingest_failed" for status in statuses)


def test_html_content_is_normalized_before_chunking() -> None:
    graph = CrawlerIngestionGraph()
    client = StubVectorClient()
    html_body = """
    <html>
      <head>
        <style>body {color: red;}</style>
      </head>
      <body>
        <header>Header</header>
        <p>Visible paragraph.</p>
        <script>console.log('hidden')</script>
        <p>Trailing paragraph.</p>
      </body>
    </html>
    """

    state = _build_state(
        content=html_body,
        content_type="text/html",
        vector_client=client,
    )

    updated_state, _ = graph.run(state, {})

    artifacts = updated_state["artifacts"]
    normalized_payload = artifacts["normalized_document"]

    assert "<script>" in normalized_payload.content_raw
    assert "console.log" not in normalized_payload.content_normalized
    assert "console.log" not in normalized_payload.primary_text

    chunk_artifact = artifacts["chunk_artifact"]
    for chunk in chunk_artifact.chunks:
        text = chunk.get("text", "")
        assert "<script>" not in text
        assert "console.log" not in text

    assert client.upserted, "expected chunk upsert calls"
    for chunk in client.upserted:
        assert "<script>" not in chunk.content
        assert "console.log" not in chunk.content


class RecordingDocumentLifecycleService(DocumentLifecycleService):
    def __init__(self) -> None:
        self.normalize_calls: list[dict[str, object]] = []
        self.status_calls: list[dict[str, object]] = []

    def normalize_from_raw(
        self,
        *,
        raw_reference: Mapping[str, object],
        tenant_id: str,
        case_id: str | None = None,
        workflow_id: str | None = None,
        source: str | None = None,
    ):
        contract = _build_contract(
            raw_reference,
            tenant_id=tenant_id,
            case_id=case_id,
            workflow_id=workflow_id,
            source=source,
        )
        self.normalize_calls.append(
            {
                "tenant_id": contract.tenant_id,
                "case_id": contract.case_id,
                "workflow_id": contract.workflow_id,
                "source": contract.source,
            }
        )
        return documents_api.normalize_from_raw(contract=contract)

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
    state = _build_state()

    updated_state, result = graph.run(state, {})

    assert result["decision"]
    assert service.normalize_calls == []
    assert service.status_calls  # status transitions flowed through the adapter
    # ensure artifacts originate from adapter results
    statuses = updated_state["artifacts"].get("status_updates", [])
    recorded_ids = {call["reason"] for call in service.status_calls if call["reason"]}
    assert recorded_ids
    assert all(status.to_dict()["reason"] in recorded_ids for status in statuses)


def test_guardrail_frontier_state_propagation() -> None:
    recorded: dict[str, object] = {}

    def _stub_guardrails(**kwargs):  # type: ignore[no-untyped-def]
        recorded["frontier_state"] = kwargs.get("frontier_state")
        return ai_api.enforce_guardrails(**kwargs)

    graph = CrawlerIngestionGraph(guardrail_enforcer=_stub_guardrails)
    state = _build_state(frontier={"policy_events": ["robots_disallow"]})
    meta = {"frontier": {"slot": "default"}}

    updated_state, result = graph.run(state, meta)

    assert recorded["frontier_state"] == {
        "slot": "default",
        "policy_events": ["robots_disallow"],
    }
    guardrail_transition = result["transitions"]["enforce_guardrails"]
    guardrail_section = guardrail_transition.guardrail
    assert guardrail_section is not None
    assert guardrail_section.policy_events == ("robots_disallow",)
    summary = updated_state["summary"]
    assert isinstance(summary, CompletionPayload)
    summary_attrs = summary.guardrails.attributes
    assert summary_attrs["frontier"]["slot"] == "default"
    assert summary_attrs["frontier"]["policy_events"] == ("robots_disallow",)


def test_delta_includes_meta_frontier_backoff() -> None:
    graph = CrawlerIngestionGraph()
    scheduled_at = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    meta = {
        "frontier": {
            "earliest_visit_at": scheduled_at,
            "policy_events": ("failure_backoff",),
            "decision": "defer",
        }
    }
    state = _build_state()

    updated_state, result = graph.run(state, meta)

    delta_section = result["transitions"]["ingest_decision"].delta
    assert delta_section is not None
    delta_attrs = dict(delta_section.attributes)
    assert delta_attrs["frontier"]["earliest_visit_at"] == scheduled_at.isoformat()
    assert delta_attrs["frontier"]["decision"] == "defer"
    assert delta_attrs["policy_events"] == ("failure_backoff",)
    summary = updated_state["summary"]
    assert isinstance(summary, CompletionPayload)
    summary_attrs = summary.delta.attributes
    assert summary_attrs["frontier"]["earliest_visit_at"] == scheduled_at.isoformat()


def test_guardrail_denied_merges_frontier_policy_events() -> None:
    graph = CrawlerIngestionGraph()
    state = _build_state(
        guardrails={"max_document_bytes": 8},
        raw_document={"content": "Very long content"},
        frontier={"policy_events": ["robots_disallow"]},
    )
    state["raw_document"]["content"] = "deny" * 10  # type: ignore[index]

    updated_state, result = graph.run(state, {})

    guardrail_section = result["transitions"]["enforce_guardrails"].guardrail
    assert guardrail_section is not None
    assert guardrail_section.policy_events == (
        "max_document_bytes",
        "robots_disallow",
    )
    summary = updated_state["summary"]
    assert isinstance(summary, CompletionPayload)
    summary_attrs = summary.guardrails.attributes
    assert summary_attrs["policy_events"] == (
        "max_document_bytes",
        "robots_disallow",
    )


def test_crawler_ingestion_spans(monkeypatch) -> None:
    import ai_core.infra.observability as observability

    recorded: list[str] = []
    snapshots: list[dict[str, Any]] = []

    def fake_observe_span(name: str, **_kwargs: Any):
        def decorator(func):
            def wrapped(*args, **kwargs):
                result = func(*args, **kwargs)
                recorded.append(name)
                if (
                    name.startswith("crawler.ingestion.")
                    and name != "crawler.ingestion.run"
                    and len(args) >= 2
                    and isinstance(args[1], Mapping)
                ):
                    self = args[0]
                    state = args[1]
                    metadata = self._collect_span_metadata(state)
                    metadata["phase"] = name
                    snapshots.append(metadata)
                return result

            return wrapped

        return decorator

    monkeypatch.setattr(observability, "observe_span", fake_observe_span)
    module = importlib.reload(crawler_ingestion_graph)
    try:
        graph = module.CrawlerIngestionGraph()
        state = _build_state(vector_client=StubVectorClient())
        graph.run(state, {"trace_id": "trace-span", "workflow_id": "flow-span"})

        expected = [
            "crawler.ingestion.update_status",
            "crawler.ingestion.guardrails",
            "crawler.ingestion.document_pipeline",
            "crawler.ingestion.ingest_decision",
            "crawler.ingestion.ingest",
            "crawler.ingestion.finish",
        ]

        assert recorded[: len(expected)] == expected
        phases = [entry.get("phase") for entry in snapshots]
        assert phases == expected
        assert all(entry.get("trace_id") == "trace-span" for entry in snapshots)
        assert all(entry.get("workflow_id") == "flow-span" for entry in snapshots)
        doc_entries = [entry for entry in snapshots if entry.get("document_id")]
        assert doc_entries
        assert all(
            isinstance(entry.get("document_id"), str) and entry["document_id"]
            for entry in doc_entries
        )
        assert recorded.count("crawler.ingestion.run") == 1
        assert recorded[-1] == "crawler.ingestion.run"
    finally:
        importlib.reload(crawler_ingestion_graph)


def test_crawler_transition_metadata_contains_ids() -> None:
    graph = CrawlerIngestionGraph()
    state = _build_state(vector_client=StubVectorClient())
    meta = {"trace_id": "trace-meta", "workflow_id": "flow-meta"}

    updated_state, result = graph.run(state, meta)

    transitions = result["transitions"]
    for context in (transition.context for transition in transitions.values()):
        assert context["trace_id"] == "trace-meta"
        assert context["workflow_id"] == "flow-meta"

    normalized = updated_state["artifacts"]["normalized_document"]
    expected_document_id = str(normalized.document_id)
    doc_ids = {
        context["document_id"]
        for context in (transition.context for transition in transitions.values())
        if context.get("document_id")
    }
    assert expected_document_id in doc_ids
