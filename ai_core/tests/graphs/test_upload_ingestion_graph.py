from __future__ import annotations

import importlib
from collections.abc import Mapping

from ai_core import api as ai_core_api
import ai_core.graphs.upload_ingestion_graph as upload_ingestion_graph
from documents.api import NormalizedDocumentPayload


def _scanner(
    _: bytes, __: dict[str, object]
) -> upload_ingestion_graph.GraphTransition:
    return upload_ingestion_graph.GraphTransition(decision="proceed", reason="clean")


def test_nominal_run() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph(quarantine_scanner=_scanner)
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert result["decision"] == "completed"
    transitions = result["transitions"]
    assert transitions["accept_upload"]["decision"] == "accepted"
    assert transitions["deduplicate"]["decision"] == "proceed"
    assert transitions["parse"]["decision"] == "parse_complete"
    assert transitions["normalize"]["decision"] == "normalize_complete"
    assert transitions["persist_document"]["decision"] == "persist_complete"
    assert result["document_id"]


def test_run_until_stops_after_marker() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
    }

    result = graph.run(payload, run_until="parse_complete")

    assert "chunk_and_embed" not in result["transitions"]
    assert result["transitions"]["parse"]["decision"] == "parse_complete"


def test_duplicate_upload_skipped() -> None:
    graph = upload_ingestion_graph.UploadIngestionGraph()
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
    }

    first = graph.run(payload)
    second = graph.run(payload)

    assert first["decision"] == "completed"
    assert second["decision"] == "skip_duplicate"
    assert second["transitions"]["deduplicate"]["decision"] == "skip_duplicate"


def test_guardrail_allow_and_delta_merge_policy_events() -> None:
    guardrail_calls: list[NormalizedDocumentPayload] = []
    baseline_calls: list[Mapping[str, object]] = []

    def guardrail_stub(
        *, normalized_document: NormalizedDocumentPayload, **_: object
    ) -> ai_core_api.GuardrailDecision:
        guardrail_calls.append(normalized_document)
        return ai_core_api.GuardrailDecision(
            "allow",
            "guardrail_ok",
            {"policy_events": ("guardrail_allow",)},
        )

    def delta_stub(
        *,
        normalized_document: NormalizedDocumentPayload,
        baseline: Mapping[str, object],
        **_: object,
    ) -> ai_core_api.DeltaDecision:
        assert guardrail_calls and normalized_document is guardrail_calls[0]
        baseline_calls.append(baseline)
        return ai_core_api.DeltaDecision(
            "new",
            "delta_new",
            {"policy_events": ("delta_new",), "version": 1},
        )

    graph = upload_ingestion_graph.UploadIngestionGraph(
        guardrail_enforcer=guardrail_stub,
        delta_decider=delta_stub,
    )
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert guardrail_calls
    assert isinstance(guardrail_calls[0], NormalizedDocumentPayload)
    assert baseline_calls and isinstance(baseline_calls[0], Mapping)
    transition = result["transitions"]["delta_and_guardrails"]
    assert transition["decision"] == "upsert"
    diagnostics = transition["diagnostics"]
    assert set(diagnostics["policy_events"]) == {"guardrail_allow", "delta_new"}
    assert diagnostics["guardrail"]["decision"] == "guardrail_allow"
    assert diagnostics["version"] == 1


def test_guardrail_deny_short_circuits_delta() -> None:
    guardrail_calls: list[NormalizedDocumentPayload] = []
    delta_called = False

    def guardrail_stub(
        *, normalized_document: NormalizedDocumentPayload, **_: object
    ) -> ai_core_api.GuardrailDecision:
        guardrail_calls.append(normalized_document)
        return ai_core_api.GuardrailDecision(
            "deny",
            "blocked",
            {"policy_events": ("upload_blocked",), "severity": "error"},
        )

    def delta_stub(**_: object) -> ai_core_api.DeltaDecision:  # pragma: no cover - should not run
        nonlocal delta_called
        delta_called = True
        return ai_core_api.DeltaDecision("new", "should_not_happen", {})

    graph = upload_ingestion_graph.UploadIngestionGraph(
        guardrail_enforcer=guardrail_stub,
        delta_decider=delta_stub,
    )
    payload = {
        "tenant_id": "tenant-a",
        "uploader_id": "user-1",
        "file_bytes": b"Hello world",
        "filename": "example.txt",
    }

    result = graph.run(payload)

    assert guardrail_calls and isinstance(guardrail_calls[0], NormalizedDocumentPayload)
    assert not delta_called
    assert result["decision"] == "skip_guardrail"
    transition = result["transitions"]["delta_and_guardrails"]
    assert transition["decision"] == "skip_guardrail"
    diagnostics = transition["diagnostics"]
    assert diagnostics["policy_events"] == ("upload_blocked",)
    assert diagnostics["allowed"] is False


def test_upload_ingestion_spans(monkeypatch) -> None:
    import ai_core.infra.observability as observability

    recorded: list[str] = []

    def fake_observe_span(name: str):
        def decorator(func):
            def wrapped(*args, **kwargs):
                recorded.append(name)
                return func(*args, **kwargs)

            return wrapped

        return decorator

    monkeypatch.setattr(observability, "observe_span", fake_observe_span)
    module = importlib.reload(upload_ingestion_graph)
    try:
        graph = module.UploadIngestionGraph(quarantine_scanner=_scanner)
        payload = {
            "tenant_id": "tenant-a",
            "uploader_id": "user-1",
            "file_bytes": b"Hello world",
            "filename": "example.txt",
        }

        graph.run(payload)

        expected = {
            "upload.ingestion.accept_upload",
            "upload.ingestion.quarantine_scan",
            "upload.ingestion.deduplicate",
            "upload.ingestion.parse",
            "upload.ingestion.normalize",
            "upload.ingestion.delta_and_guardrails",
            "upload.ingestion.persist_document",
            "upload.ingestion.chunk_and_embed",
            "upload.ingestion.lifecycle_hook",
            "upload.ingestion.finalize",
        }

        assert expected.issubset(set(recorded))
    finally:
        importlib.reload(upload_ingestion_graph)
