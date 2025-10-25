from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ai_core import services
from ai_core.graph.schemas import ToolContext


class _DummyCheckpointer:
    def __init__(self) -> None:
        self.saved: list[tuple[Any, Any]] = []

    def load(self, ctx):  # type: ignore[no-untyped-def]
        return {}

    def save(self, ctx, state):  # type: ignore[no-untyped-def]
        self.saved.append((ctx, state))


class _DummyRunner:
    def __init__(self, ledger_meta: dict[str, Any]):
        self.ledger_meta = ledger_meta

    def run(self, state, meta):  # type: ignore[no-untyped-def]
        services.ledger.record(dict(self.ledger_meta))
        return {"state": "updated"}, {"ok": True}


@pytest.mark.django_db
def test_execute_graph_emits_cost_summary_and_updates_observation(monkeypatch):
    request = SimpleNamespace(
        headers={"Content-Type": "application/json"},
        META={},
        body=json.dumps({}).encode(),
    )

    tool_context = ToolContext(
        tenant_id="tenant-123",
        case_id="case-456",
        trace_id="trace-abc",
    )
    normalized_meta = {
        "tenant_id": "tenant-123",
        "case_id": "case-456",
        "trace_id": "trace-abc",
        "graph_name": "custom.graph",
        "graph_version": "v7",
        "tool_context": tool_context,
        "ledger": {"id": "ledger-run-1"},
        "cost": {"total_usd": 0.05},
    }

    dummy_checkpointer = _DummyCheckpointer()
    ledger_calls: list[dict[str, Any]] = []
    observation_calls: list[dict[str, Any]] = []
    emitted_events: list[tuple[str, dict[str, Any]]] = []

    def fake_normalize_meta(_request):  # type: ignore[no-untyped-def]
        return dict(normalized_meta)

    def fake_update_observation(**kwargs):  # type: ignore[no-untyped-def]
        observation_calls.append(kwargs)

    def fake_emit_event(name, payload=None):  # type: ignore[no-untyped-def]
        emitted_events.append((name, dict(payload or {})))

    def fake_start_trace(**kwargs):  # type: ignore[no-untyped-def]
        return None

    def fake_end_trace():  # type: ignore[no-untyped-def]
        return None

    def fake_ledger_record(meta):  # type: ignore[no-untyped-def]
        ledger_calls.append(dict(meta))

    monkeypatch.setattr(services, "_normalize_meta", fake_normalize_meta)
    monkeypatch.setattr(services, "_get_checkpointer", lambda: dummy_checkpointer)
    monkeypatch.setattr(services.ledger, "record", fake_ledger_record)
    monkeypatch.setattr(services, "update_observation", fake_update_observation)
    monkeypatch.setattr(services, "emit_event", fake_emit_event)
    monkeypatch.setattr(services, "lf_tracing_enabled", lambda: True)
    monkeypatch.setattr(services, "lf_start_trace", fake_start_trace)
    monkeypatch.setattr(services, "lf_end_trace", fake_end_trace)

    runner = _DummyRunner(
        {
            "label": "llm-step",
            "model": "gpt-test",
            "usage": {"cost": {"usd": 0.07}},
            "id": "ledger-entry-1",
        }
    )

    response = services.execute_graph(request, runner)

    assert response.data == {"ok": True}
    assert ledger_calls, "ledger record should be invoked by the runner"

    initial_observation = next(
        call
        for call in observation_calls
        if call.get("metadata", {}).get("ledger.id") == "ledger-run-1"
    )
    metadata = initial_observation["metadata"]
    assert metadata["tenant.id"] == "tenant-123"
    assert metadata["case.id"] == "case-456"
    assert metadata["graph.version"] == "v7"
    assert metadata["cost.total_usd"] == pytest.approx(0.05)

    final_metadata = [
        call["metadata"]
        for call in observation_calls
        if "metadata" in call and "cost.total_usd" in call["metadata"]
    ][-1]
    assert final_metadata["cost.total_usd"] == pytest.approx(0.12)

    event_name, payload = emitted_events[-1]
    assert event_name == "cost.summary"
    assert payload["total_usd"] == pytest.approx(0.12)
    assert payload["tenant_id"] == "tenant-123"
    assert payload["case_id"] == "case-456"
    assert payload["graph_version"] == "v7"
    assert payload["components"][0]["source"] == "meta"
    assert payload["components"][1]["ledger_entry_id"] == "ledger-entry-1"
    reconciliation = payload.get("reconciliation")
    assert reconciliation["ledger_id"] == "ledger-run-1"
    assert "ledger-entry-1" in reconciliation["entry_ids"]
