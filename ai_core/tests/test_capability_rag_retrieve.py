from __future__ import annotations

from types import SimpleNamespace

from ai_core.agent.capabilities import rag_retrieve
from ai_core.agent.capabilities.registry import execute
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    return ToolContext(scope=scope, business=BusinessContext(), metadata={})


def test_rag_retrieve_happy_path_returns_matches(monkeypatch):
    def _fake_retrieve(_context, _params):
        return SimpleNamespace(
            matches=[
                {"chunk_id": "c1", "score": 0.9, "text": "hello"},
                {"source_id": "s2", "score": 0.5, "snippet": "world"},
            ],
            meta=None,
        )

    monkeypatch.setattr(rag_retrieve, "_retrieve", _fake_retrieve)

    output = execute(
        "rag.retrieve",
        _tool_context(),
        RuntimeConfig(execution_scope="TENANT"),
        {"query": "q"},
    )

    assert len(output.matches) == 2
    assert output.matches[0]["chunk_id"] == "c1"
    assert output.matches[1]["source_id"] == "s2"


def test_rag_retrieve_validates_input_and_clamps_top_k(monkeypatch):
    seen = {}

    def _fake_retrieve(_context, _params):
        seen["top_k"] = _params.top_k
        return SimpleNamespace(matches=[], meta=None)

    monkeypatch.setattr(rag_retrieve, "_retrieve", _fake_retrieve)

    execute(
        "rag.retrieve",
        _tool_context(),
        RuntimeConfig(execution_scope="TENANT"),
        {"query": "q", "top_k": 0},
    )
    assert seen["top_k"] == 1

    execute(
        "rag.retrieve",
        _tool_context(),
        RuntimeConfig(execution_scope="TENANT"),
        {"query": "q", "top_k": 100},
    )
    assert seen["top_k"] == 50
