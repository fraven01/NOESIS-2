from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_core.agent.capabilities import rag_compose
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


def test_rag_compose_returns_answer(monkeypatch):
    def _fake_compose(_context, _params):
        return SimpleNamespace(
            answer="ok",
            used_sources=["s1"],
            debug_meta={"t": 1},
        )

    monkeypatch.setattr(rag_compose, "_compose", _fake_compose)

    output = execute(
        "rag.compose",
        _tool_context(),
        RuntimeConfig(execution_scope="TENANT"),
        {"question": "q", "snippets": [{"text": "t", "source": "s1"}]},
    )

    assert output.answer == "ok"
    assert output.used_sources == ["s1"]
    assert output.telemetry == {"t": 1}


def test_rag_compose_rejects_empty_question():
    with pytest.raises(ValueError):
        execute(
            "rag.compose",
            _tool_context(),
            RuntimeConfig(execution_scope="TENANT"),
            {"question": " ", "snippets": []},
        )
