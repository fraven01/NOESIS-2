from __future__ import annotations

from types import SimpleNamespace

from ai_core.agent.capabilities import registry as capability_registry
from ai_core.agent.runtime import AgentRuntime
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


def test_rag_query_flow_runs_and_produces_answer_and_claim_map(monkeypatch):
    def _fake_execute(name, _ctx, _cfg, payload):
        if name == "rag.retrieve":
            return SimpleNamespace(
                matches=[
                    {"chunk_id": "c1", "score": 0.9, "text": "alpha"},
                    {"chunk_id": "c2", "score": 0.8, "text": "beta"},
                ],
                telemetry={},
                routing=None,
            )
        if name == "rag.compose":
            return SimpleNamespace(
                answer="final answer", used_sources=["c1"], telemetry={}
            )
        if name == "rag.evidence":
            return SimpleNamespace(
                claim_to_citation={"claim": ["c1"]}, ungrounded_claims=[]
            )
        raise AssertionError(f"unexpected capability {name}")

    monkeypatch.setattr(capability_registry, "execute", _fake_execute)

    runtime = AgentRuntime()
    record = runtime.start(
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="rag_query",
        flow_input={"question": "q", "top_k": 2},
    )

    assert record["output"]["answer"] == "final answer"
    assert record["output"]["claim_to_citation"] == {"claim": ["c1"]}
    assert len(record["output"]["retrieval_matches"]) == 2
