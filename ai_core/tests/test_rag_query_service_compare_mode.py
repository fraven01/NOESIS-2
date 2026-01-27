from __future__ import annotations

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.services.rag_query import RagQueryService
from ai_core.tool_contracts.base import ToolContext


def test_rag_query_service_compare_mode_returns_two_results_and_diff(monkeypatch):
    def _fake_legacy(_state, _meta):
        return {}, {
            "answer": "legacy",
            "citations": ["c1", "c2"],
            "claim_to_citation": {"claim": ["c1"]},
        }

    def _fake_runtime_start(
        self, *, tool_context, runtime_config, flow_name, flow_input
    ):
        return {
            "run_id": "run",
            "output": {
                "answer": "runtime",
                "citations": ["c1"],
                "claim_to_citation": {"claim": ["c1"]},
            },
            "decision_log": [],
            "stop_decision": {
                "status": "succeeded",
                "reason": "ok",
                "evidence_refs": [],
            },
        }

    monkeypatch.setattr(
        "ai_core.services.rag_query.run_retrieval_augmented_generation",
        _fake_legacy,
    )
    monkeypatch.setattr(
        "ai_core.services.rag_query.AgentRuntime.start", _fake_runtime_start
    )

    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    tool_context = ToolContext(scope=scope, business=BusinessContext(), metadata={})

    service = RagQueryService()
    result = service.execute(
        tool_context=tool_context,
        question="q",
        mode="compare",
    )

    assert "legacy" in result
    assert "runtime" in result
    assert "diff" in result
    assert result["diff"]["citations_count_legacy"] == 2
    assert result["diff"]["citations_count_runtime"] == 1
