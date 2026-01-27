from __future__ import annotations

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


def test_rag_evidence_returns_claim_map():
    output = execute(
        "rag.evidence",
        _tool_context(),
        RuntimeConfig(execution_scope="TENANT"),
        {
            "answer": "Alpha beta. Gamma delta.",
            "citations": [
                {"id": "c1", "snippet": "alpha something"},
                {"id": "c2", "snippet": "gamma related"},
            ],
        },
    )

    assert output.claim_to_citation["Alpha beta"] == ["c1"]
    assert output.claim_to_citation["Gamma delta"] == ["c2"]
    assert output.ungrounded_claims == []


def test_rag_evidence_deterministic_for_same_inputs():
    payload = {
        "answer": "One two. Three four.",
        "citations": [
            {"id": "a", "snippet": "one"},
            {"id": "b", "snippet": "three"},
        ],
    }
    runtime_config = RuntimeConfig(execution_scope="TENANT")
    ctx = _tool_context()

    out1 = execute("rag.evidence", ctx, runtime_config, payload)
    out2 = execute("rag.evidence", ctx, runtime_config, payload)

    assert out1.claim_to_citation == out2.claim_to_citation
    assert out1.ungrounded_claims == out2.ungrounded_claims
