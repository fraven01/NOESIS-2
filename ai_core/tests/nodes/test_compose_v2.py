from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.nodes import compose


def _tool_context(*, tenant_id: str, case_id: str | None, run_id: str):
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id="trace-1",
        invocation_id="inv-1",
        run_id=run_id,
        service_id="test-worker",
    )
    business = BusinessContext(case_id=case_id)
    return scope.to_tool_context(business=business)


def test_compose_falls_back_to_v1_on_invalid_json(monkeypatch):
    calls = []

    def fake_call(label, prompt, metadata, **kwargs):
        calls.append((metadata.get("prompt_version"), kwargs.get("response_format")))
        if len(calls) == 1:
            return {"text": "not-json", "usage": {}, "model": "m"}
        return {"text": "plain answer", "usage": {}, "model": "m"}

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    params = compose.ComposeInput(question="Q?", snippets=[])
    result = compose.run(context, params)

    assert result.answer == "not-json"
    assert calls[0][0] == "v2"
    # Result of Step 219: we removed the explicit json_object mode
    assert calls[0][1] is None
    # We no longer retry with v1, we just return the raw text if no structure found
    assert len(calls) == 1


def test_compose_handles_tag_format(monkeypatch):
    tag_payload = """
<thought>Analysis here.</thought>
<answer>Final answer.</answer>
<meta>{"used_sources": [{"id": "s1", "label": "L1", "relevance_score": 0.5}], "suggested_followups": ["F1"]}</meta>
"""

    def fake_call(label, prompt, metadata, **kwargs):
        return {"text": tag_payload, "usage": {}, "model": "m"}

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-2")
    params = compose.ComposeInput(question="Q?", snippets=[])
    result = compose.run(context, params)

    assert result.answer == "Final answer."
    assert result.reasoning.analysis == "Analysis here."
    assert len(result.used_sources) == 1
    assert result.used_sources[0].label == "L1"
    assert result.suggested_followups == ["F1"]
