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

    assert result.answer == "plain answer"
    assert calls[0][0] == "v2"
    assert calls[0][1] == {"type": "json_object"}
    assert calls[1][0] == "v1"
