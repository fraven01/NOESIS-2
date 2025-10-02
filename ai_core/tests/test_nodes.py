import pytest

from ai_core.infra.mask_prompt import mask_prompt
from ai_core.infra.prompts import load
from ai_core.nodes import (
    retrieve,
    compose,
    extract,
    classify,
    assess,
    draft_blocks,
    needs,
)
from ai_core.nodes._prompt_runner import run_prompt_node
from ai_core.rag.schemas import Chunk

META = {
    "tenant": "t1",
    "case": "c1",
    "trace_id": "tr",
    "tenant_schema": "tenant-schema-1",
}


def test_retrieve_returns_snippets(monkeypatch):
    class _Router:
        def __init__(self, chunks):
            self._chunks = chunks
            self.last_call = {}

        def search(
            self,
            query,
            tenant_id,
            *,
            case_id=None,
            top_k=5,
            filters=None,
        ):
            self.last_call = {
                "tenant_id": tenant_id,
                "case_id": case_id,
                "top_k": top_k,
                "filters": filters,
            }
            return self._chunks

    chunk = Chunk(
        "Hello 123",
        {
            "tenant": "t1",
            "case": "c1",
            "source": "s1",
            "hash": "h",
            "id": "doc-1",
            "score": 0.42,
            "category": "demo",
        },
    )
    router = _Router([chunk])
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)
    state, result = retrieve.run({"query": "Hello"}, META.copy(), top_k=3)
    snippet = result["snippets"][0]
    assert snippet == {
        "text": "Hello 123",
        "source": "s1",
        "score": 0.42,
        "hash": "h",
        "id": "doc-1",
        "meta": {"tenant": "t1", "case": "c1", "category": "demo"},
    }
    assert state["snippets"] == result["snippets"]
    assert router.last_call == {
        "tenant_id": "t1",
        "case_id": "c1",
        "top_k": 3,
        "filters": {"tenant": "t1", "case": "c1"},
    }


@pytest.mark.parametrize("scoped_router", [False, True])
def test_retrieve_snippets_shape(monkeypatch, scoped_router):
    retrieve._reset_router_for_tests()

    chunk = Chunk(
        "Body",
        {
            "source": "src",
            "score": 0.9,
            "hash": "hash-1",
            "id": "doc-1",
            "custom_kv": "value",
        },
    )

    class _BaseRouter:
        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append({"query": query, **kwargs})
            return [chunk]

    base_router = _BaseRouter()

    if scoped_router:

        class _ScopedRouter:
            def __init__(self, inner):
                self.inner = inner
                self.tenants = []

            def for_tenant(self, tenant_id, tenant_schema=None):
                self.tenants.append((tenant_id, tenant_schema))
                return self.inner

        router = _ScopedRouter(base_router)
    else:
        router = base_router

    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)

    state = {"query": "hello"}
    meta = {
        "tenant": "tenant-1",
        "case": "case-1",
        "tenant_schema": "tenant-1-schema",
    }
    _, result = retrieve.run(state, meta, top_k=2)

    assert result["snippets"], "expected snippets in result"
    snippet = result["snippets"][0]
    assert snippet == {
        "text": "Body",
        "source": "src",
        "score": 0.9,
        "hash": "hash-1",
        "id": "doc-1",
        "meta": {"custom_kv": "value"},
    }

    if scoped_router:
        assert router.tenants == [("tenant-1", "tenant-1-schema")]
        assert base_router.calls[0]["filters"] == {"case": "case-1"}
    else:
        assert base_router.calls[0]["filters"] == {
            "tenant": "tenant-1",
            "case": "case-1",
        }


def test_retrieve_requires_tenant_id(monkeypatch):
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: None)
    with pytest.raises(ValueError, match="tenant_id required"):
        retrieve.run({"query": "Hello"}, {"case": "c1"})


def _mock_call(called):
    def inner(label, prompt, metadata, **kwargs):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        return {"text": "resp", "usage": {}, "model": "m"}

    return inner


def test_compose_masks_and_sets_version(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {
        "question": "Number 1234?",
        "snippets": [{"text": "Answer", "source": "s"}],
    }
    prompt = load("retriever/answer")
    snippets_text = "\n".join(s.get("text", "") for s in state["snippets"])
    expected_prompt = mask_prompt(
        f"{prompt['text']}\n\nQuestion: {state['question']}\nContext:\n{snippets_text}"
    )
    new_state, result = compose.run(state, META.copy())
    assert called["label"] == "synthesize"
    assert called["prompt"] == expected_prompt
    assert called["meta"]["prompt_version"] == "v1"
    assert new_state["answer"] == "resp"
    assert result["answer"] == "resp"


def test_extract_classify_assess(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {"text": "Fact 42"}
    extract_prompt = mask_prompt(f"{load('extract/items')['text']}\n\n{state['text']}")
    new_state, _ = extract.run(state, META.copy())
    assert called["label"] == "extract"
    assert called["prompt"] == extract_prompt
    assert new_state["items"] == "resp"

    classify_prompt = mask_prompt(
        f"{load('classify/mitbestimmung')['text']}\n\n{state['text']}"
    )
    new_state, _ = classify.run(state, META.copy())
    assert called["label"] == "classify"
    assert called["prompt"] == classify_prompt
    assert new_state["classification"] == "resp"

    assess_prompt = mask_prompt(f"{load('assess/risk')['text']}\n\n{state['text']}")
    new_state, _ = assess.run(state, META.copy())
    assert called["label"] == "analyze"
    assert called["prompt"] == assess_prompt
    assert new_state["risk"] == "resp"


def test_prompt_runner_default(monkeypatch):
    called = {}

    def fake_load(alias):
        called["alias"] = alias
        return {"text": "Prompt", "version": "v42"}

    def fake_mask(value, **kwargs):
        called["masked"] = value
        return value

    def fake_call(label, prompt, metadata):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        return {"text": "resp"}

    monkeypatch.setattr("ai_core.nodes._prompt_runner.load", fake_load)
    monkeypatch.setattr("ai_core.nodes._prompt_runner.mask_prompt", fake_mask)
    monkeypatch.setattr("ai_core.nodes._prompt_runner.client.call", fake_call)

    state = {"text": "Sensitive"}
    meta = META.copy()
    new_state, node_meta = run_prompt_node(
        trace_name="unit",
        prompt_alias="scope/test",
        llm_label="label",
        state_key="result",
        state=state,
        meta=meta,
    )

    assert called["alias"] == "scope/test"
    assert called["label"] == "label"
    assert called["prompt"].endswith("\n\nSensitive")
    assert called["meta"]["prompt_version"] == "v42"
    assert meta["prompt_version"] == "v42"
    assert new_state["result"] == "resp"
    assert node_meta == {"result": "resp", "prompt_version": "v42"}


def test_prompt_runner_with_result_shaper(monkeypatch):
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.load",
        lambda alias: {"text": "Prompt", "version": "v1"},
    )

    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_prompt", lambda value, **kwargs: value
    )

    def fake_call(label, prompt, metadata):
        return {"text": "resp", "usage": {}}

    monkeypatch.setattr("ai_core.nodes._prompt_runner.client.call", fake_call)

    shaped_state, shaped_meta = run_prompt_node(
        trace_name="unit",
        prompt_alias="alias",
        llm_label="label",
        state_key="value",
        state={},
        meta=META.copy(),
        result_shaper=lambda result: (result["text"].upper(), {"raw": result["text"]}),
    )

    assert shaped_state["value"] == "RESP"
    assert shaped_meta["value"] == "RESP"
    assert shaped_meta["raw"] == "resp"
    assert shaped_meta["prompt_version"] == "v1"


def test_draft_blocks(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {}
    new_state, result = draft_blocks.run(state, META.copy())
    assert called["label"] == "draft"
    # all three prompt segments should be present
    assert "Systembeschreibung" in called["prompt"]
    assert "Funktionsliste" in called["prompt"]
    assert "Standard-Klauselvorschläge" in called["prompt"]
    assert new_state["draft"] == "resp"
    assert result["draft"] == "resp"


def test_needs_mapping():
    state = {"info_state": {"purpose": "Acme", "extra": "foo"}}
    new_state, result = needs.run(state, META.copy())
    assert result["filled"] == ["purpose"]
    assert result["missing"] == ["deployment_model", "main_components"]
    assert result["ignored"] == ["extra"]
    assert new_state["needs"] == result


def test_tracing_called(monkeypatch):
    payloads = []
    monkeypatch.setattr("ai_core.infra.tracing.emit_span", lambda *a, **k: None)
    monkeypatch.setattr(
        "ai_core.infra.tracing.emit_event", lambda p: payloads.append(p)
    )
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {"question": "Q?", "snippets": []}
    compose.run(state, META.copy())
    assert payloads[0]["event"] == "node.start"
    assert payloads[0]["node"] == "compose"
    assert payloads[0]["tenant"] == "t1"
    assert payloads[0]["prompt_version"] == "v1"
