import pytest

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask_prompt
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

META = {"tenant": "t1", "case": "c1", "trace_id": "tr"}


def test_retrieve_returns_snippets(settings):
    settings.RAG_ENABLED = True

    class _Client:
        def __init__(self, chunks):
            self._chunks = chunks

        def search(self, query, filters, top_k=5):
            return self._chunks

    chunk = Chunk(
        "Hello 123", {"tenant": "t1", "case": "c1", "source": "s1", "hash": "h"}
    )
    client = _Client([chunk])
    state, result = retrieve.run({"query": "Hello"}, META.copy(), client=client)
    assert result["snippets"][0]["text"] == "Hello 123"
    assert result["snippets"][0]["source"] == "s1"
    assert state["snippets"] == result["snippets"]


def test_retrieve_skips_when_disabled(settings, caplog):
    settings.RAG_ENABLED = False
    with caplog.at_level("INFO"):
        state, result = retrieve.run({"query": "Hello"}, META.copy(), client=None)
    assert result["snippets"] == []
    assert state["snippets"] == []
    assert result["confidence"] == 0.0
    assert "RAG is disabled" in caplog.text


def test_retrieve_requires_client_when_enabled(settings):
    settings.RAG_ENABLED = True
    with pytest.raises(ValueError):
        retrieve.run({"query": "Hello"}, META.copy(), client=None)


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

    def fake_mask(value):
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
    monkeypatch.setattr("ai_core.nodes._prompt_runner.mask_prompt", lambda value: value)

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
    monkeypatch.setattr(
        "ai_core.infra.tracing._dispatch_langfuse", lambda *a, **k: None
    )
    monkeypatch.setattr("ai_core.infra.tracing._log", lambda p: payloads.append(p))
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {"question": "Q?", "snippets": []}
    compose.run(state, META.copy())
    assert payloads[0]["event"] == "node.start"
    assert payloads[0]["node"] == "compose"
    assert payloads[0]["tenant"] == "t1"
    assert payloads[0]["prompt_version"] == "v1"
