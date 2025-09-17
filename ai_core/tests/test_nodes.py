import pytest

from ai_core.nodes import (
    retrieve,
    compose,
    extract,
    classify,
    assess,
    draft_blocks,
    needs,
)
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
    new_state, result = compose.run(state, META.copy())
    assert called["label"] == "synthesize"
    assert "XXXX" in called["prompt"]
    assert called["meta"]["prompt_version"] == "v1"
    assert new_state["answer"] == "resp"
    assert result["answer"] == "resp"


def test_extract_classify_assess(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    state = {"text": "Fact 42"}
    new_state, _ = extract.run(state, META.copy())
    assert called["label"] == "extract"
    assert "XXXX" in called["prompt"]
    assert new_state["items"] == "resp"

    new_state, _ = classify.run(state, META.copy())
    assert called["label"] == "classify"
    assert new_state["classification"] == "resp"

    new_state, _ = assess.run(state, META.copy())
    assert called["label"] == "analyze"
    assert new_state["risk"] == "resp"


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
