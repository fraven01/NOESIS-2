from __future__ import annotations

from ai_core.llm.client import LlmClientError
from ai_core.rag.standalone_question import generate_standalone_question
from ai_core.tests.utils import make_test_tool_context


def test_generate_standalone_question_returns_original_when_no_history():
    context = make_test_tool_context(case_id="case-1")
    result = generate_standalone_question("What is it?", [], context)

    assert result.question == "What is it?"
    assert result.source == "original"


def test_generate_standalone_question_uses_llm_when_history_present(monkeypatch):
    context = make_test_tool_context(case_id="case-1")
    history = [{"role": "user", "content": "Tell me about the Pro Plan."}]

    def fake_call(label: str, prompt: str, metadata: dict) -> dict:
        return {"text": "How much does the Pro Plan cost?"}

    monkeypatch.setattr("ai_core.rag.standalone_question.llm_client.call", fake_call)

    result = generate_standalone_question("How much is it?", history, context)

    assert result.question == "How much does the Pro Plan cost?"
    assert result.source == "llm"


def test_generate_standalone_question_falls_back_on_error(monkeypatch):
    context = make_test_tool_context(case_id="case-1")
    history = [{"role": "user", "content": "Tell me about the Pro Plan."}]

    def fake_call(label: str, prompt: str, metadata: dict) -> dict:
        raise LlmClientError("boom")

    monkeypatch.setattr("ai_core.rag.standalone_question.llm_client.call", fake_call)

    result = generate_standalone_question("How much is it?", history, context)

    assert result.question == "How much is it?"
    assert result.source == "fallback"
