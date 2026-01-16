from ai_core.rag.answer_guardrails import evaluate_answer_guardrails


def test_answer_guardrail_insufficient_snippets(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_MIN_SNIPPETS", "2")
    result = evaluate_answer_guardrails([{"score": 0.9}])
    assert result.allowed is False
    assert result.reason == "insufficient_snippets"


def test_answer_guardrail_low_score(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_MIN_TOP_SCORE", "0.5")
    result = evaluate_answer_guardrails([{"score": 0.2}, {"score": 0.1}])
    assert result.allowed is False
    assert result.reason == "low_top_score"


def test_answer_guardrail_ok(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_MIN_SNIPPETS", "1")
    monkeypatch.setenv("RAG_GUARDRAIL_MIN_TOP_SCORE", "0.2")
    result = evaluate_answer_guardrails([{"score": 0.4}])
    assert result.allowed is True
    assert result.reason == "ok"
