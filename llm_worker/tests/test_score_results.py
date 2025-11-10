from __future__ import annotations

import json
import os

from ai_core.llm.client import LlmClientError
from llm_worker.graphs import run_score_results
from llm_worker import tasks


def _sample_results():
    return [
        {"id": "a", "title": "Alpha", "snippet": "First hit"},
        {"id": "b", "title": "Beta", "snippet": "Second hit"},
    ]


def test_run_score_results_sorts_rankings(monkeypatch):
    calls: dict[str, object] = {}

    def fake_call(label, prompt, metadata):
        calls["label"] = label
        calls["prompt"] = prompt
        calls["metadata"] = metadata
        return {
            "text": json.dumps(
                {
                    "ranked": [
                        {"id": "b", "score": 55, "reasons": ["ok precision"]},
                        {"id": "a", "score": 95, "reasons": ["matches intent"]},
                        {"id": "missing", "score": 10},
                    ]
                }
            ),
            "usage": {"prompt_tokens": 42},
            "latency_ms": 250.0,
            "model": "fast-model",
        }

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    result = run_score_results(
        control={"model_preset": "fast", "temperature": 0.05},
        data={
            "query": "policy for contractors",
            "results": _sample_results(),
            "criteria": ["aktueller Stand", "deutsches Recht"],
            "k": 1,
        },
        meta={"tenant_id": "t-1", "case_id": "c-1", "trace_id": "trace-1"},
    )

    assert calls["label"] == "fast"
    assert "policy for contractors" in calls["prompt"]
    assert len(result["ranked"]) == 2
    assert result["ranked"][0]["id"] == "a"
    assert result["top_k"][0]["id"] == "a"
    assert result["usage"]["prompt_tokens"] == 42
    assert result["latency_s"] == 0.25
    assert result["model"] == "fast-model"


def test_run_score_results_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "ai_core.llm.client.call",
        lambda *_, **__: {
            "text": "DONE",
            "usage": {},
            "latency_ms": None,
            "model": "fast",
        },
    )

    result = run_score_results(
        control=None,
        data={
            "query": "alpha",
            "results": _sample_results(),
        },
    )

    assert result["ranked"] == []
    assert result["top_k"] == []
    assert result["usage"] == {}
    assert result["latency_s"] is None


def test_run_graph_routes_score_results(monkeypatch):
    sentinel = {"ranked": [{"id": "a", "score": 50, "reasons": []}]}

    def fake_run_score(control, data, meta):
        assert meta["task_type"] == "score_results"
        return sentinel

    def unexpected_runner(_graph_name):
        raise AssertionError("graph runner should not be called for score_results")

    monkeypatch.setattr(tasks, "run_score_results", fake_run_score)
    monkeypatch.setattr(tasks, "get_graph_runner", unexpected_runner)

    payload = tasks.run_graph.run(
        graph_name="dummy",
        state={"step": "start"},
        meta={
            "task_type": "score_results",
            "control": {"model_preset": "fast"},
            "data": {
                "query": "alpha",
                "results": _sample_results(),
            },
        },
    )

    assert payload["state"] == {"step": "start"}
    assert payload["result"] is sentinel
    assert payload["cost_summary"] is None


def test_run_score_results_fallbacks_on_invalid_model(monkeypatch):
    attempts: list[str] = []

    def fake_call(label, prompt, metadata):
        attempts.append(label)
        if len(attempts) == 1:
            raise LlmClientError("Invalid model", status=400)
        return {
            "text": json.dumps(
                {"ranked": [{"id": "a", "score": 77, "reasons": ["good"]}]}
            ),
            "usage": {},
            "latency_ms": 50,
            "model": label,
        }

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    result = run_score_results(
        control={"model_preset": "broken"},
        data={"query": "abc", "results": _sample_results()},
    )

    assert attempts == ["broken", "fast"]
    assert result["ranked"][0]["id"] == "a"


def test_run_score_results_tries_default_label(monkeypatch):
    attempts: list[str] = []

    def fake_call(label, prompt, metadata):
        attempts.append(label)
        if label != "default":
            raise LlmClientError("Invalid model", status=400)
        return {
            "text": json.dumps({"ranked": [{"id": "a", "score": 60, "reasons": []}]}),
            "usage": {},
            "latency_ms": 10,
            "model": label,
        }

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    result = run_score_results(
        control={"model_preset": "vertex_ai/gemini-2.5-flash"},
        data={"query": "alpha", "results": _sample_results()},
    )

    assert attempts == ["vertex_ai/gemini-2.5-flash", "fast", "default"]
    assert result["ranked"][0]["score"] == 60


def test_run_score_results_sets_temperature_for_gpt5(monkeypatch):
    recorded_temps: list[str | None] = []

    def fake_call(label, prompt, metadata):
        recorded_temps.append(os.environ.get("LITELLM_TEMPERATURE"))
        return {
            "text": json.dumps({"ranked": [{"id": "a", "score": 50, "reasons": []}]}),
            "usage": {},
            "latency_ms": 5,
            "model": label,
        }

    monkeypatch.setattr("ai_core.llm.client.call", fake_call)

    run_score_results(
        control={"model_preset": "openai/gpt-5-nano", "temperature": 0.05},
        data={"query": "alpha", "results": _sample_results()},
    )

    assert recorded_temps == [None]
