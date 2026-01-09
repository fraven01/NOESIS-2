from __future__ import annotations

from typing import Any

from llm_worker import runner


class _DummyAsyncResult:
    id = "task-1"

    def get(self, *, timeout: float | None = None, propagate: bool = True):
        return {"state": {}, "result": {}, "cost_summary": None}


def _setup_runner(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_signature(_name, *, kwargs, queue):
        captured["queue"] = queue

        class _DummySignature:
            def __init__(self, payload):
                self.kwargs = payload
                self.options = {}

            def clone(self):
                return self

        return _DummySignature(kwargs)

    monkeypatch.setattr(runner.current_app, "signature", fake_signature)
    monkeypatch.setattr(
        runner,
        "with_scope_apply_async",
        lambda _signature, _scope: _DummyAsyncResult(),
    )
    return captured


def _scope() -> dict[str, str]:
    return {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "service_id": "svc",
    }


def test_submit_worker_task_defaults_to_high(monkeypatch) -> None:
    captured = _setup_runner(monkeypatch)

    payload, completed = runner.submit_worker_task(
        task_payload={},
        scope=_scope(),
        graph_name="rag.default",
    )

    assert captured["queue"] == "agents-high"
    assert completed is True
    assert payload["task_id"] == "task-1"


def test_submit_worker_task_priority_low(monkeypatch) -> None:
    captured = _setup_runner(monkeypatch)

    runner.submit_worker_task(
        task_payload={},
        scope=_scope(),
        graph_name="rag.default",
        priority="low",
    )

    assert captured["queue"] == "agents-low"


def test_submit_worker_task_payload_priority(monkeypatch) -> None:
    captured = _setup_runner(monkeypatch)

    runner.submit_worker_task(
        task_payload={"priority": "low"},
        scope=_scope(),
        graph_name="rag.default",
    )

    assert captured["queue"] == "agents-low"
