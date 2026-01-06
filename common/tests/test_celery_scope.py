from __future__ import annotations

from celery import chain, current_app, signature

from ai_core.infra.policy import get_session_scope
from common.celery import ScopedTask, with_scope_apply_async


def test_scoped_task_sets_and_clears_session_scope():
    captured_kwargs: dict[str, object] = {}

    class _DummyTask(ScopedTask):
        abstract = False

        def run(self, **kwargs):  # type: ignore[override]
            captured_kwargs.update(kwargs)
            return get_session_scope()

    task = _DummyTask()
    task.bind(current_app)

    meta = {
        "scope_context": {
            "tenant_id": "tenant-1",
            "trace_id": "trace-1",
            "invocation_id": "inv-1",
            "run_id": "run-1",
        },
        "business_context": {"case_id": "case-1"},
    }
    result = task.__call__(foo="bar", meta=meta)

    assert result == ("tenant-1", "case-1", "trace-1||case-1||tenant-1")
    assert captured_kwargs["foo"] == "bar"
    assert captured_kwargs["meta"] == meta
    assert get_session_scope() is None


def test_with_scope_apply_async_does_not_inject_scope(monkeypatch):
    monkeypatch.setattr("common.celery._OTEL_AVAILABLE", False)
    sig = chain(signature("task_a"), signature("task_b"))

    captured: dict[str, object] = {}

    def _fake_apply_async(self, *args, **kwargs):
        captured["signature"] = self
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "scheduled"

    monkeypatch.setattr(
        type(sig),
        "apply_async",
        _fake_apply_async,
        raising=False,
    )

    scope = {"tenant_id": "t1", "case_id": "c1", "trace_id": "tr1"}
    result = with_scope_apply_async(sig, scope, countdown=5)

    assert result == "scheduled"
    scoped = captured["signature"]

    for sub_sig in scoped.tasks:  # type: ignore[union-attr]
        assert "tenant_id" not in sub_sig.kwargs
        assert "case_id" not in sub_sig.kwargs
        assert "trace_id" not in sub_sig.kwargs
        assert "session_salt" not in sub_sig.kwargs

    assert captured["kwargs"] == {"countdown": 5}
