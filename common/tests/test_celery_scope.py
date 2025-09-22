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

    result = task.__call__(
        foo="bar",
        tenant_id="tenant-1",
        case_id="case-1",
        trace_id="trace-1",
    )

    assert result == ("tenant-1", "case-1", "trace-1||case-1||tenant-1")
    assert captured_kwargs == {"foo": "bar"}
    assert get_session_scope() is None


def test_with_scope_apply_async_injects_scope(monkeypatch):
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
    expected_salt = "tr1||c1||t1"

    for sub_sig in scoped.tasks:  # type: ignore[union-attr]
        assert sub_sig.kwargs["tenant_id"] == "t1"
        assert sub_sig.kwargs["case_id"] == "c1"
        assert sub_sig.kwargs["trace_id"] == "tr1"
        assert sub_sig.kwargs["session_salt"] == expected_salt

    assert captured["kwargs"] == {"countdown": 5}
