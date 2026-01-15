import json
import logging
from typing import Any
from uuid import uuid4

import pytest
from django.http import HttpResponse

import ai_core.infra.observability as observability
from ai_core.infra import object_store, pii
from ai_core.infra.config import get_config
from ai_core.infra.resp import apply_std_headers
from ai_core.infra.observability import (
    emit_event,
    end_trace,
    observe_span,
    record_span,
    start_trace,
    update_observation,
)
from common.constants import (
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)


def test_get_config_reads_env(monkeypatch):
    get_config.cache_clear()
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    monkeypatch.setenv("LITELLM_TIMEOUTS", json.dumps({"fast": 5, "default": 30}))

    cfg = get_config()
    assert cfg.litellm_base_url == "http://litellm.local"
    assert cfg.litellm_api_key == "secret"
    assert cfg.redis_url == "redis://localhost:6379/0"
    assert cfg.langfuse_public_key == "pub"
    assert cfg.langfuse_secret_key == "sec"
    assert cfg.timeouts == {"fast": 5, "default": 30}


def test_apply_std_headers_sets_metadata_headers_for_success():
    resp = HttpResponse("ok", status=200)
    meta = {
        "scope_context": {
            "trace_id": "abc123",
            "tenant_id": "tenant-1",
            "invocation_id": "inv-1",
            "run_id": "run-1",
        },
        "business_context": {
            "case_id": "case-1",
        },
        "key_alias": "alias-1",
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    }

    result = apply_std_headers(resp, meta)

    assert result[X_TRACE_ID_HEADER] == "abc123"
    assert result[X_CASE_ID_HEADER] == "case-1"
    assert result[X_TENANT_ID_HEADER] == "tenant-1"
    assert result[X_KEY_ALIAS_HEADER] == "alias-1"
    assert result["traceparent"] == meta["traceparent"]


def test_apply_std_headers_skips_missing_optional_headers():
    resp = HttpResponse("ok", status=200)
    meta = {
        "scope_context": {
            "trace_id": "abc123",
            "tenant_id": "tenant-1",
            "invocation_id": "inv-2",
            "run_id": "run-2",
        },
        "business_context": {
            "case_id": "case-1",
        },
    }

    result = apply_std_headers(resp, meta)

    assert X_KEY_ALIAS_HEADER not in result
    assert result[X_TRACE_ID_HEADER] == "abc123"
    assert "traceparent" not in result


def test_apply_std_headers_ignores_non_success_responses():
    meta = {
        "scope_context": {
            "trace_id": "abc123",
            "tenant_id": "tenant-1",
            "invocation_id": "inv-3",
            "run_id": "run-3",
        },
        "business_context": {
            "case_id": "case-1",
        },
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    }

    for status in (400, 500):
        resp = HttpResponse("error", status=status)
        result = apply_std_headers(resp, meta)
        assert X_TRACE_ID_HEADER not in result
        assert X_CASE_ID_HEADER not in result
        assert X_TENANT_ID_HEADER not in result
        assert X_KEY_ALIAS_HEADER not in result
        assert "traceparent" not in result


def test_pii_mask_leaves_plain_numbers():
    assert pii.mask("User 123") == "User 123"


def test_object_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    object_store.write_json("tenant/case/state.json", {"ok": True})
    assert object_store.read_json("tenant/case/state.json") == {"ok": True}
    object_store.put_bytes("tenant/case/raw/data.bin", b"hi")
    stored = tmp_path / ".ai_core_store/tenant/case/raw/data.bin"
    assert stored.read_bytes() == b"hi"

    object_store.write_bytes("tenant/case/raw/data-copy.bin", b"hi-2")
    stored_copy = tmp_path / ".ai_core_store/tenant/case/raw/data-copy.bin"
    assert stored_copy.read_bytes() == b"hi-2"


def test_sanitize_identifier_replaces_invalid_characters():
    assert object_store.sanitize_identifier("tenant name!@#") == "tenant_name___"


def test_sanitize_identifier_rejects_unsafe_sequences():
    with pytest.raises(ValueError):
        object_store.sanitize_identifier("..")
    with pytest.raises(ValueError):
        object_store.sanitize_identifier("tenant/abc")


def test_observe_span_uses_tracer(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeContext:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            calls.append(("enter", self.name))
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            calls.append(("exit", self.name))
            return False

    class FakeTracer:
        def start_as_current_span(self, name: str, attributes=None):  # noqa: D401
            calls.append(("start", name))
            return FakeContext(name)

    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_get_tracer", lambda: FakeTracer())

    @observe_span(name="unit.test")
    def sample(value: int) -> int:
        return value + 1

    assert sample(1) == 2
    assert calls[0] == ("start", "unit.test")
    assert calls[1] == ("enter", "unit.test")
    assert calls[-1] == ("exit", "unit.test")


def test_observe_span_no_tracer_noop(monkeypatch):
    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_get_tracer", lambda: None)

    @observe_span(name="noop")
    def sample(counter: list[int]) -> None:
        counter.append(1)

    bucket: list[int] = []
    sample(bucket)
    assert bucket == [1]


def test_update_observation_sets_attributes(monkeypatch):
    recorded: dict[str, Any] = {}
    user_id = str(uuid4())

    class FakeSpan:
        def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
            recorded[key] = value

    monkeypatch.setattr(observability, "_get_current_span", lambda: FakeSpan())

    update_observation(
        user_id=user_id,
        session_id="case-1",
        tags=["a", "b"],
        metadata={"foo": "bar"},
    )

    assert recorded["user.id"] == user_id
    assert recorded["session.id"] == "case-1"
    assert recorded["tags"] == ["a", "b"]
    assert recorded["meta.foo"] == "bar"


def test_record_span_invokes_tracer(monkeypatch):
    spans: list[tuple[str, dict[str, Any]]] = []

    class FakeSpan:
        def __init__(self, attrs: dict[str, Any]) -> None:
            self.attrs = attrs

        def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
            self.attrs[key] = value

    class FakeContext:
        def __init__(self, name: str, attrs: dict[str, Any]) -> None:
            self.name = name
            self.attrs = attrs

        def __enter__(self) -> FakeSpan:
            spans.append((self.name, dict(self.attrs)))
            return FakeSpan(self.attrs)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeTracer:
        def start_as_current_span(self, name: str, attributes=None):  # noqa: D401
            return FakeContext(name, dict(attributes or {}))

    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_get_tracer", lambda: FakeTracer())
    monkeypatch.setattr(observability, "_get_current_span", lambda: FakeSpan({}))

    record_span(
        "unit.record",
        attributes={"value": 1, "flag": True, "trace_id": "abc"},
    )

    name, attrs = spans[0]
    assert name == "unit.record"
    assert attrs["value"] == 1
    assert attrs["flag"] is True
    assert attrs["trace_id"] == "abc"


def test_emit_event_attaches_to_span(monkeypatch):
    events: list[tuple[str, dict[str, Any]]] = []

    class FakeSpan:
        def add_event(self, name: str, attributes=None) -> None:  # noqa: D401
            events.append((name, dict(attributes or {})))

    monkeypatch.setattr(observability, "_get_current_span", lambda: FakeSpan())

    emit_event({"event": "node.start", "foo": "bar"})

    assert events == [("node.start", {"foo": "bar"})]


def test_emit_event_accepts_event_name_signature(monkeypatch):
    events: list[tuple[str, dict[str, Any]]] = []

    class FakeSpan:
        def add_event(self, name: str, attributes=None) -> None:  # noqa: D401
            events.append((name, dict(attributes or {})))

    monkeypatch.setattr(observability, "_get_current_span", lambda: FakeSpan())

    emit_event("node.start", {"foo": "bar"})

    assert events == [("node.start", {"foo": "bar"})]


def test_emit_event_prints_without_span(monkeypatch, caplog):
    monkeypatch.setattr(observability, "_get_current_span", lambda: None)

    with caplog.at_level(logging.INFO, logger=observability.LOGGER.name):
        emit_event({"event": "node.start", "foo": "bar"})

    record = caplog.records[-1]
    assert record.message == "observability.event"
    assert record.event == "node.start"
    assert record.foo == "bar"


def test_start_and_end_trace_manage_context(monkeypatch):
    seen: list[str] = []
    user_id = str(uuid4())

    class FakeContext:
        def __enter__(self) -> None:  # noqa: D401
            seen.append("enter")
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401
            seen.append("exit")
            return False

    class FakeTracer:
        def start_as_current_span(
            self, name: str, attributes=None, **kwargs
        ):  # noqa: D401
            seen.append(name)
            return FakeContext()

    monkeypatch.setattr(observability, "tracing_enabled", lambda: True)
    monkeypatch.setattr(observability, "_get_tracer", lambda: FakeTracer())

    start_trace(
        name="root",
        user_id=user_id,
        session_id="case",
        metadata={"k": "v"},
    )
    end_trace()

    assert seen[0] == "root"
    assert seen[1:] == ["enter", "exit"]
