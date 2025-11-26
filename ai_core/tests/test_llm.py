from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from ai_core.infra.mask_prompt import mask_prompt
from ai_core.llm import routing
from ai_core.llm.client import LlmClientError, RateLimitError, call
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
from common.logging import mask_value


def test_resolve_merges_base_and_override(tmp_path, monkeypatch):
    base_file = tmp_path / "MODEL_ROUTING.yaml"
    base_file.write_text(
        json.dumps({"simple-query": "gpt-3.5", "default": "base-default"})
    )
    override_file = tmp_path / "MODEL_ROUTING.local.yaml"
    override_file.write_text(json.dumps({"default": "override-default"}))

    monkeypatch.setattr(routing, "ROUTING_FILE", base_file)
    monkeypatch.setattr(routing, "LOCAL_OVERRIDE_FILE", override_file)
    routing.load_map.cache_clear()

    assert routing.resolve("simple-query") == "gpt-3.5"
    assert routing.resolve("default") == "override-default"
    with pytest.raises(ValueError):
        routing.resolve("missing")


def test_llm_client_masks_and_records(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
        "key_alias": "alias-01",
    }
    sanitized_prompt = mask_prompt("secret")

    class CaptureCall:
        def __init__(self):
            self.calls = 0
            self.idempotency_headers: list[str] = []

        def __call__(self, url: str, headers: dict[str, str], json: dict[str, Any]):
            assert json["messages"][0]["content"] == sanitized_prompt
            assert headers["Authorization"] == "Bearer token"
            assert headers[X_TRACE_ID_HEADER] == "tr1"
            assert headers[X_CASE_ID_HEADER] == "c1"
            assert headers[X_TENANT_ID_HEADER] == "t1"
            assert headers[X_KEY_ALIAS_HEADER] == "alias-01"
            self.idempotency_headers.append(headers[IDEMPOTENCY_KEY_HEADER])
            self.calls += 1

            class Resp:
                status_code = 200
                headers: dict[str, str] = {}

                def json(self):
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }

            return Resp()

    capture = CaptureCall()

    ledger_calls = {}

    def mock_record(meta):
        ledger_calls["meta"] = meta

    monkeypatch.setattr("ai_core.llm.client.requests.post", capture)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", mock_record)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "token")
    monkeypatch.setenv("LITELLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")

    from ai_core.infra import config as conf

    conf.get_config.cache_clear()
    routing.load_map.cache_clear()

    res = call("simple-query", sanitized_prompt, metadata)
    assert res["model"]
    assert res["prompt_version"] == "v1"
    assert ledger_calls["meta"]["label"] == "simple-query"
    assert ledger_calls["meta"]["tenant"] == "t1"
    assert ledger_calls["meta"]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }
    assert "text" not in ledger_calls["meta"]
    assert capture.calls == 1
    assert capture.idempotency_headers == ["c1:simple-query:v1"]


def test_llm_client_flattens_structured_content(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    class StructuredContent:
        def __call__(self, url: str, headers: dict[str, str], json: dict[str, Any]):
            class Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"text": "Alpha"},
                                        {"content": "Beta"},
                                        {"type": "output_text", "text": "Gamma"},
                                    ],
                                    "thinking_blocks": [],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                    }

            return Resp()

    handler = StructuredContent()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)
    _prepare_env(monkeypatch)

    result = call("simple-query", "prompt", metadata)
    assert result["text"] == "Alpha\nBeta\nGamma"


def test_llm_client_falls_back_to_choice_text(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    class ChoiceText:
        def __call__(self, url, headers, json):
            class Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {
                                "text": "Direct output",
                                "message": {"role": "assistant"},
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    }

            return Resp()

    handler = ChoiceText()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)
    _prepare_env(monkeypatch)

    result = call("simple-query", "prompt", metadata)
    assert result["text"] == "Direct output"


def test_llm_client_raises_when_content_missing(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    class MissingContent:
        def __call__(self, url, headers, json):
            class Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {"message": {"role": "assistant", "thinking_blocks": []}}
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }

            return Resp()

    handler = MissingContent()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)
    _prepare_env(monkeypatch)

    with pytest.raises(LlmClientError) as excinfo:
        call("simple-query", "prompt", metadata)

    assert "missing content" in str(excinfo.value).lower()


def test_llm_idempotency_key_changes_with_prompt_version(monkeypatch):
    metadata_v1 = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }
    metadata_v2 = {**metadata_v1, "prompt_version": "v2"}

    class CaptureHeaders:
        def __init__(self):
            self.headers: list[str] = []

        def __call__(self, url: str, headers: dict[str, str], json: dict[str, Any]):
            self.headers.append(headers[IDEMPOTENCY_KEY_HEADER])

            class Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }

            return Resp()

    capture = CaptureHeaders()

    monkeypatch.setattr("ai_core.llm.client.requests.post", capture)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "token")

    from ai_core.infra import config as conf

    conf.get_config.cache_clear()

    call("simple-query", "prompt-1", metadata_v1)
    call("simple-query", "prompt-1", metadata_v2)

    assert capture.headers == ["c1:simple-query:v1", "c1:simple-query:v2"]


def _prepare_env(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "token")
    monkeypatch.setenv("LITELLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    from ai_core.infra import config as conf

    conf.get_config.cache_clear()
    routing.load_map.cache_clear()


def test_llm_client_logs_masked_context_on_5xx(monkeypatch):
    metadata = {
        "tenant": "tenant-123",
        "case": "case-456",
        "trace_id": "trace-789",
        "prompt_version": "v1",
        "key_alias": "alias-999",
    }

    warnings: list[tuple[str, dict[str, object]]] = []

    def fake_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append((message, kwargs))

    monkeypatch.setattr("ai_core.llm.client.logger.warning", fake_warning)
    monkeypatch.setattr("ai_core.llm.client.time.sleep", lambda duration: None)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysFail:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1

            class Resp:
                status_code = 502
                headers: dict[str, str] = {}

                def json(self) -> dict[str, object]:
                    return {}

            return Resp()

    monkeypatch.setattr("ai_core.llm.client.requests.post", AlwaysFail())
    _prepare_env(monkeypatch)

    with pytest.raises(LlmClientError):
        call("simple-query", "secret", metadata)

    expected_extra = {
        "trace_id": mask_value(metadata["trace_id"]),
        "case_id": mask_value(metadata["case"]),
        "tenant": mask_value(metadata["tenant"]),
        "key_alias": mask_value(metadata["key_alias"]),
        "status": 502,
    }

    for _, kwargs in warnings:
        assert kwargs.get("extra") == expected_extra


def test_llm_client_logs_masked_context_on_request_error(monkeypatch):
    metadata = {
        "tenant": "tenant-123",
        "case": "case-456",
        "trace_id": "trace-789",
        "prompt_version": "v1",
        "key_alias": "alias-999",
    }

    warnings: list[tuple[str, dict[str, object]]] = []

    def fake_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append((message, kwargs))

    monkeypatch.setattr("ai_core.llm.client.logger.warning", fake_warning)
    monkeypatch.setattr("ai_core.llm.client.time.sleep", lambda duration: None)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysFail:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1
            raise requests.RequestException("boom")

    monkeypatch.setattr("ai_core.llm.client.requests.post", AlwaysFail())
    _prepare_env(monkeypatch)

    with pytest.raises(LlmClientError) as excinfo:
        call("simple-query", "secret", metadata)

    assert str(excinfo.value) == "boom"

    expected_extra = {
        "trace_id": mask_value(metadata["trace_id"]),
        "case_id": mask_value(metadata["case"]),
        "tenant": mask_value(metadata["tenant"]),
        "key_alias": mask_value(metadata["key_alias"]),
        "status": None,
    }

    assert warnings
    assert warnings[0][0] == "llm request error"
    assert warnings[0][1].get("extra") == expected_extra


def test_llm_client_raises_llmclienterror_with_json_error(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    sleep_calls: list[float] = []

    monkeypatch.setattr(
        "ai_core.llm.client.time.sleep",
        lambda duration: sleep_calls.append(duration),
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysFail:
        def __init__(self):
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1

            class Resp:
                status_code = 502
                headers: dict[str, str] = {}
                text = ""

                def json(self):
                    return {
                        "detail": "backend exploded",
                        "code": "bad_gateway",
                        "status": 502,
                    }

            return Resp()

    handler = AlwaysFail()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    _prepare_env(monkeypatch)

    with pytest.raises(LlmClientError) as excinfo:
        call("simple-query", "secret", metadata)

    assert handler.calls == 1
    err = excinfo.value
    assert err.detail == "backend exploded"
    assert err.code == "bad_gateway"
    assert err.status == 502
    assert str(err) == "backend exploded (status=502, code=bad_gateway)"
    assert sleep_calls == []


def test_llm_client_raises_llmclienterror_with_text_error(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    sleep_calls: list[float] = []

    monkeypatch.setattr(
        "ai_core.llm.client.time.sleep",
        lambda duration: sleep_calls.append(duration),
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysFail:
        def __init__(self):
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1

            class Resp:
                status_code = 503
                headers: dict[str, str] = {}
                text = "service unavailable"

                def json(self):
                    raise ValueError("no json")

            return Resp()

    handler = AlwaysFail()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    _prepare_env(monkeypatch)

    with pytest.raises(LlmClientError) as excinfo:
        call("simple-query", "secret", metadata)

    assert handler.calls == 1
    err = excinfo.value
    assert err.detail == "service unavailable"
    assert err.code is None
    assert err.status == 503
    assert str(err) == "service unavailable (status=503)"
    assert sleep_calls == []


def test_llm_client_raises_rate_limit_error_with_json_body(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    sleep_calls: list[float] = []

    monkeypatch.setattr(
        "ai_core.llm.client.time.sleep",
        lambda duration: sleep_calls.append(duration),
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysRateLimited:
        def __init__(self):
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1

            class Resp:
                status_code = 429
                headers: dict[str, str] = {}
                text = ""

                def json(self):
                    return {"detail": "slow down", "code": "rate_limit", "status": 429}

            return Resp()

    handler = AlwaysRateLimited()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    _prepare_env(monkeypatch)

    with pytest.raises(RateLimitError) as excinfo:
        call("simple-query", "secret", metadata)

    assert handler.calls == 1
    err = excinfo.value
    assert err.detail == "slow down"
    assert err.code == "rate_limit"
    assert err.status == 429
    assert str(err) == "slow down (status=429, code=rate_limit)"
    assert sleep_calls == []


def test_llm_client_raises_rate_limit_error_with_text_body(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
    }

    sleep_calls: list[float] = []

    monkeypatch.setattr(
        "ai_core.llm.client.time.sleep",
        lambda duration: sleep_calls.append(duration),
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class AlwaysRateLimited:
        def __init__(self):
            self.calls = 0

        def __call__(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ):
            self.calls += 1

            class Resp:
                status_code = 429
                headers: dict[str, str] = {}
                text = "too many"

                def json(self):
                    raise ValueError("not json")

            return Resp()

    handler = AlwaysRateLimited()
    monkeypatch.setattr("ai_core.llm.client.requests.post", handler)
    _prepare_env(monkeypatch)

    with pytest.raises(RateLimitError) as excinfo:
        call("simple-query", "secret", metadata)

    assert handler.calls == 1
    err = excinfo.value
    assert err.detail == "too many"
    assert err.code is None
    assert err.status == 429
    assert str(err) == "too many (status=429)"
    assert sleep_calls == []


def test_llm_client_updates_observation_on_success(monkeypatch):
    metadata = {
        "tenant": "tenant-1",
        "case": "case-1",
        "trace_id": "trace-1",
        "prompt_version": "v1",
    }

    observation_calls: list[dict[str, Any]] = []

    def fake_update_observation(**fields: Any) -> None:
        observation_calls.append(fields)

    monkeypatch.setattr(
        "ai_core.llm.client.update_observation", fake_update_observation
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class Resp:
        status_code = 200
        headers: dict[str, str] = {"x-litellm-cache-hit": "true"}

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 40},
            }

    monkeypatch.setattr(
        "ai_core.llm.client.requests.post", lambda *args, **kwargs: Resp()
    )
    _prepare_env(monkeypatch)

    prompt = "x" * 600
    result = call("simple-query", prompt, metadata)

    assert result["usage"] == {"prompt_tokens": 20, "completion_tokens": 40}
    assert result["cache_hit"] is True
    assert result["latency_ms"] is not None

    success_call = observation_calls[-1]
    success_meta = success_call["metadata"]
    resolved_model = routing.resolve("simple-query")
    assert success_meta["status"] == "success"
    assert success_meta["model.id"] == resolved_model
    assert success_meta["usage.prompt_tokens"] == 20
    assert success_meta["usage.completion_tokens"] == 40
    assert success_meta["usage.total_tokens"] == 60
    assert success_meta["cache_hit"] is True
    assert len(success_meta["input.masked_prompt"]) == 512
    assert success_meta["input.masked_prompt"] == "x" * 512


def test_llm_client_updates_observation_on_error(monkeypatch):
    metadata = {
        "tenant": "tenant-2",
        "case": "case-2",
        "trace_id": "trace-2",
        "prompt_version": "v2",
    }

    observation_calls: list[dict[str, Any]] = []

    def fake_update_observation(**fields: Any) -> None:
        observation_calls.append(fields)

    monkeypatch.setattr(
        "ai_core.llm.client.update_observation", fake_update_observation
    )
    monkeypatch.setattr("ai_core.llm.client.ledger.record", lambda meta: None)

    class Resp:
        status_code = 400
        headers: dict[str, str] = {"x-litellm-cache-hit": "false"}

        def json(self):
            return {"detail": "bad request", "status": 400, "code": "invalid"}

    monkeypatch.setattr(
        "ai_core.llm.client.requests.post", lambda *args, **kwargs: Resp()
    )
    _prepare_env(monkeypatch)

    prompt = "prompt-preview"
    with pytest.raises(LlmClientError):
        call("simple-query", prompt, metadata)

    error_call = observation_calls[-1]
    error_meta = error_call["metadata"]
    assert error_meta["status"] == "error"
    assert error_meta["model.id"] == routing.resolve("simple-query")
    assert error_meta["error.type"] == "LlmClientError"
    assert error_meta["error.message"] == "bad request"
    assert error_meta["provider.http_status"] == 400
    assert error_meta["cache_hit"] is False
    assert error_meta["input.masked_prompt"] == prompt
