from __future__ import annotations

import json
from typing import Any

import pytest

from ai_core.llm import routing
from ai_core.llm.client import call


def test_resolve_reads_yaml(tmp_path, monkeypatch):
    mapping = {"simple-query": "gpt-3.5"}
    file = tmp_path / "MODEL_ROUTING.yaml"
    file.write_text(json.dumps(mapping))
    monkeypatch.setattr(routing, "ROUTING_FILE", file)
    routing.load_map.cache_clear()

    assert routing.resolve("simple-query") == "gpt-3.5"
    with pytest.raises(ValueError):
        routing.resolve("missing")


def test_llm_client_masks_records_and_retries(monkeypatch):
    metadata = {
        "tenant": "t1",
        "case": "c1",
        "trace_id": "tr1",
        "prompt_version": "v1",
        "key_alias": "alias-01",
    }

    class FailOnce:
        def __init__(self):
            self.calls = 0
            self.headers: list[str] = []

        def __call__(
            self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int
        ):
            assert json["messages"][0]["content"] == "XXXX"
            assert headers["Authorization"] == "Bearer token"
            assert headers["X-Trace-ID"] == "tr1"
            assert headers["X-Case-ID"] == "c1"
            assert headers["X-Tenant-ID"] == "t1"
            assert headers["X-Key-Alias"] == "alias-01"
            self.headers.append(headers["Idempotency-Key"])
            self.calls += 1
            if self.calls == 1:

                class Resp:
                    status_code = 500

                    def json(self):
                        return {}

                return Resp()
            else:

                class Resp:
                    status_code = 200

                    def json(self):
                        return {
                            "choices": [{"message": {"content": "ok"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }

                return Resp()

    fail_once = FailOnce()

    ledger_calls = {}

    def mock_record(meta):
        ledger_calls["meta"] = meta

    monkeypatch.setattr("ai_core.llm.client.requests.post", fail_once)
    monkeypatch.setattr("ai_core.llm.client.ledger.record", mock_record)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "token")

    from ai_core.infra import config as conf

    conf.get_config.cache_clear()

    res = call("simple-query", "secret", metadata)
    assert res["model"]
    assert res["prompt_version"] == "v1"
    assert ledger_calls["meta"]["label"] == "simple-query"
    assert ledger_calls["meta"]["tenant"] == "t1"
    assert ledger_calls["meta"]["usage"]["in_tokens"] == 1
    assert "text" not in ledger_calls["meta"]
    assert fail_once.calls == 2
    assert fail_once.headers == ["c1:simple-query:v1:1", "c1:simple-query:v1:2"]


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

        def __call__(
            self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int
        ):
            self.headers.append(headers["Idempotency-Key"])

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

    assert capture.headers == ["c1:simple-query:v1:1", "c1:simple-query:v2:1"]
