"""Tests for the LiteLLM-backed embedding client."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ai_core.rag import embeddings
from common.logging import log_context


class _DummyLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.calls.append(("warning", event, dict(kwargs)))

    def error(self, event: str, **kwargs: object) -> None:
        self.calls.append(("error", event, dict(kwargs)))

    def info(self, event: str, **kwargs: object) -> None:  # pragma: no cover - helper
        self.calls.append(("info", event, dict(kwargs)))


@pytest.mark.django_db
def test_embed_timeout_enforces_limit_and_logs_key_alias(
    monkeypatch, settings
) -> None:
    embeddings.reset_embedding_client()
    settings.EMBEDDINGS_MODEL_PRIMARY = "primary"
    settings.EMBEDDINGS_MODEL_FALLBACK = "fallback"
    settings.EMBEDDINGS_TIMEOUT_SECONDS = 0.01

    config = SimpleNamespace(
        litellm_base_url="https://litellm.example",
        litellm_api_key="test-key",
        redis_url="",
        langfuse_public_key="",
        langfuse_secret_key="",
        timeouts={},
    )
    monkeypatch.setattr(embeddings, "get_config", lambda: config)

    calls: list[dict[str, object]] = []

    def _fake_embedding(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        model = kwargs.get("model")
        if model == "primary":
            time.sleep(0.05)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    monkeypatch.setattr(embeddings, "litellm_embedding", _fake_embedding)

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(embeddings, "logger", dummy_logger)

    with log_context(key_alias="alias-test"):
        client = embeddings.EmbeddingClient.from_settings()
        result = client.embed(["hello"])

    assert result.model == "fallback"
    assert result.model_used == "fallback"
    assert result.attempts == 2
    assert result.timeout_s == pytest.approx(0.01)

    assert len(calls) == 2
    assert calls[0]["model"] == "primary"
    assert calls[1]["model"] == "fallback"
    assert calls[0]["timeout"] == pytest.approx(0.01)
    assert calls[1]["timeout"] == pytest.approx(0.01)

    warning_events = [
        data
        for level, event, data in dummy_logger.calls
        if level == "warning" and event == "embeddings.batch_failed"
    ]
    assert warning_events
    assert warning_events[0]["key_alias"] == "alias-test"

    final_events = [
        event for level, event, _ in dummy_logger.calls if level == "error"
    ]
    assert "embeddings.batch_failed_final" not in final_events

    embeddings.reset_embedding_client()
