"""Tests for :mod:`ai_core.rag.embeddings.EmbeddingClient`."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from structlog.testing import capture_logs

from ai_core.rag.embeddings import (
    EmbeddingBatchResult,
    EmbeddingClient,
    EmbeddingClientError,
)


@pytest.fixture(autouse=True)
def _embed_settings(settings):
    settings.EMBEDDINGS_PROVIDER = "litellm"
    settings.EMBEDDINGS_MODEL_PRIMARY = "text-embedding-primary"
    settings.EMBEDDINGS_MODEL_FALLBACK = "text-embedding-fallback"
    settings.EMBEDDINGS_TIMEOUT_SECONDS = None


@pytest.fixture
def mock_config(mocker):
    config = SimpleNamespace(
        litellm_base_url="https://mock.litellm",
        litellm_api_key="mock-key",
        timeouts={"embeddings": 1},
    )
    return mocker.patch("ai_core.rag.embeddings.get_config", return_value=config)


@pytest.mark.parametrize("status_code", [500, 429])
def test_embed_promotes_fallback_for_retryable_status_codes(
    mock_config, mocker, status_code
):
    client = EmbeddingClient(
        provider="litellm",
        primary_model="text-embedding-primary",
        fallback_model="text-embedding-fallback",
    )

    attempts: list[str] = []

    def _litellm_embedding(**kwargs):
        model = kwargs["model"]
        attempts.append(model)

        def _handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload.get("model") == model
            if model == "text-embedding-primary":
                return httpx.Response(status_code=status_code, json={"error": "boom"})
            response = [
                {"embedding": [float(idx + 1) for idx, _ in enumerate(kwargs["input"])]}
                for _ in kwargs["input"]
            ]
            return httpx.Response(status_code=200, json={"data": response})

        transport = httpx.MockTransport(_handler)
        with httpx.Client(transport=transport, base_url=kwargs["api_base"]) as http_client:
            response = http_client.post(
                "/embeddings",
                json={"model": model, "input": kwargs["input"]},
            )
        response.raise_for_status()
        data = response.json().get("data", [])
        return SimpleNamespace(data=data)

    mocker.patch(
        "ai_core.rag.embeddings.litellm_embedding",
        side_effect=_litellm_embedding,
    )

    with capture_logs() as logs:
        result = client.embed(["hello"])

    assert attempts == ["text-embedding-primary", "text-embedding-fallback"]
    assert isinstance(result, EmbeddingBatchResult)
    assert result.model == "text-embedding-fallback"
    assert result.model_used == "fallback"
    assert result.attempts == 2

    warning_events = [log for log in logs if log["event"] == "embeddings.batch_failed"]
    assert len(warning_events) == 1
    assert warning_events[0]["status_code"] == status_code
    assert warning_events[0]["retry"] is True


def test_embed_stops_after_unauthorised_error(mock_config, mocker):
    client = EmbeddingClient(
        provider="litellm",
        primary_model="text-embedding-primary",
        fallback_model="text-embedding-fallback",
    )

    def _litellm_embedding(**kwargs):
        model = kwargs["model"]

        def _handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload.get("model") == model
            return httpx.Response(status_code=401, json={"error": "unauthorised"})

        transport = httpx.MockTransport(_handler)
        with httpx.Client(transport=transport, base_url=kwargs["api_base"]) as http_client:
            response = http_client.post(
                "/embeddings",
                json={"model": model, "input": kwargs["input"]},
            )
        response.raise_for_status()
        return SimpleNamespace(data=response.json().get("data", []))

    mocker.patch("ai_core.rag.embeddings.litellm_embedding", side_effect=_litellm_embedding)

    with capture_logs() as logs, pytest.raises(EmbeddingClientError):
        client.embed(["hello"])

    warning_events = [log for log in logs if log["event"] == "embeddings.batch_failed"]
    error_events = [log for log in logs if log["event"] == "embeddings.batch_failed_final"]

    assert len(warning_events) == 1
    assert warning_events[0]["status_code"] == 401
    assert warning_events[0]["retry"] is False
    assert len(error_events) == 1


def test_timeout_promotes_fallback_and_surfaces_timeout_error(mock_config, mocker):
    client = EmbeddingClient(
        provider="litellm",
        primary_model="text-embedding-primary",
        fallback_model="text-embedding-fallback",
    )

    def _litellm_embedding(**kwargs):
        model = kwargs["model"]

        def _handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode()) if request.content else {}
            assert payload.get("model") == model
            if model == "text-embedding-primary":
                raise httpx.ReadTimeout("primary timed out", request=request)
            response = [
                {"embedding": [1.0 for _ in kwargs["input"]]}
                for _ in kwargs["input"]
            ]
            return httpx.Response(status_code=200, json={"data": response})

        transport = httpx.MockTransport(_handler)
        with httpx.Client(transport=transport, base_url=kwargs["api_base"]) as http_client:
            response = http_client.post(
                "/embeddings",
                json={"model": model, "input": kwargs["input"]},
            )
        response.raise_for_status()
        return SimpleNamespace(data=response.json().get("data", []))

    mocker.patch("ai_core.rag.embeddings.litellm_embedding", side_effect=_litellm_embedding)

    with capture_logs() as logs:
        result = client.embed(["hello"])

    assert result.model_used == "fallback"
    warning_events = [log for log in logs if log["event"] == "embeddings.batch_failed"]
    assert len(warning_events) == 1
    assert warning_events[0]["exc_type"] == "EmbeddingTimeoutError"
    assert warning_events[0]["retry"] is True


@pytest.mark.parametrize(
    "provider_data",
    [
        None,
        [],
        [{"not_embedding": []}],
    ],
)
def test_empty_or_null_payload_returns_empty_batch(
    mock_config, mocker, provider_data
):
    client = EmbeddingClient(
        provider="litellm",
        primary_model="text-embedding-primary",
        fallback_model="text-embedding-fallback",
    )

    def _litellm_embedding(**kwargs):

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, json={"data": provider_data})

        transport = httpx.MockTransport(_handler)
        with httpx.Client(transport=transport, base_url=kwargs["api_base"]) as http_client:
            response = http_client.post(
                "/embeddings",
                json={"model": kwargs["model"], "input": kwargs["input"]},
            )
        response.raise_for_status()
        return SimpleNamespace(data=response.json().get("data", []))

    mocker.patch("ai_core.rag.embeddings.litellm_embedding", side_effect=_litellm_embedding)

    result = client.embed(["hello", "world"])

    assert isinstance(result, EmbeddingBatchResult)
    assert result.vectors == []
    assert result.model == "text-embedding-primary"
    assert result.model_used == "primary"
    assert result.attempts == 1
