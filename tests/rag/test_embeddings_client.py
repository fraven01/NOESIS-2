"""Tests for :mod:`ai_core.rag.embeddings.EmbeddingClient`."""

from __future__ import annotations

from types import SimpleNamespace

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

    def _fake_openai_init(*args, **kwargs):
        class MockEmbeddings:
            def create(self, input, model):
                attempts.append(model)
                if model == "text-embedding-primary":
                    err = Exception("boom")
                    err.status_code = status_code
                    raise err
                
                data = []
                for idx, _ in enumerate(input):
                    item = SimpleNamespace(embedding=[float(idx + 1)])
                    data.append(item)
                return SimpleNamespace(data=data)

        class MockClient:
            def __init__(self):
                self.embeddings = MockEmbeddings()
        return MockClient()

    mocker.patch("ai_core.rag.embeddings.OpenAI", side_effect=_fake_openai_init)

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

    def _fake_openai_init(*args, **kwargs):
        class MockEmbeddings:
            def create(self, input, model):
                err = Exception("unauthorised")
                err.status_code = 401
                raise err

        class MockClient:
            def __init__(self):
                self.embeddings = MockEmbeddings()
        return MockClient()

    mocker.patch("ai_core.rag.embeddings.OpenAI", side_effect=_fake_openai_init)

    with capture_logs() as logs, pytest.raises(EmbeddingClientError):
        client.embed(["hello"])

    warning_events = [log for log in logs if log["event"] == "embeddings.batch_failed"]
    error_events = [
        log for log in logs if log["event"] == "embeddings.batch_failed_final"
    ]

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

    def _fake_openai_init(*args, **kwargs):
        class MockEmbeddings:
            def create(self, input, model):
                if model == "text-embedding-primary":
                    raise TimeoutError("primary timed out")
                
                data = []
                for _ in input:
                    item = SimpleNamespace(embedding=[1.0])
                    data.append(item)
                return SimpleNamespace(data=data)

        class MockClient:
            def __init__(self):
                self.embeddings = MockEmbeddings()
        return MockClient()

    mocker.patch("ai_core.rag.embeddings.OpenAI", side_effect=_fake_openai_init)

    with capture_logs() as logs:
        result = client.embed(["hello"])

    assert result.model_used == "fallback"
    warning_events = [log for log in logs if log["event"] == "embeddings.batch_failed"]
    assert len(warning_events) == 1
    # The exception type might be TimeoutError or EmbeddingTimeoutError depending on where it's caught/raised
    # In _invoke_provider, TimeoutError is not caught/wrapped unless it comes from _execute_with_timeout
    # But here we raise it directly from create.
    # _execute_with_timeout catches Exception and checks _is_timeout_exception.
    # If so, it raises EmbeddingTimeoutError.
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
def test_empty_or_null_payload_returns_empty_batch(mock_config, mocker, provider_data):
    client = EmbeddingClient(
        provider="litellm",
        primary_model="text-embedding-primary",
        fallback_model="text-embedding-fallback",
    )

    def _fake_openai_init(*args, **kwargs):
        class MockEmbeddings:
            def create(self, input, model):
                if provider_data is None:
                    return SimpleNamespace(data=None)
                if provider_data == []:
                    return SimpleNamespace(data=[])
                
                # Case [{"not_embedding": []}]
                # We return an object where .embedding is None or missing?
                # If missing, AttributeError.
                # If we want to simulate "invalid" item that results in empty batch:
                # embeddings.py checks: if embedding_values is None: return []
                # So we return item with embedding=None
                data = []
                for item_dict in provider_data:
                    # If item_dict has "not_embedding", it lacks "embedding"
                    # We simulate item with embedding=None
                    item = SimpleNamespace(embedding=None)
                    data.append(item)
                return SimpleNamespace(data=data)

        class MockClient:
            def __init__(self):
                self.embeddings = MockEmbeddings()
        return MockClient()

    mocker.patch("ai_core.rag.embeddings.OpenAI", side_effect=_fake_openai_init)

    result = client.embed(["hello", "world"])

    assert isinstance(result, EmbeddingBatchResult)
    assert result.vectors == []
    assert result.model == "text-embedding-primary"
    assert result.model_used == "primary"
    assert result.attempts == 1
