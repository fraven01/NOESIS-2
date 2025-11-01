import pytest
import requests

from documents.storage import InMemoryStorage


class _StubResponse:
    def __init__(self, *, status_code: int, content: bytes = b"payload") -> None:
        self.status_code = status_code
        self.content = content


def test_inmemory_storage_fetches_http_uri(monkeypatch):
    storage = InMemoryStorage(http_timeout=3.0)
    captured_timeout = {}

    def _fake_get(url, *, timeout):
        captured_timeout["value"] = timeout
        return _StubResponse(status_code=200, content=b"data")

    monkeypatch.setattr("documents.storage.requests.get", _fake_get)

    payload = storage.get("https://example.com/resource.png")

    assert payload == b"data"
    assert captured_timeout["value"] == 3.0


def test_inmemory_storage_http_error_raises_value_error(monkeypatch):
    storage = InMemoryStorage()

    def _fake_get(url, *, timeout):
        return _StubResponse(status_code=404, content=b"")

    monkeypatch.setattr("documents.storage.requests.get", _fake_get)

    with pytest.raises(ValueError) as exc:
        storage.get("https://example.com/missing.png")

    assert str(exc.value) == "storage_uri_http_error"


def test_inmemory_storage_http_request_exception(monkeypatch):
    storage = InMemoryStorage()

    def _fake_get(url, *, timeout):
        raise requests.RequestException("boom")

    monkeypatch.setattr("documents.storage.requests.get", _fake_get)

    with pytest.raises(ValueError) as exc:
        storage.get("https://example.com/fails.png")

    assert str(exc.value) == "storage_uri_fetch_failed"
