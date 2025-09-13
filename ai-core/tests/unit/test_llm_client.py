from unittest.mock import patch

from apps.llm import client


def test_client_call_routes_and_records(monkeypatch):
    """Ensure call routes to correct model and records metadata."""
    metadata = {"tenant": "t", "case": "c"}
    response_data = {
        "text": "hi",
        "usage": {"input_tokens": 1, "output_tokens": 2, "cost": 0.3},
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return response_data

    def fake_post(url, json, timeout):
        assert url == "http://example.com"
        assert json["model"] == "openai/gpt-4o"
        assert json["prompt"] == "hello"
        return FakeResponse()

    monkeypatch.setenv("LITELLM_BASE_URL", "http://example.com")

    with patch("apps.infra.ledger.record") as record_mock:
        with patch("requests.post", side_effect=fake_post):
            result = client.call("draft", "hello", metadata)

    record_mock.assert_called_once_with(metadata)
    assert result["text"] == "hi"
    assert result["usage"]["model"] == "openai/gpt-4o"
    assert result["usage"]["input_tokens"] == 1
