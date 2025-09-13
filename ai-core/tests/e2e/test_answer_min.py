from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)


def _fake_llm(*args, **kwargs):
    return {"text": "", "usage": {}}


def test_answer_min(monkeypatch):
    monkeypatch.setattr("apps.llm.client.call", _fake_llm)
    monkeypatch.setattr("apps.infra.rate_limit.ready", lambda: True)
    monkeypatch.setattr("apps.infra.object_store.ready", lambda: True)

    headers = {"X-Tenant-ID": "t", "X-Case-ID": "c"}
    resp = client.post("/answer", json={"question": "hi"}, headers=headers)
    assert resp.status_code == 200

    data = resp.json()["result"]
    assert set(data.keys()) == {"answer", "citations", "gaps"}
    assert resp.headers["x-prompt-version"] == "v1"
    assert "x-trace-id" in resp.headers
