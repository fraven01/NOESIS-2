import json
import pytest

from ai_core.infra import rate_limit


@pytest.mark.django_db
def test_ping_ok(client, monkeypatch):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    resp = client.get(
        "/ai/ping/",
        HTTP_X_TENANT_ID="t1",
        HTTP_X_CASE_ID="c1",
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert resp["X-Trace-ID"]


@pytest.mark.django_db
def test_rag_query_missing_headers(client):
    resp = client.post(
        "/v1/ai/rag/query/",
        data=json.dumps({"question": "Ping?"}),
        content_type="application/json",
    )
    assert resp.status_code == 400

