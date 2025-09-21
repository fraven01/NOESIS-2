import pytest
from pathlib import Path

from ai_core.infra import object_store, rate_limit


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
def test_scope_missing_headers(client):
    resp = client.post("/ai/scope/", data={}, content_type="application/json")
    assert resp.status_code == 400


@pytest.mark.django_db
def test_scope_rate_limited(client, monkeypatch):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: False)
    resp = client.post(
        "/ai/scope/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID="t1",
        HTTP_X_CASE_ID="c1",
    )
    assert resp.status_code == 429


@pytest.mark.django_db
def test_scope_success_persists_state(client, monkeypatch, tmp_path):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    monkeypatch.setattr("ai_core.views._resolve_tenant_id", lambda request: "resolved")

    tenant_header = "tenant-header"
    resp = client.post(
        "/ai/scope/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=tenant_header,
        HTTP_X_CASE_ID="c1",
    )
    assert resp.status_code == 200
    assert resp["X-Trace-ID"]
    state_file = Path(tmp_path, tenant_header, "c1", "state.json")
    assert state_file.exists()
