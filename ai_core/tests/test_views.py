import io
import json
import logging
from types import SimpleNamespace

import pytest
from django.test import RequestFactory

from ai_core import views
from ai_core.infra import object_store, rate_limit
from common import logging as common_logging
from common.middleware import RequestLogContextMiddleware


class DummyRedis:
    def __init__(self):
        self.store = {}

    def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def expire(self, key, ttl):
        return None


@pytest.mark.django_db
def test_ping_view_applies_rate_limit(
    client, monkeypatch, test_tenant_schema_name
):
    tenant_schema = test_tenant_schema_name
    monkeypatch.setattr(rate_limit, "get_quota", lambda: 1)
    rate_limit._get_redis.cache_clear()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: DummyRedis())

    resp1 = client.get(
        "/ai/ping/",
        HTTP_X_CASE_ID="c",
        HTTP_X_TENANT_ID=tenant_schema,
    )
    assert resp1.status_code == 200
    assert resp1.json() == {"ok": True}
    assert resp1["X-Trace-ID"]
    assert resp1["X-Case-ID"] == "c"
    assert resp1["X-Tenant-ID"] == tenant_schema
    assert "X-Key-Alias" not in resp1
    resp2 = client.get(
        "/ai/ping/",
        HTTP_X_CASE_ID="c",
        HTTP_X_TENANT_ID=tenant_schema,
    )
    assert resp2.status_code == 429
    assert resp2.json()["detail"] == "rate limit"
    assert "X-Trace-ID" not in resp2


@pytest.mark.django_db
def test_missing_case_header_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid case header"


@pytest.mark.django_db
def test_invalid_case_header_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="not/allowed",
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid case header"


@pytest.mark.django_db
def test_tenant_header_mismatch_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=f"{test_tenant_schema_name}-other",
        HTTP_X_CASE_ID="c",
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "tenant mismatch"


@pytest.mark.django_db
def test_missing_tenant_resolution_returns_400(client, monkeypatch):
    monkeypatch.setattr("ai_core.views._resolve_tenant_id", lambda request: None)
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_CASE_ID="c",
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "tenant not resolved"


@pytest.mark.django_db
def test_intake_persists_state_and_headers(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="  case-123  ",
    )
    assert resp.status_code == 200
    assert resp["X-Trace-ID"]
    assert resp["X-Case-ID"] == "case-123"
    assert resp["X-Tenant-ID"] == test_tenant_schema_name
    assert "X-Key-Alias" not in resp
    assert resp.json()["tenant"] == test_tenant_schema_name

    state = object_store.read_json(f"{test_tenant_schema_name}/case-123/state.json")
    assert state["meta"]["tenant"] == test_tenant_schema_name
    assert state["meta"]["case"] == "case-123"


@pytest.mark.django_db
def test_scope_and_needs_flow(client, monkeypatch, tmp_path, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="c",
    )

    resp_scope = client.post(
        "/ai/scope/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="c",
    )
    assert resp_scope.status_code == 200
    assert resp_scope.json()["missing"] == ["scope"]
    state = object_store.read_json(f"{test_tenant_schema_name}/c/state.json")
    assert state["missing"] == ["scope"]

    resp_needs = client.post(
        "/ai/needs/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="c",
    )
    assert resp_needs.status_code == 200
    assert resp_needs.json()["missing"] == ["scope"]


@pytest.mark.django_db
def test_sysdesc_requires_no_missing(client, monkeypatch, tmp_path, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    object_store.write_json(
        f"{test_tenant_schema_name}/c/state.json", {"missing": ["scope"]}
    )
    resp_skip = client.post(
        "/ai/sysdesc/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="c",
    )
    assert resp_skip.status_code == 200
    assert resp_skip.json()["missing"] == ["scope"]

    object_store.write_json(
        f"{test_tenant_schema_name}/c/state.json", {"missing": []}
    )
    resp_desc = client.post(
        "/ai/sysdesc/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID=test_tenant_schema_name,
        HTTP_X_CASE_ID="c",
    )
    assert resp_desc.status_code == 200
    body = resp_desc.json()
    assert "description" in body


def test_request_logging_context_includes_metadata(
    monkeypatch,
    tmp_path,
    settings,
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.addFilter(common_logging.RequestTaskContextFilter())
    formatter = logging.Formatter(settings.LOGGING["formatters"]["verbose"]["format"])
    handler.setFormatter(formatter)

    logger = logging.getLogger("ai_core.tests.graph")
    original_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    class _LoggingGraph:
        def run(self, state, meta):
            context = common_logging.get_log_context()
            assert context["trace_id"] == meta["trace_id"]
            assert context["case_id"] == meta["case"]
            assert context["tenant"] == meta["tenant"]
            assert context.get("key_alias") == meta.get("key_alias")
            logger.info("graph-run")
            return state, {"ok": True}

    monkeypatch.setattr(views, "info_intake", _LoggingGraph())

    factory = RequestFactory()
    request = factory.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        HTTP_X_TENANT_ID="autotest",
        HTTP_X_CASE_ID="case-123",
        HTTP_X_KEY_ALIAS="alias-1234",
    )
    request.tenant = SimpleNamespace(schema_name="autotest")

    middleware = RequestLogContextMiddleware(views.intake)

    try:
        resp = middleware(request)
    finally:
        logger.removeHandler(handler)
        logger.propagate = original_propagate

    assert resp.status_code == 200

    output = log_buffer.getvalue()
    assert "graph-run" in output
    assert "trace=-" not in output
    assert "case=-" not in output
    assert "tenant=-" not in output
    assert "key_alias=-" not in output
    assert common_logging.get_log_context() == {}


def test_state_helpers_sanitize_identifiers(monkeypatch, tmp_path):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    views._save_state("Tenant Name", "Case*ID", {"ok": True})

    safe_tenant = object_store.sanitize_identifier("Tenant Name")
    safe_case = object_store.sanitize_identifier("Case*ID")

    unsafe_path = tmp_path / "Tenant Name"
    assert not unsafe_path.exists()

    stored = tmp_path / safe_tenant / safe_case / "state.json"
    assert stored.exists()
    assert json.loads(stored.read_text()) == {"ok": True}

    loaded = views._load_state("Tenant Name", "Case*ID")
    assert loaded == {"ok": True}


def test_state_helpers_reject_unsafe_identifiers(monkeypatch, tmp_path):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    with pytest.raises(ValueError):
        views._save_state("tenant/../", "case", {})

    with pytest.raises(ValueError):
        views._load_state("tenant", "../case")
