import io
import json
import logging
from types import SimpleNamespace

import pytest
from django.test import RequestFactory

from ai_core import views
from ai_core.infra import object_store, rate_limit
from common import logging as common_logging
from common.constants import (
    META_CASE_ID_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
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
def test_ping_view_applies_rate_limit(client, monkeypatch, test_tenant_schema_name):
    tenant_schema = test_tenant_schema_name
    monkeypatch.setattr(rate_limit, "get_quota", lambda: 1)
    rate_limit._get_redis.cache_clear()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: DummyRedis())

    resp1 = client.get(
        "/ai/ping/",
        **{META_CASE_ID_KEY: "c", META_TENANT_ID_KEY: tenant_schema},
    )
    assert resp1.status_code == 200
    assert resp1.json() == {"ok": True}
    assert resp1[X_TRACE_ID_HEADER]
    assert resp1[X_CASE_ID_HEADER] == "c"
    assert resp1[X_TENANT_ID_HEADER] == tenant_schema
    assert X_KEY_ALIAS_HEADER not in resp1
    resp2 = client.get(
        "/ai/ping/",
        **{META_CASE_ID_KEY: "c", META_TENANT_ID_KEY: tenant_schema},
    )
    assert resp2.status_code == 429
    assert resp2.json()["detail"] == "rate limit"
    assert X_TRACE_ID_HEADER not in resp2


@pytest.mark.django_db
def test_missing_case_header_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid case header"


@pytest.mark.django_db
def test_invalid_case_header_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "not/allowed",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid case header"


@pytest.mark.django_db
def test_tenant_schema_header_mismatch_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: "tenant-header",
            META_TENANT_SCHEMA_KEY: f"{test_tenant_schema_name}-other",
            META_CASE_ID_KEY: "c",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "tenant schema mismatch"


@pytest.mark.django_db
def test_tenant_schema_header_match_allows_request(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    seen = {}

    def _check(tenant, now=None):
        seen["tenant"] = tenant
        return True

    monkeypatch.setattr(rate_limit, "check", _check)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: "tenant-header",
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "c",
        },
    )
    assert resp.status_code == 200
    assert resp[X_TENANT_ID_HEADER] == "tenant-header"
    assert resp[X_CASE_ID_HEADER] == "c"
    assert resp.json()["tenant"] == "tenant-header"
    assert seen["tenant"] == "tenant-header"


@pytest.mark.django_db
def test_missing_tenant_resolution_returns_400(client, monkeypatch):
    monkeypatch.setattr("ai_core.views._resolve_tenant_id", lambda request: None)
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_CASE_ID_KEY: "c"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "tenant not resolved"


@pytest.mark.django_db
def test_intake_persists_state_and_headers(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    tenant_header = "tenant-header"
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: tenant_header,
            META_CASE_ID_KEY: "  case-123  ",
        },
    )
    assert resp.status_code == 200
    assert resp[X_TRACE_ID_HEADER]
    assert resp[X_CASE_ID_HEADER] == "case-123"
    assert resp[X_TENANT_ID_HEADER] == tenant_header
    assert X_KEY_ALIAS_HEADER not in resp
    assert resp.json()["tenant"] == tenant_header

    state = object_store.read_json(f"{tenant_header}/case-123/state.json")
    assert state["meta"]["tenant"] == tenant_header
    assert state["meta"]["tenant_schema"] == test_tenant_schema_name
    assert state["meta"]["case"] == "case-123"


@pytest.mark.django_db
def test_scope_and_needs_flow(client, monkeypatch, tmp_path, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "c"},
    )

    resp_scope = client.post(
        "/ai/scope/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "c"},
    )
    assert resp_scope.status_code == 200
    assert resp_scope.json()["missing"] == ["scope"]
    state = object_store.read_json(f"{test_tenant_schema_name}/c/state.json")
    assert state["missing"] == ["scope"]

    resp_needs = client.post(
        "/ai/needs/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "c"},
    )
    assert resp_needs.status_code == 200
    assert resp_needs.json()["missing"] == ["scope"]


@pytest.mark.django_db
def test_sysdesc_requires_no_missing(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    object_store.write_json(
        f"{test_tenant_schema_name}/c/state.json", {"missing": ["scope"]}
    )
    resp_skip = client.post(
        "/ai/sysdesc/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "c"},
    )
    assert resp_skip.status_code == 200
    assert resp_skip.json()["missing"] == ["scope"]

    object_store.write_json(f"{test_tenant_schema_name}/c/state.json", {"missing": []})
    resp_desc = client.post(
        "/ai/sysdesc/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "c"},
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
        **{
            META_TENANT_ID_KEY: "autotest",
            META_CASE_ID_KEY: "case-123",
            META_KEY_ALIAS_KEY: "alias-1234",
        },
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
