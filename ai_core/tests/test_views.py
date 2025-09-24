import json
from types import SimpleNamespace

import pytest
from rest_framework.response import Response
from django.conf import settings
from django.test import RequestFactory
from structlog.testing import capture_logs

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
    error_body = resp2.json()
    assert error_body["detail"] == "Rate limit exceeded for tenant."
    assert error_body["code"] == "rate_limit_exceeded"
    assert X_TRACE_ID_HEADER not in resp2


@pytest.mark.django_db
def test_v1_ping_does_not_require_authorization(
    client, test_tenant_schema_name
):
    response = client.get(
        "/v1/ai/ping/",
        **{META_CASE_ID_KEY: "case-auth", META_TENANT_ID_KEY: test_tenant_schema_name},
    )

    assert response.status_code == 200
    assert "WWW-Authenticate" not in response


@pytest.mark.django_db
def test_missing_case_header_returns_400(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name},
    )
    assert resp.status_code == 400
    error_body = resp.json()
    assert error_body["detail"] == "Case header is required and must use the documented format."
    assert error_body["code"] == "invalid_case_header"


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
    error_body = resp.json()
    assert error_body["detail"] == "Case header is required and must use the documented format."
    assert error_body["code"] == "invalid_case_header"


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
    error_body = resp.json()
    assert error_body["detail"] == "Tenant schema header does not match resolved schema."
    assert error_body["code"] == "tenant_schema_mismatch"


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
    error_body = resp.json()
    assert error_body["detail"] == "Tenant schema could not be resolved from headers."
    assert error_body["code"] == "tenant_not_found"


@pytest.mark.django_db
def test_non_json_payload_returns_415(client, test_tenant_schema_name):
    resp = client.post(
        "/ai/intake/",
        data="raw body",
        content_type="text/plain",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "case"},
    )

    assert resp.status_code == 415
    error_body = resp.json()
    assert error_body["detail"] == "Request payload must be encoded as application/json."
    assert error_body["code"] == "unsupported_media_type"

    v1_response = client.post(
        "/v1/ai/intake/",
        data="raw body",
        content_type="text/plain",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "case"},
    )

    assert v1_response.status_code == 415
    v1_error = v1_response.json()
    assert v1_error["detail"] == "Request payload must be encoded as application/json."
    assert v1_error["code"] == "unsupported_media_type"


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


def test_request_logging_context_includes_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    logger = common_logging.get_logger("ai_core.tests.graph")

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

    with capture_logs() as logs:
        resp = middleware(request)

    assert resp.status_code == 200

    events = [entry for entry in logs if entry.get("event") == "graph-run"]
    assert events, "expected graph-run log entry"
    event = events[0]
    assert event["trace_id"] != "-"
    assert event["case_id"] != "-"
    assert event["tenant"] != "-"
    assert event.get("key_alias", "") != "-"
    assert common_logging.get_log_context() == {}


@pytest.mark.django_db
def test_legacy_routes_emit_deprecation_headers(client, test_tenant_schema_name):
    headers = {META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "case-legacy"}

    legacy_response = client.get("/ai/ping/", **headers)
    assert legacy_response.status_code == 200
    assert legacy_response["Deprecation"] == settings.API_DEPRECATIONS["ai-core-legacy"][
        "deprecation"
    ]
    assert legacy_response["Sunset"] == settings.API_DEPRECATIONS["ai-core-legacy"]["sunset"]

    v1_response = client.get("/v1/ai/ping/", **headers)
    assert v1_response.status_code == 200
    assert "Deprecation" not in v1_response
    assert "Sunset" not in v1_response


@pytest.mark.django_db
def test_legacy_post_routes_emit_deprecation_headers(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(views, "_run_graph", lambda request, graph: Response({"ok": True}))

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "post-legacy",
    }

    legacy_response = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **headers,
    )
    assert legacy_response.status_code == 200
    assert "Deprecation" in legacy_response
    assert "Sunset" in legacy_response

    v1_response = client.post(
        "/v1/ai/intake/",
        data={},
        content_type="application/json",
        **headers,
    )
    assert v1_response.status_code == 200
    assert "Deprecation" not in v1_response
    assert "Sunset" not in v1_response


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
