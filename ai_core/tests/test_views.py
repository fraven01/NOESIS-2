import json
import uuid
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from rest_framework.request import Request
from rest_framework.response import Response

from django.conf import settings
from django.test import RequestFactory
from structlog.testing import capture_logs

from ai_core import services, views
from ai_core.graph.core import GraphContext
from ai_core.graph.schemas import ToolContext
from ai_core.infra import object_store, rate_limit
from ai_core.graph import registry
import ai_core.nodes.retrieve as retrieve
from ai_core.nodes.retrieve import RetrieveInput
from ai_core.schemas import CrawlerRunRequest, RagQueryRequest
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_client import HybridSearchResult
from ai_core.rag.guardrails import GuardrailLimits, GuardrailSignals
from ai_core.tool_contracts import InconsistentMetadataError, NotFoundError
from common import logging as common_logging
from common.constants import (
    COLLECTION_ID_HEADER_CANDIDATES,
    IDEMPOTENCY_KEY_HEADER,
    META_COLLECTION_ID_KEY,
    META_CASE_ID_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
from common.middleware import RequestLogContextMiddleware


@pytest.fixture(autouse=True)
def _reset_log_context():
    try:
        yield
    finally:
        common_logging.clear_log_context()


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
def test_v1_ping_does_not_require_authorization(client, test_tenant_schema_name):
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
    assert (
        error_body["detail"]
        == "Case header is required and must use the documented format."
    )
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
    assert (
        error_body["detail"]
        == "Case header is required and must use the documented format."
    )
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
    assert (
        error_body["detail"] == "Tenant schema header does not match resolved schema."
    )
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
    assert resp.json()["tenant_id"] == "tenant-header"
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
    assert (
        error_body["detail"] == "Request payload must be encoded as application/json."
    )
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
def test_intake_rejects_invalid_metadata_type(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    response = client.post(
        "/ai/intake/",
        data=json.dumps({"metadata": ["not", "an", "object"]}),
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "case"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_request"
    assert "metadata must be a JSON object" in body["detail"]


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
    assert resp.json()["tenant_id"] == tenant_header

    state = object_store.read_json(f"{tenant_header}/case-123/state.json")
    assert state["meta"]["tenant_id"] == tenant_header
    assert state["meta"]["tenant_schema"] == test_tenant_schema_name
    assert state["meta"]["case_id"] == "case-123"


def test_request_logging_context_includes_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    logger = common_logging.get_logger("ai_core.tests.graph")

    class _LoggingGraph:
        def run(self, state, meta):
            context = common_logging.get_log_context()
            assert context["trace_id"] == meta["trace_id"]
            assert context["case_id"] == meta["case_id"]
            assert context["tenant"] == meta["tenant_id"]
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
    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-legacy",
    }

    legacy_response = client.get("/ai/ping/", **headers)
    assert legacy_response.status_code == 200
    assert (
        legacy_response["Deprecation"]
        == settings.API_DEPRECATIONS["ai-core-legacy"]["deprecation"]
    )
    assert (
        legacy_response["Sunset"]
        == settings.API_DEPRECATIONS["ai-core-legacy"]["sunset"]
    )

    v1_response = client.get("/v1/ai/ping/", **headers)
    assert v1_response.status_code == 200
    assert "Deprecation" not in v1_response
    assert "Sunset" not in v1_response


@pytest.mark.django_db
def test_legacy_post_routes_emit_deprecation_headers(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(
        views, "_run_graph", lambda request, graph: Response({"ok": True})
    )

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


def _graph_context(tenant: str, case_id: str) -> GraphContext:
    return GraphContext(
        tenant_id=tenant,
        case_id=case_id,
        trace_id="test-trace",
        graph_name="info_intake",
    )


def test_state_helpers_sanitize_identifiers(monkeypatch, tmp_path):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    ctx = _graph_context("Tenant Name", "Case*ID")
    views.CHECKPOINTER.save(ctx, {"ok": True})

    safe_tenant = object_store.sanitize_identifier("Tenant Name")
    safe_case = object_store.sanitize_identifier("Case*ID")

    unsafe_path = tmp_path / "Tenant Name"
    assert not unsafe_path.exists()

    stored = tmp_path / safe_tenant / safe_case / "state.json"
    assert stored.exists()
    assert json.loads(stored.read_text()) == {"ok": True}

    loaded = views.CHECKPOINTER.load(ctx)
    assert loaded == {"ok": True}


def test_state_helpers_reject_unsafe_identifiers(monkeypatch, tmp_path):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    with pytest.raises(ValueError):
        views.CHECKPOINTER.save(_graph_context("tenant/../", "case"), {})

    with pytest.raises(ValueError):
        views.CHECKPOINTER.load(_graph_context("tenant", "../case"))


def test_state_helpers_recover_from_corrupted_checkpoint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    ctx = _graph_context("Tenant Name", "Case*ID")
    checkpoint_path = tmp_path / views.CHECKPOINTER._path(ctx)  # type: ignore[attr-defined]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{invalid json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        loaded = views.CHECKPOINTER.load(ctx)

    assert loaded == {}
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {}
    assert "graph.checkpoint.corrupted" in caplog.text


def test_graph_view_lazy_registration_reuses_cached_runner(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})

    module = ModuleType("ai_core.graphs.lazy_mod")

    def _run(state, meta):
        return state or {}, {"result": True, "meta": meta["graph_name"]}

    module.run = _run
    monkeypatch.setattr(views, "lazy_mod", module, raising=False)

    calls: list[ModuleType] = []

    def _module_runner(mod):
        calls.append(mod)
        return SimpleNamespace(run=lambda state, meta: (state, {"ok": True}))

    monkeypatch.setattr(views, "module_runner", _module_runner)

    view_cls = type("LazyView", (views._GraphView,), {"graph_name": "lazy_mod"})
    view = view_cls()

    first = view.get_graph()
    second = view.get_graph()

    assert first is second
    assert calls == [module]
    assert registry.get("lazy_mod") is first


@pytest.mark.django_db
def test_graph_view_missing_runner_returns_server_error(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delattr("ai_core.views.info_intake", raising=False)
    client.raise_request_exception = False

    response = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: test_tenant_schema_name, META_CASE_ID_KEY: "case"},
    )

    assert response.status_code == 500
    common_logging.clear_log_context()


@pytest.mark.django_db
def test_graph_view_propagates_tool_context(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr("ai_core.graph.schemas.get_quota", lambda: 1)

    captured_meta: dict[str, object] = {}
    task_calls: list[object] = []
    captured_request: dict[str, Request] = {}

    original_normalize_meta = views.normalize_meta

    def _capturing_normalize_meta(request: Request):
        captured_request["request"] = request
        return original_normalize_meta(request)

    class _DummyRunner:
        def run(self, state, meta):
            captured_meta["meta"] = meta
            task_calls.append(meta["tool_context"])
            return state or {}, {"ok": True}

    monkeypatch.setattr(views, "normalize_meta", _capturing_normalize_meta)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: _DummyRunner())

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-tool",
        "HTTP_IDEMPOTENCY_KEY": "idem-123",
    }

    response = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    tool_context = captured_meta["meta"]["tool_context"]
    assert isinstance(tool_context, dict)
    context = ToolContext.model_validate(tool_context)
    assert context.tenant_id == test_tenant_schema_name
    assert context.case_id == "case-tool"
    assert context.idempotency_key == "idem-123"
    assert context.trace_id == captured_meta["meta"]["trace_id"]
    assert context.metadata["graph_name"] == captured_meta["meta"]["graph_name"]
    assert context.metadata["graph_version"] == captured_meta["meta"]["graph_version"]
    assert context.metadata["requested_at"] == captured_meta["meta"]["requested_at"]
    assert context.metadata["requested_at"]
    assert ToolContext.model_validate(task_calls[0]) == context

    request_obj = captured_request["request"]
    assert isinstance(request_obj.tool_context, ToolContext)
    assert request_obj.tool_context.tenant_id == test_tenant_schema_name
    assert request_obj.tool_context.case_id == "case-tool"
    assert request_obj.tool_context.idempotency_key == "idem-123"
    assert response.headers.get(IDEMPOTENCY_KEY_HEADER) == "idem-123"


@pytest.mark.django_db
def test_response_includes_key_alias_header(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    response = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-alias",
            META_KEY_ALIAS_KEY: "alias-1",
        },
    )

    assert response.status_code == 200
    assert response[X_KEY_ALIAS_HEADER] == "alias-1"
    common_logging.clear_log_context()


@pytest.mark.django_db
def test_v1_intake_rate_limit_returns_json_error(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "get_quota", lambda: 1)
    rate_limit._get_redis.cache_clear()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: DummyRedis())
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-rate",
    }

    first = client.post(
        "/v1/ai/intake/",
        data={},
        content_type="application/json",
        **headers,
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/ai/intake/",
        data={},
        content_type="application/json",
        **headers,
    )

    assert second.status_code == 429
    body = second.json()
    assert body == {
        "detail": "Rate limit exceeded for tenant.",
        "code": "rate_limit_exceeded",
    }
    common_logging.clear_log_context()


@pytest.mark.django_db
def test_corrupted_state_file_returns_400(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    tenant = test_tenant_schema_name
    case = "case-corrupt"
    safe_tenant = object_store.sanitize_identifier(tenant)
    safe_case = object_store.sanitize_identifier(case)
    state_path = tmp_path / safe_tenant / safe_case / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("[]", encoding="utf-8")

    response = client.post(
        "/ai/intake/",
        data={},
        content_type="application/json",
        **{META_TENANT_ID_KEY: tenant, META_CASE_ID_KEY: case},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    common_logging.clear_log_context()


@pytest.mark.django_db
def test_ingestion_run_rejects_blank_document_ids(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps({"document_ids": ["  "], "embedding_profile": "standard"}),
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-ingest",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "document_ids" in body["detail"]


@pytest.mark.django_db
def test_ingestion_run_rejects_empty_embedding_profile(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    valid_document_id = str(uuid.uuid4())
    response = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps(
            {"document_ids": [valid_document_id], "embedding_profile": "   "}
        ),
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-ingest",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "embedding_profile" in body["detail"]


@pytest.mark.django_db
def test_ingestion_run_normalises_payload_before_dispatch(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    captured: dict[str, object] = {}

    def _fake_partition(tenant, case, document_ids):
        captured["partition_args"] = (tenant, case, list(document_ids))
        return ["doc-trimmed"], []

    fake_vector = SimpleNamespace(
        id="vector-space",
        schema="rag",
        backend="pgvector",
        dimension=1536,
    )
    fake_resolution = SimpleNamespace(vector_space=fake_vector)
    fake_binding = SimpleNamespace(profile_id="standard", resolution=fake_resolution)

    def _fake_resolve(profile: str):
        captured["resolved_profile_input"] = profile
        return fake_binding

    dispatched: dict[str, object] = {}

    def _fake_delay(tenant, case, document_ids, profile_id, **kwargs):
        dispatched["args"] = (tenant, case, document_ids, profile_id)
        dispatched["kwargs"] = kwargs

    monkeypatch.setattr(views, "partition_document_ids", _fake_partition)
    monkeypatch.setattr(views, "resolve_ingestion_profile", _fake_resolve)
    monkeypatch.setattr(views.run_ingestion, "delay", _fake_delay)

    raw_id = str(uuid.uuid4())
    response = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps(
            {
                "document_ids": [f" {raw_id} "],
                "embedding_profile": " standard ",
            }
        ),
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-ingest",
        },
    )

    assert response.status_code == 202
    assert captured["resolved_profile_input"] == "standard"
    assert captured["partition_args"][2] == [raw_id]
    assert dispatched["args"][2] == ["doc-trimmed"]
    assert dispatched["args"][3] == "standard"
    assert dispatched["kwargs"]["tenant_schema"] == test_tenant_schema_name


def test_rag_query_request_collection_list_overrides_body():
    list_id_one = str(uuid.uuid4())
    list_id_two = str(uuid.uuid4())
    body_collection = str(uuid.uuid4())

    request_model = RagQueryRequest(
        question="Welche Daten liegen vor?",
        hybrid={"alpha": 0.7},
        collection_id=body_collection,
        filters={"collection_ids": [list_id_one, list_id_two]},
    )

    assert request_model.collection_id == body_collection
    assert request_model.filters == {
        "collection_id": body_collection,
        "collection_ids": [body_collection, list_id_one, list_id_two],
    }


def test_rag_query_request_applies_body_collection_when_no_list():
    collection_value = str(uuid.uuid4())

    request_model = RagQueryRequest(
        question="Welche Daten liegen vor?",
        hybrid={"alpha": 0.7},
        collection_id=collection_value,
        filters={},
    )

    assert request_model.collection_id == collection_value
    assert request_model.filters == {
        "collection_id": collection_value,
        "collection_ids": [collection_value],
    }


def test_collection_header_bridge_respects_priority():
    header_value = str(uuid.uuid4())
    body_value = str(uuid.uuid4())
    list_value = [str(uuid.uuid4())]

    base_payload = {"collection_id": body_value, "filters": {}}

    request = SimpleNamespace(headers={X_COLLECTION_ID_HEADER: header_value}, META={})
    bridged = services._apply_collection_header_bridge(request, base_payload)
    assert bridged["collection_id"] == body_value

    request.headers[X_COLLECTION_ID_HEADER] = header_value
    payload_with_list = {
        "filters": {"collection_ids": list_value},
        "collection_id": body_value,
    }
    bridged_with_list = services._apply_collection_header_bridge(
        request, payload_with_list
    )
    assert bridged_with_list.get("collection_id") == body_value
    assert bridged_with_list["filters"]["collection_ids"] == list_value

    request.headers[X_COLLECTION_ID_HEADER] = header_value
    payload_only_list = {"filters": {"collection_ids": list_value}}
    bridged_only_list = services._apply_collection_header_bridge(
        request, payload_only_list
    )
    assert "collection_id" not in bridged_only_list
    assert bridged_only_list["filters"]["collection_ids"] == list_value

    payload_missing = {"filters": {}}
    bridged_missing = services._apply_collection_header_bridge(request, payload_missing)
    assert bridged_missing["collection_id"] == header_value


def test_collection_scope_priority_end_to_end():
    header_value = str(uuid.uuid4())
    body_value = str(uuid.uuid4())
    list_value = [str(uuid.uuid4()), str(uuid.uuid4())]

    request = SimpleNamespace(
        headers={X_COLLECTION_ID_HEADER: header_value},
        META={},
    )

    payload = {
        "question": "Welche Daten liegen vor?",
        "hybrid": {"alpha": 0.7},
        "collection_id": body_value,
        "filters": {"collection_ids": list_value},
    }

    bridged_payload = services._apply_collection_header_bridge(request, payload)
    request_model = RagQueryRequest(**bridged_payload)

    assert request_model.collection_id == body_value
    assert request_model.filters == {
        "collection_id": body_value,
        "collection_ids": [body_value, *list_value],
    }


def test_collection_header_bridge_accepts_candidate_headers():
    alternate_header = COLLECTION_ID_HEADER_CANDIDATES[1]
    header_value = str(uuid.uuid4())

    request = SimpleNamespace(headers={alternate_header: header_value}, META={})
    bridged = services._apply_collection_header_bridge(request, {})

    assert bridged["collection_id"] == header_value


def test_collection_header_bridge_uses_meta_fallback():
    header_value = str(uuid.uuid4())

    request = SimpleNamespace(headers={}, META={META_COLLECTION_ID_KEY: header_value})
    bridged = services._apply_collection_header_bridge(request, {})

    assert bridged["collection_id"] == header_value


@pytest.mark.django_db
def test_rag_query_endpoint_builds_tool_context_and_retrieve_input(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    tenant_id = "tenant-rag"
    case_id = "case-rag-001"
    payload = {
        "question": " Welche Richtlinien gelten? ",
        "query": " travel policy ",
        "filters": {"tags": ["travel"]},
        "process": "travel",
        "doc_class": "policy",
        "visibility": "tenant",
        "visibility_override_allowed": True,
        "hybrid": {"alpha": 0.7},
    }

    retrieval_payload = {
        "alpha": 0.7,
        "min_sim": 0.15,
        "top_k_effective": 1,
        "matches_returned": 1,
        "max_candidates_effective": 50,
        "vector_candidates": 37,
        "lexical_candidates": 41,
        "deleted_matches_blocked": 0,
        "visibility_effective": "active",
        "took_ms": 42,
        "routing": {
            "profile": "standard",
            "vector_space_id": "rag/global",
        },
    }
    snippets_payload = [
        {
            "id": "doc-871#p3",
            "text": "Rücksendungen sind innerhalb von 30 Tagen möglich, sofern das Produkt unbenutzt ist.",
            "score": 0.82,
            "source": "policies/returns.md",
            "hash": "7f3d6a2c",
        }
    ]

    recorded: dict[str, object] = {}

    class DummyCheckpointer:
        def __init__(self) -> None:
            self.saved_state: dict[str, Any] | None = None

        def load(self, ctx):
            recorded["load_ctx"] = ctx
            return {}

        def save(self, ctx, state):
            recorded["save_ctx"] = ctx
            state_copy = dict(state)
            self.saved_state = state_copy
            recorded["saved_state"] = state_copy

    dummy_checkpointer = DummyCheckpointer()
    monkeypatch.setattr(views, "CHECKPOINTER", dummy_checkpointer)

    def _run(state, meta):
        tool_context_payload = meta.get("tool_context")
        context = ToolContext.model_validate(tool_context_payload)
        recorded["tool_context"] = context
        params = RetrieveInput.from_state(state)
        recorded["params"] = params
        recorded["state_before"] = dict(state)
        augmented_state = dict(state)
        augmented_state.update(
            {
                "snippets": snippets_payload,
                "matches": snippets_payload,
                "retrieval": retrieval_payload,
                "answer": "Synthesised",
            }
        )
        recorded["final_state"] = dict(augmented_state)
        return augmented_state, {
            "answer": "Synthesised",
            "prompt_version": "v1",
            "retrieval": dict(retrieval_payload),
            "snippets": [dict(snippets_payload[0])],
        }

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={**payload, "collection_id": str(uuid.uuid4())},
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: tenant_id,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: case_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "answer": "Synthesised",
        "prompt_version": "v1",
        "retrieval": retrieval_payload,
        "snippets": snippets_payload,
    }
    assert payload["answer"] == "Synthesised"
    assert payload["prompt_version"] == "v1"
    retrieval = payload["retrieval"]
    assert retrieval["alpha"] == 0.7
    assert retrieval["min_sim"] == 0.15
    assert retrieval["top_k_effective"] == 1
    assert retrieval["matches_returned"] == 1
    assert retrieval["visibility_effective"] == "active"
    assert retrieval["took_ms"] == 42
    assert retrieval["routing"]["profile"] == "standard"
    assert retrieval["routing"]["vector_space_id"] == "rag/global"
    snippet = payload["snippets"][0]
    assert isinstance(snippet["score"], float)
    assert snippet["text"]
    assert snippet["source"]

    tool_context = recorded["tool_context"]
    assert isinstance(tool_context, ToolContext)
    assert tool_context.tenant_id == tenant_id
    assert tool_context.case_id == case_id
    assert tool_context.metadata.get("graph_name") == "rag.default"

    params = recorded["params"]
    assert isinstance(params, RetrieveInput)
    assert params.query == "travel policy"
    filters = dict(params.filters or {})
    assert filters.get("tags") == ["travel"]
    assert params.process == "travel"
    assert params.doc_class == "policy"
    assert params.visibility == "tenant"
    assert params.visibility_override_allowed is True
    assert params.hybrid == {"alpha": 0.7}

    state_before = recorded["state_before"]
    assert state_before["question"] == "Welche Richtlinien gelten?"
    assert state_before["query"] == "travel policy"

    final_state = recorded["final_state"]
    retrieval_state = final_state["retrieval"]
    assert retrieval_state["alpha"] == 0.7
    assert retrieval_state["min_sim"] == 0.15
    assert retrieval_state["top_k_effective"] == 1
    assert retrieval_state["matches_returned"] == 1
    assert retrieval_state["visibility_effective"] == "active"
    assert retrieval_state["took_ms"] == 42
    assert retrieval_state["routing"]["profile"] == "standard"
    assert retrieval_state["routing"]["vector_space_id"] == "rag/global"
    snippet_state = final_state["snippets"][0]
    assert isinstance(snippet_state["score"], float)
    assert snippet_state["text"]
    assert snippet_state["source"]
    assert dummy_checkpointer.saved_state == final_state


@pytest.mark.django_db
def test_rag_query_endpoint_rejects_invalid_graph_payload(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    def _run(state, meta):
        return state, {"answer": "incomplete", "prompt_version": "v1"}

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": "Was gilt?",
            "hybrid": {"alpha": 0.3},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-invalid",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "retrieval" in payload
    assert payload["retrieval"] == ["This field is required."]
    assert "snippets" in payload
    assert payload["snippets"] == ["This field is required."]


@pytest.mark.django_db
def test_rag_query_endpoint_allows_blank_answer(
    client, monkeypatch, test_tenant_schema_name
):
    """Ensure an empty composed answer is accepted by the contract validator."""

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    retrieval_payload = {
        "alpha": 0.7,
        "min_sim": 0.15,
        "top_k_effective": 1,
        "matches_returned": 1,
        "max_candidates_effective": 50,
        "vector_candidates": 2,
        "lexical_candidates": 1,
        "deleted_matches_blocked": 0,
        "visibility_effective": "active",
        "took_ms": 42,
        "routing": {"profile": "standard", "vector_space_id": "rag/global"},
    }
    snippets_payload = [
        {
            "id": "doc-1#p1",
            "text": "Relevanter Abschnitt",
            "score": 0.42,
            "source": "policies/travel.md",
            "hash": "deadbeef",
        }
    ]

    def _run(state, meta):
        new_state = dict(state)
        new_state["retrieval"] = retrieval_payload
        new_state["snippets"] = snippets_payload
        payload = {
            "answer": "",
            "prompt_version": "v1",
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
        }
        return new_state, payload

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": "Was gilt?",
            "hybrid": {"alpha": 0.3},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-blank-answer",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == ""
    assert payload["prompt_version"] == "v1"
    assert payload["retrieval"]["vector_candidates"] == 2
    assert payload["snippets"][0]["source"] == "policies/travel.md"


@pytest.mark.django_db
def test_rag_query_endpoint_rejects_missing_prompt_version(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    retrieval_payload = {
        "alpha": 0.5,
        "min_sim": 0.1,
        "top_k_effective": 1,
        "matches_returned": 1,
        "max_candidates_effective": 10,
        "vector_candidates": 4,
        "lexical_candidates": 3,
        "deleted_matches_blocked": 0,
        "visibility_effective": "active",
        "took_ms": 17,
        "routing": {"profile": "standard", "vector_space_id": "rag/global"},
    }
    snippets_payload = [
        {
            "id": "doc-1",
            "text": "Snippet",
            "score": 0.75,
            "source": "handbook.md",
        }
    ]

    def _run(state, meta):
        augmented = dict(state)
        augmented.update(
            {
                "snippets": snippets_payload,
                "matches": snippets_payload,
                "retrieval": retrieval_payload,
                "answer": "Synthesised",
            }
        )
        return augmented, {
            "answer": "Synthesised",
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
        }

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": "Was gilt?",
            "hybrid": {"alpha": 0.3},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-missing-prompt-version",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert "prompt_version" in payload
    assert payload["prompt_version"] == ["This field is required."]


@pytest.mark.django_db
def test_rag_query_endpoint_normalises_numeric_types(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    retrieval_payload = {
        "alpha": "0.65",
        "min_sim": "0.35",
        "top_k_effective": "2",
        "matches_returned": "1",
        "max_candidates_effective": "20",
        "vector_candidates": "11",
        "lexical_candidates": "7",
        "deleted_matches_blocked": "1",
        "visibility_effective": "active",
        "took_ms": "21",
        "routing": {"profile": "standard", "vector_space_id": "rag/global"},
    }
    snippets_payload = [
        {
            "id": "doc-2",
            "text": "Some snippet",
            "score": "0.82",
            "source": "faq.md",
        }
    ]

    def _run(state, meta):
        augmented = dict(state)
        augmented.update(
            {
                "snippets": snippets_payload,
                "matches": snippets_payload,
                "retrieval": retrieval_payload,
                "answer": "Typed",
            }
        )
        return augmented, {
            "answer": "Typed",
            "prompt_version": "v2",
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
        }

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": "Was gilt?",
            "hybrid": {"alpha": 0.5},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-types",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    retrieval = payload["retrieval"]
    assert isinstance(retrieval["top_k_effective"], int)
    assert retrieval["top_k_effective"] == 2
    assert isinstance(retrieval["matches_returned"], int)
    assert retrieval["matches_returned"] == 1
    assert isinstance(retrieval["max_candidates_effective"], int)
    assert retrieval["max_candidates_effective"] == 20
    assert isinstance(retrieval["vector_candidates"], int)
    assert retrieval["vector_candidates"] == 11
    assert isinstance(retrieval["lexical_candidates"], int)
    assert retrieval["lexical_candidates"] == 7
    assert isinstance(retrieval["deleted_matches_blocked"], int)
    assert retrieval["deleted_matches_blocked"] == 1
    assert isinstance(retrieval["took_ms"], int)
    assert retrieval["took_ms"] == 21
    assert isinstance(retrieval["alpha"], float)
    assert retrieval["alpha"] == pytest.approx(0.65)
    assert isinstance(retrieval["min_sim"], float)
    assert retrieval["min_sim"] == pytest.approx(0.35)

    snippet = payload["snippets"][0]
    assert isinstance(snippet["score"], float)
    assert snippet["score"] == pytest.approx(0.82)


@pytest.mark.django_db
def test_rag_query_endpoint_applies_top_k_override(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(views, "assert_case_active", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("ai_core.nodes._hybrid_params.TOPK_DEFAULT", 1)

    class DummyCheckpointer:
        def load(self, _context):
            return {}

        def save(self, _context, _state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    class _Router:
        def __init__(self):
            self.calls: list[int] = []

        def hybrid_search(
            self,
            query,
            *,
            case_id=None,
            top_k=5,
            **_kwargs,
        ):
            self.calls.append(int(top_k))
            tenant_meta = test_tenant_schema_name
            case_meta = case_id or "case-topk"
            chunks = [
                Chunk(
                    f"Snippet {index}",
                    {
                        "id": f"doc-{index}",
                        "score": 0.9 - (index * 0.01),
                        "source": f"doc-{index}.md",
                        "tenant_id": tenant_meta,
                        "case_id": case_meta,
                    },
                )
                for index in range(5)
            ]
            return HybridSearchResult(
                chunks=chunks,
                vector_candidates=len(chunks),
                lexical_candidates=len(chunks),
                fused_candidates=len(chunks),
                duration_ms=1.0,
                alpha=0.7,
                min_sim=0.15,
                vec_limit=10,
                lex_limit=10,
            )

    router = _Router()
    monkeypatch.setattr(retrieve, "_ROUTER", None)
    monkeypatch.setattr(retrieve, "_get_router", lambda: router)

    def _run_graph(request_obj, _graph):
        body = request_obj.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        state = json.loads(body or "{}")
        context = ToolContext(
            tenant_id=request_obj.META.get(META_TENANT_ID_KEY, ""),
            tenant_schema=request_obj.META.get(META_TENANT_SCHEMA_KEY),
            case_id=request_obj.META.get(META_CASE_ID_KEY, ""),
            trace_id="test-trace",
            metadata={"graph_name": "rag.default", "graph_version": "test"},
        )
        params = retrieve.RetrieveInput.from_state(state)
        output = retrieve.run(context, params)
        retrieval_meta = output.meta.model_dump(mode="json", exclude_none=True)
        payload = {
            "answer": "Stub answer",
            "prompt_version": "test",
            "retrieval": retrieval_meta,
            "snippets": output.matches,
        }
        return Response(payload)

    monkeypatch.setattr(views, "_run_graph", _run_graph)

    payload = {
        "question": "Welche Richtlinien gelten?",
        "top_k": 5,
        "hybrid": {},
        "collection_id": str(uuid.uuid4()),
    }

    response = client.post(
        "/v1/ai/rag/query/",
        data=payload,
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-topk",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert router.calls and router.calls[-1] == 5
    retrieval = body["retrieval"]
    assert retrieval["top_k_effective"] == 5
    assert len(body["snippets"]) <= 5


@pytest.mark.django_db
def test_rag_query_endpoint_surfaces_diagnostics(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            return None

    monkeypatch.setattr(views, "CHECKPOINTER", DummyCheckpointer())

    retrieval_payload = {
        "alpha": 0.65,
        "min_sim": 0.35,
        "top_k_effective": 2,
        "matches_returned": 1,
        "max_candidates_effective": 20,
        "vector_candidates": 11,
        "lexical_candidates": 7,
        "deleted_matches_blocked": 1,
        "visibility_effective": "active",
        "took_ms": 21,
        "routing": {
            "profile": "standard",
            "vector_space_id": "rag/global",
            "mode": "fallback",
        },
        "lexical_variant": "fallback",
    }
    snippets_payload = [
        {
            "id": "doc-3",
            "text": "Diagnostic snippet",
            "score": 0.73,
            "source": "guide.md",
        }
    ]

    def _run(state, meta):
        augmented = dict(state)
        augmented.update(
            {
                "snippets": snippets_payload,
                "matches": snippets_payload,
                "retrieval": retrieval_payload,
                "answer": "Synthesised",
            }
        )
        return augmented, {
            "answer": "Synthesised",
            "prompt_version": "v3",
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
            "graph_debug": {"lexical_primary_failed": "tuple index out of range"},
        }

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": "Was gilt?",
            "hybrid": {"alpha": 0.5},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-diagnostics",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Synthesised"
    assert payload["prompt_version"] == "v3"
    assert payload["retrieval"]["lexical_candidates"] == 7
    assert payload["retrieval"]["matches_returned"] == 1
    assert payload["retrieval"]["routing"]["profile"] == "standard"
    assert "lexical_variant" not in payload["retrieval"]
    diagnostics = payload.get("diagnostics")
    assert diagnostics
    assert (
        diagnostics["response"]["graph_debug"]["lexical_primary_failed"]
        == "tuple index out of range"
    )
    retrieval_diag = diagnostics["retrieval"]
    assert retrieval_diag["lexical_variant"] == "fallback"
    assert retrieval_diag["routing"]["mode"] == "fallback"


def test_normalise_rag_response_stringifies_uuid_values():
    routing_space = uuid.uuid4()
    snippet_id = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    trace_uuid = uuid.uuid4()
    extra_meta_key = uuid.uuid4()
    retrieval_extra_uuid = uuid.uuid4()
    match_uuid = uuid.uuid4()

    payload = {
        "answer": "Done",
        "prompt_version": "v5",
        "retrieval": {
            "alpha": 0.5,
            "min_sim": 0.15,
            "top_k_effective": 3,
            "matches_returned": 1,
            "max_candidates_effective": 25,
            "vector_candidates": 7,
            "lexical_candidates": 2,
            "deleted_matches_blocked": 0,
            "visibility_effective": "active",
            "took_ms": 12,
            "routing": {
                "profile": "standard",
                "vector_space_id": routing_space,
            },
            "internal_debug": {"id": retrieval_extra_uuid},
        },
        "snippets": [
            {
                "id": snippet_id,
                "text": "Snippet text",
                "score": 0.42,
                "source": "doc.md",
                "meta": {
                    "doc_uuid": doc_uuid,
                    extra_meta_key: {"nested": {"inner_uuid": uuid.uuid4()}},
                },
            }
        ],
        "graph_debug": {"trace_id": trace_uuid},
        "matches": [{"meta": {"chunk_id": match_uuid}}],
    }

    normalised = views._normalise_rag_response(payload)

    routing = normalised["retrieval"]["routing"]
    assert isinstance(routing["vector_space_id"], str)

    retrieval_diagnostics = normalised["diagnostics"]["retrieval"]
    assert retrieval_diagnostics["internal_debug"]["id"] == str(retrieval_extra_uuid)

    snippet = normalised["snippets"][0]
    assert isinstance(snippet["id"], str)
    assert isinstance(snippet["meta"]["doc_uuid"], str)
    assert isinstance(snippet["meta"][str(extra_meta_key)]["nested"]["inner_uuid"], str)

    diagnostics = normalised["diagnostics"]["response"]
    assert isinstance(diagnostics["graph_debug"]["trace_id"], str)
    assert diagnostics["matches"][0]["meta"]["chunk_id"] == str(match_uuid)


@pytest.mark.django_db
def test_rag_query_endpoint_populates_query_from_question(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    recorded: dict[str, object] = {}

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            state_copy = dict(state)
            recorded["saved_state"] = state_copy

    dummy_checkpointer = DummyCheckpointer()
    monkeypatch.setattr(views, "CHECKPOINTER", dummy_checkpointer)

    def _run(state, meta):
        params = RetrieveInput.from_state(state)
        recorded["params"] = params
        recorded["state"] = dict(state)
        retrieval_payload = {
            "alpha": 0.5,
            "min_sim": 0.1,
            "top_k_effective": 1,
            "matches_returned": 1,
            "max_candidates_effective": 25,
            "vector_candidates": 12,
            "lexical_candidates": 18,
            "deleted_matches_blocked": 0,
            "visibility_effective": "tenant",
            "took_ms": 30,
            "routing": {
                "profile": "standard",
                "vector_space_id": "rag/global",
            },
        }
        snippets_payload = [
            {
                "id": "doc-99",
                "text": "Snippet",
                "score": 0.73,
                "source": "faq.md",
            }
        ]
        augmented_state = dict(state)
        augmented_state.update(
            {
                "snippets": snippets_payload,
                "matches": snippets_payload,
                "retrieval": retrieval_payload,
                "answer": "Ready",
            }
        )
        recorded["final_state"] = dict(augmented_state)
        return augmented_state, {
            "answer": "Ready",
            "prompt_version": "v1",
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
        }

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    question_text = "Was ist RAG?"
    hybrid_config = {"alpha": 0.5}
    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "question": question_text,
            "hybrid": hybrid_config,
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-002",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Ready"
    assert payload["prompt_version"] == "v1"
    retrieval = payload["retrieval"]
    assert retrieval["alpha"] == 0.5
    assert retrieval["min_sim"] == 0.1
    assert retrieval["top_k_effective"] == 1
    assert retrieval["visibility_effective"] == "tenant"
    assert retrieval["took_ms"] == 30
    assert retrieval["routing"]["profile"] == "standard"
    assert retrieval["routing"]["vector_space_id"] == "rag/global"
    snippet = payload["snippets"][0]
    assert isinstance(snippet["score"], float)
    assert snippet["text"]
    assert snippet["source"]

    params = recorded["params"]
    assert isinstance(params, RetrieveInput)
    assert params.query == question_text
    assert params.hybrid == hybrid_config

    state = recorded["state"]
    assert state["question"] == question_text
    assert state["query"] == question_text
    assert state["hybrid"] == hybrid_config

    assert recorded["saved_state"] == recorded["final_state"]


@pytest.mark.django_db
def test_rag_query_endpoint_returns_not_found_when_no_matches(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            raise AssertionError("save should not be called on not-found")

    dummy_checkpointer = DummyCheckpointer()
    monkeypatch.setattr(views, "CHECKPOINTER", dummy_checkpointer)

    def _run(state, meta):
        raise NotFoundError("No matching documents were found for the query.")

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "query": "no hits",
            "hybrid": {"alpha": 0.5},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-404",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No matching documents were found for the query.",
        "code": "rag_no_matches",
    }


@pytest.mark.django_db
def test_rag_query_endpoint_returns_422_on_inconsistent_metadata(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)

    class DummyCheckpointer:
        def load(self, ctx):
            return {}

        def save(self, ctx, state):
            raise AssertionError(
                "save should not be called when metadata is inconsistent"
            )

    dummy_checkpointer = DummyCheckpointer()
    monkeypatch.setattr(views, "CHECKPOINTER", dummy_checkpointer)

    def _run(state, meta):
        raise InconsistentMetadataError("reindex required")

    graph_runner = SimpleNamespace(run=_run)
    monkeypatch.setattr(views, "get_graph_runner", lambda name: graph_runner)

    response = client.post(
        "/v1/ai/rag/query/",
        data={
            "query": "bad meta",
            "hybrid": {"alpha": 0.5},
            "collection_id": str(uuid.uuid4()),
        },
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-rag-422",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "retrieval_inconsistent_metadata"
    assert "reindex required" in payload["detail"]


def test_build_crawler_state_provides_guardrail_inputs(monkeypatch):
    def _fail_enforce(*args, **kwargs):  # pragma: no cover - safety guard
        raise AssertionError("guardrails enforcement should be delegated to the graph")

    monkeypatch.setattr(views.guardrails_middleware, "enforce_guardrails", _fail_enforce)

    request = CrawlerRunRequest.model_validate(
        {
            "mode": "manual",
            "workflow_id": "wf-manual",
            "origins": [
                {
                    "url": "https://example.org/doc",
                    "content": "hello world",
                    "content_type": "text/plain",
                    "fetch": False,
                }
            ],
        }
    )
    meta = {"tenant_id": "tenant-guard", "case_id": "case-guard"}

    builds = views._build_crawler_state(meta, request)
    assert len(builds) == 1
    guardrails = builds[0].state.get("guardrails")
    assert isinstance(guardrails, dict)
    limits = guardrails.get("limits")
    signals = guardrails.get("signals")
    assert isinstance(limits, GuardrailLimits)
    assert limits.max_document_bytes is None
    assert isinstance(signals, GuardrailSignals)
    assert signals.canonical_source == "https://example.org/doc"
    assert signals.host == "example.org"
    assert signals.document_bytes == len("hello world".encode("utf-8"))
    assert guardrails.get("error_builder") is views._build_guardrail_error
    gating_input = builds[0].state.get("gating_input", {})
    assert gating_input.get("limits") is limits
    assert gating_input.get("signals") is signals


@pytest.mark.django_db
def test_crawler_runner_manual_multi_origin(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    ingestion_calls: list[tuple[dict, dict, str | None]] = []

    def _fake_start_ingestion(request_data, meta, idempotency_key=None):
        ingestion_calls.append((request_data, meta, idempotency_key))
        document_id = request_data["document_ids"][0]
        return SimpleNamespace(
            data={"ingestion_run_id": f"ingest-{document_id}", "status": "queued"}
        )

    monkeypatch.setattr(services, "start_ingestion_run", _fake_start_ingestion)

    class _DummyDecision:
        def __init__(self, document_id: str):
            self.payload = SimpleNamespace(document_id=document_id)

    class _DummyGraph:
        def __init__(self) -> None:
            self.upsert_handler = None

        def start_crawl(self, state):
            cloned = dict(state)
            control = dict(cloned.get("control", {}))
            control.setdefault("shadow_mode", True)
            cloned["control"] = control
            cloned["normalize_input"] = {
                "document_id": cloned.get("document_id"),
                "tags": control.get("tags", []),
            }
            cloned["transitions"] = {
                "crawler.fetch": {"decision": "skipped"},
                "crawler.ingest_decision": {"decision": "upsert"},
            }
            cloned["ingest_action"] = "upsert"
            cloned["gating_score"] = 0.95
            cloned["content_hash"] = f"hash-{cloned['document_id']}"
            return cloned

        def run(self, state, meta):
            result_state = dict(state)
            if self.upsert_handler is not None:
                artifacts = result_state.setdefault("artifacts", {})
                artifacts["upsert_result"] = self.upsert_handler(
                    _DummyDecision(state.get("document_id"))
                )
            result = {
                "graph_run_id": f"run-{state.get('document_id')}",
                "decision": "upsert",
            }
            return result_state, result

    monkeypatch.setattr(
        views,
        "crawler_ingestion_graph",
        SimpleNamespace(build_graph=lambda: _DummyGraph()),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-crawl",
    }

    payload = {
        "mode": "manual",
        "workflow_id": "crawler-demo",
        "collection_id": "6d6fba7c-1c62-4f0a-8b8b-7efb4567a0aa",
        "tags": ["handbook", "hr"],
        "snapshot": {"enabled": True, "label": "debug"},
        "review": "required",
        "dry_run": True,
        "origins": [
            {
                "url": "https://example.com/docs/handbook",
                "content": "first origin",
                "content_type": "text/plain",
                "fetch": False,
            },
            {
                "url": "https://example.com/docs/policies",
                "content": "second origin",
                "content_type": "text/plain",
                "fetch": False,
                "tags": ["policy"],
            },
        ],
    }

    first = client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["workflow_id"] == "crawler-demo"
    assert first_payload["mode"] == "manual"
    assert len(first_payload["origins"]) == 2
    assert len(first_payload["transitions"]) == 2
    assert len(first_payload["telemetry"]) == 2
    assert first_payload["errors"] == []
    assert first_payload["idempotent"] is False
    assert ingestion_calls
    assert {entry["origin"] for entry in first_payload["origins"]} == {
        "https://example.com/docs/handbook",
        "https://example.com/docs/policies",
    }

    second = client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert second.status_code == 200
    assert second.json()["idempotent"] is True


@pytest.mark.django_db
def test_crawler_runner_propagates_idempotency_key(
    client, monkeypatch, tmp_path, test_tenant_schema_name
):
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    recorded_keys: list[str | None] = []

    def _start_ingestion(request_data, meta, idempotency_key=None):  # type: ignore[no-untyped-def]
        recorded_keys.append(idempotency_key)
        document_id = request_data["document_ids"][0]
        return SimpleNamespace(
            data={"status": "queued", "ingestion_run_id": document_id}
        )

    monkeypatch.setattr(services, "start_ingestion_run", _start_ingestion)

    class _Graph:
        def __init__(self) -> None:
            self.upsert_handler = None

        def start_crawl(self, state):  # type: ignore[no-untyped-def]
            updated = dict(state)
            updated.setdefault("control", {})
            updated["normalize_input"] = {"document_id": updated.get("document_id")}
            updated["transitions"] = {"crawler.ingest_decision": {"decision": "upsert"}}
            updated["ingest_action"] = "upsert"
            updated["gating_score"] = 1.0
            return updated

        def run(self, state, meta):  # type: ignore[no-untyped-def]
            artifacts = state.setdefault("artifacts", {})
            if self.upsert_handler is not None:
                artifacts["upsert_result"] = self.upsert_handler(
                    SimpleNamespace(attributes={})
                )
            return state, {"graph_run_id": "run", "decision": "upsert"}

    monkeypatch.setattr(
        views,
        "crawler_ingestion_graph",
        SimpleNamespace(build_graph=lambda: _Graph()),
    )

    payload = {
        "mode": "manual",
        "workflow_id": "crawler-propagate",
        "origins": [
            {
                "url": "https://example.org/manual",
                "content": "manual payload",
                "content_type": "text/plain",
                "fetch": False,
            }
        ],
    }

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-id",
        IDEMPOTENCY_KEY_HEADER: "idem-crawler-1",
    }

    response = client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    assert response[IDEMPOTENCY_KEY_HEADER] == "idem-crawler-1"
    body = response.json()
    assert body["idempotent"] is True
    assert recorded_keys == ["idem-crawler-1"]
