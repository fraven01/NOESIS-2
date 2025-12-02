import json
import uuid
from types import SimpleNamespace

from ai_core import views, services
from ai_core.infra import rate_limit
from ai_core.tests.doubles import MockTenantContext, MockTenant
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from django_tenants.middleware.main import TenantMainMiddleware


def test_ingestion_run_rejects_blank_document_ids(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr("customers.tenant_context.TenantContext", MockTenantContext)
    monkeypatch.setattr(
        TenantMainMiddleware,
        "get_tenant",
        lambda self, model, hostname: MockTenant(schema_name=test_tenant_schema_name),
    )
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    response = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps({"document_ids": ["  "], "embedding_profile": "standard"}),
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: "case-test",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "document_ids" in body["detail"]


def test_ingestion_run_rejects_empty_embedding_profile(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr("customers.tenant_context.TenantContext", MockTenantContext)
    monkeypatch.setattr(
        TenantMainMiddleware,
        "get_tenant",
        lambda self, model, hostname: MockTenant(schema_name=test_tenant_schema_name),
    )
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

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
            META_CASE_ID_KEY: "case-test",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "validation_error"
    assert "embedding_profile" in body["detail"]


def test_ingestion_run_normalises_payload_before_dispatch(
    client, monkeypatch, test_tenant_schema_name
):
    monkeypatch.setattr("customers.tenant_context.TenantContext", MockTenantContext)
    monkeypatch.setattr(
        TenantMainMiddleware,
        "get_tenant",
        lambda self, model, hostname: MockTenant(schema_name=test_tenant_schema_name),
    )
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        services, "emit_ingestion_case_event", lambda *args, **kwargs: None
    )

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
            META_CASE_ID_KEY: "case-test",
        },
    )

    assert response.status_code == 202
    assert captured["resolved_profile_input"] == "standard"
    assert captured["partition_args"][2] == [raw_id]
    assert dispatched["args"][2] == ["doc-trimmed"]
    assert dispatched["args"][3] == "standard"
    assert dispatched["kwargs"]["tenant_schema"] == test_tenant_schema_name
