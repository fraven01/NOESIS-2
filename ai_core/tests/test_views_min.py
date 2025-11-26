import json
import pytest

from ai_core.infra import rate_limit
from cases.services import get_or_create_case_for
from cases.models import Case
from customers.models import Tenant
from django_tenants.utils import tenant_context


@pytest.mark.django_db
def test_ping_ok(client, monkeypatch, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)

    get_or_create_case_for(tenant, "c1")
    resp = client.get(
        "/ai/ping/",
        HTTP_X_TENANT_ID=tenant.schema_name,
        HTTP_X_CASE_ID="c1",
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert resp["X-Trace-ID"]


@pytest.mark.django_db
def test_ping_missing_case(client, monkeypatch, settings, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    settings.AUTO_CREATE_CASES = False

    # Use the existing test tenant
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    resp = client.get(
        "/ai/ping/",
        HTTP_X_TENANT_ID=tenant.schema_name,
        HTTP_X_CASE_ID="missing",
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "case_not_found"


@pytest.mark.django_db
def test_rag_query_missing_headers(client):
    resp = client.post(
        "/v1/ai/rag/query/",
        data=json.dumps({"question": "Ping?"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.django_db
def test_ingestion_status_includes_case_details(client, monkeypatch, test_tenant_schema_name):
    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    with tenant_context(tenant):
        case = Case.objects.create(tenant=tenant, external_id="case-status", phase="review")
        case.events.create(
            tenant=tenant,
            event_type="ingestion_run_completed",
            source="ingestion",
            payload={"status": "succeeded"},
        )

    monkeypatch.setattr(
        "ai_core.views.get_latest_ingestion_run",
        lambda tenant_id, case_id: {
            "run_id": "run-latest",
            "status": "succeeded",
            "document_ids": [],
            "invalid_document_ids": [],
        },
    )

    resp = client.get(
        "/ai/rag/ingestion/status/",
        HTTP_X_TENANT_ID=tenant.schema_name,
        HTTP_X_CASE_ID="case-status",
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["case_status"] == case.status
    assert payload["case_phase"] == "review"
    assert payload["latest_case_event"]["event_type"] == "ingestion_run_completed"
    assert payload["latest_case_event"]["payload"]["status"] == "succeeded"
