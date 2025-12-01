import pytest

from ai_core import case_events
from ai_core.case_events import emit_ingestion_case_event
from cases.models import CaseEvent
from customers.models import Tenant
from documents.models import DocumentIngestionRun


@pytest.mark.django_db
def test_emit_ingestion_case_event_records_latest_run(
    test_tenant_schema_name, monkeypatch
):
    from cases.models import Case
    from django_tenants.utils import tenant_context

    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    with tenant_context(tenant):
        Case.objects.create(tenant=tenant, external_id="case-hook")

    DocumentIngestionRun.objects.create(
        tenant_id=tenant.schema_name,
        case="case-hook",
        run_id="run-hook",
        status="running",
        queued_at="2024-02-01T00:00:00Z",
    )

    captured: list[dict[str, object]] = []

    def fake_emit_event(name, payload=None):  # type: ignore[no-untyped-def]
        captured.append({"name": name, "payload": payload})

    monkeypatch.setattr(case_events, "emit_event", fake_emit_event)

    emit_ingestion_case_event(
        tenant.schema_name, "case-hook", run_id="run-hook", context="running"
    )

    event = CaseEvent.objects.get(case__external_id="case-hook")
    assert event.event_type == "ingestion_run_started"
    assert event.payload["run_id"] == "run-hook"
    assert captured[0]["name"] == "case.lifecycle.ingestion"
    assert captured[0]["payload"]["case_id"] == "case-hook"
