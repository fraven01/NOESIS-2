import pytest

from cases.services import record_ingestion_case_event
from customers.models import Tenant
from documents.models import DocumentIngestionRun


@pytest.mark.django_db
def test_record_ingestion_case_event_creates_event(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    from cases.models import Case
    from django_tenants.utils import tenant_context

    with tenant_context(tenant):
        Case.objects.create(tenant=tenant, external_id="case-event")
    run = DocumentIngestionRun.objects.create(
        tenant_id=tenant.schema_name,
        case="case-event",
        run_id="run-1",
        status="succeeded",
        queued_at="2024-01-01T00:00:00Z",
        inserted_documents=2,
    )

    event = record_ingestion_case_event(tenant, "case-event", run)

    assert event.event_type == "ingestion_run_completed"
    assert event.case.external_id == "case-event"
    assert event.payload["run_id"] == "run-1"
    assert event.payload["status"] == "succeeded"
    event.case.refresh_from_db()
    assert event.case.phase == "evidence_collection"
