import pytest

from cases.lifecycle import (
    apply_lifecycle_definition,
    update_case_from_collection_search,
)
from cases.models import Case, CaseEvent
from customers.models import Tenant


@pytest.mark.django_db
def test_update_case_from_collection_search_creates_events(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    from django_tenants.utils import tenant_context

    with tenant_context(tenant):
        Case.objects.create(tenant=tenant, external_id="case-lifecycle")

    from ai_core.contracts import BusinessContext, ScopeContext

    tool_context = ScopeContext(
        tenant_id=tenant.schema_name,
        trace_id="trace-42",
        invocation_id="invoke-1",
        run_id="run-1",
    ).to_tool_context(
        business=BusinessContext(
            workflow_id="wf-123",
            collection_id="legal-news",
        )
    )

    state = {
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        "transitions": [
            {"node": "strategy_generated", "decision": "ok"},
            {"node": "ingest_triggered", "decision": "triggered"},
            {"node": "hitl_pending", "decision": "waiting"},
        ],
    }

    result = update_case_from_collection_search(
        tenant.schema_name, "case-lifecycle", state
    )

    case = Case.objects.get(external_id="case-lifecycle", tenant=tenant)
    assert case.phase == "external_review"
    assert result is not None
    assert result.case == case
    assert result.event_types[-1] == "collection_search:hitl_pending"
    assert result.collection_scope == "legal-news"

    events = list(case.events.order_by("created_at"))
    assert [event.event_type for event in events] == [
        "collection_search:strategy_generated",
        "collection_search:ingest_triggered",
        "collection_search:hitl_pending",
    ]
    assert {event.graph_name for event in events} == {"collection_search"}
    assert {event.collection_id for event in events} == {"legal-news"}
    assert events[-1].workflow_id == "wf-123"
    assert events[-1].trace_id == "trace-42"


@pytest.mark.django_db
def test_update_case_from_collection_search_sets_completed_phase(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    from django_tenants.utils import tenant_context

    with tenant_context(tenant):
        Case.objects.create(tenant=tenant, external_id="case-phase")

    from ai_core.contracts import BusinessContext, ScopeContext

    tool_context = ScopeContext(
        tenant_id=tenant.schema_name,
        trace_id="trace-77",
        invocation_id="invoke-2",
        run_id="run-2",
    ).to_tool_context(business=BusinessContext(workflow_id="wf-321"))

    state = {
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        "input": {"collection_scope": "fin-law"},
        "transitions": [
            {"node": "ingest_triggered", "decision": "triggered"},
            {"node": "verified", "decision": "complete"},
        ],
    }

    result = update_case_from_collection_search(tenant.schema_name, "case-phase", state)

    case = Case.objects.get(external_id="case-phase", tenant=tenant)
    assert case.phase == "search_completed"
    assert case.events.filter(event_type="collection_search:verified").exists()
    assert result is not None


@pytest.mark.django_db
def test_apply_lifecycle_definition_uses_tenant_definition(
    test_tenant_schema_name,
):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    tenant.case_lifecycle_definition = {
        "phases": ["anzeige", "verhandlung"],
        "transitions": [
            {
                "from_phase": None,
                "to_phase": "anzeige",
                "trigger_events": ["ingestion_run_queued"],
            },
            {
                "from_phase": "anzeige",
                "to_phase": "verhandlung",
                "trigger_events": ["collection_search:verified"],
            },
        ],
    }
    tenant.save()

    case = Case.objects.create(tenant=tenant, external_id="custom-lifecycle")
    CaseEvent.objects.create(
        case=case,
        tenant=tenant,
        event_type="ingestion_run_queued",
    )
    CaseEvent.objects.create(
        case=case,
        tenant=tenant,
        event_type="collection_search:verified",
    )

    apply_lifecycle_definition(
        case,
        list(case.events.order_by("created_at")),
    )

    case.refresh_from_db()
    assert case.phase == "verhandlung"
