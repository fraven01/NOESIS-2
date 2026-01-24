"""Tests for query planner."""

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.rag.query_planner import plan_query
from ai_core.rag.filter_spec import FilterSpec
from ai_core.tool_contracts import tool_context_from_scope


def _context():
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invoke",
        run_id="run",
    )
    business = BusinessContext(collection_id="col-1")
    return tool_context_from_scope(scope, business)


def test_plan_query_uses_doc_class():
    plan = plan_query(
        "data retention",
        context=_context(),
        doc_class="policy",
        filters=None,
    )
    assert plan.doc_type == "policy"
    assert plan.queries[0] == "data retention"
    assert any("policy" in q for q in plan.queries)


def test_plan_query_constraints_from_filters():
    plan = plan_query(
        "incident response",
        context=_context(),
        doc_class=None,
        filters=FilterSpec(
            tenant_id="tenant",
            collection_id="col-2",
            metadata={
                "must_include": ["breach", "timeline"],
                "date_from": "2024-01-01",
                "date_to": "2024-12-31",
            },
        ),
    )
    constraints = plan.constraints
    assert "col-2" in constraints.collections
    assert "breach" in constraints.must_include
    assert constraints.date_from == "2024-01-01"
