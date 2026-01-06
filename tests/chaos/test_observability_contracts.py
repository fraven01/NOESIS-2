"""Chaos tests for observability contract compliance.

Tests ScopeContext/BusinessContext tag propagation to Langfuse spans, ensuring
proper metadata is attached for tracing, debugging, and operational insights.

Contract under test:
- ai_core/infra/observability.py: record_span, emit_event
- ai_core/contracts/scope.py: ScopeContext propagation
- ai_core/contracts/business.py: BusinessContext propagation
"""

from __future__ import annotations

import pytest

import ai_core.infra.observability as observability
from tests.chaos.conftest import _build_chaos_meta

pytestmark = pytest.mark.chaos


def test_scope_context_tags_in_span_metadata(langfuse_mock, chaos_env):
    """ScopeContext fields propagated as Langfuse span metadata.

    Verifies that infrastructure IDs (tenant_id, trace_id, invocation_id, run_id)
    are attached to spans for correlation and debugging.
    """
    meta = _build_chaos_meta(
        tenant_id="obs-tenant",
        trace_id="obs-trace",
        invocation_id="obs-invocation",
        run_id="obs-run",
        case_id="obs-case",
    )

    scope_context = meta["scope_context"]

    # Emit a span with scope context
    observability.record_span(
        "test.operation",
        trace_id=scope_context["trace_id"],
        attributes={
            "tenant_id": scope_context["tenant_id"],
            "invocation_id": scope_context["invocation_id"],
            "run_id": scope_context["run_id"],
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    # Verify trace_id at span level
    assert span.trace_id == "obs-trace"

    # Verify scope context fields in metadata
    assert span.metadata.get("tenant_id") == "obs-tenant"
    assert span.metadata.get("invocation_id") == "obs-invocation"
    assert span.metadata.get("run_id") == "obs-run"


def test_business_context_tags_in_span_metadata(langfuse_mock, chaos_env):
    """BusinessContext fields propagated as Langfuse span metadata.

    Verifies that business domain IDs (case_id, collection_id, workflow_id)
    are attached to spans for business-level filtering and analytics.
    """
    meta = _build_chaos_meta(
        tenant_id="obs-biz-tenant",
        trace_id="obs-biz-trace",
        case_id="obs-biz-case",
        collection_id="obs-biz-collection",
        workflow_id="obs-biz-workflow",
    )

    business_context = meta["business_context"]

    # Emit a span with business context
    observability.record_span(
        "test.business.operation",
        trace_id=meta["scope_context"]["trace_id"],
        attributes={
            "case_id": business_context.get("case_id"),
            "collection_id": business_context.get("collection_id"),
            "workflow_id": business_context.get("workflow_id"),
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    # Verify business context fields in metadata
    assert span.metadata.get("case_id") == "obs-biz-case"
    assert span.metadata.get("collection_id") == "obs-biz-collection"
    assert span.metadata.get("workflow_id") == "obs-biz-workflow"


def test_combined_scope_and_business_context_in_span(langfuse_mock, chaos_env):
    """Both scope and business context fields present in span metadata."""
    meta = _build_chaos_meta(
        tenant_id="obs-combined-tenant",
        trace_id="obs-combined-trace",
        run_id="obs-combined-run",
        case_id="obs-combined-case",
        collection_id="obs-combined-collection",
    )

    scope_context = meta["scope_context"]
    business_context = meta["business_context"]

    # Emit a span with both contexts
    observability.record_span(
        "test.combined.operation",
        trace_id=scope_context["trace_id"],
        attributes={
            # Scope fields
            "tenant_id": scope_context["tenant_id"],
            "run_id": scope_context["run_id"],
            # Business fields
            "case_id": business_context.get("case_id"),
            "collection_id": business_context.get("collection_id"),
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    # Verify scope context
    assert span.metadata.get("tenant_id") == "obs-combined-tenant"
    assert span.metadata.get("run_id") == "obs-combined-run"

    # Verify business context
    assert span.metadata.get("case_id") == "obs-combined-case"
    assert span.metadata.get("collection_id") == "obs-combined-collection"


def test_event_emission_with_context_fields(langfuse_mock, chaos_env):
    """Observability events include scope/business context fields."""
    meta = _build_chaos_meta(
        tenant_id="obs-event-tenant",
        trace_id="obs-event-trace",
        case_id="obs-event-case",
    )

    scope_context = meta["scope_context"]
    business_context = meta["business_context"]

    # Emit an event with context
    observability.emit_event(
        {
            "event": "test.event",
            "tenant": scope_context["tenant_id"],
            "trace_id": scope_context["trace_id"],
            "case_id": business_context.get("case_id"),
            "message": "Test event with context",
        }
    )

    assert len(langfuse_mock.events) == 1
    event = langfuse_mock.events[0]

    assert event.get("event") == "test.event"
    assert event.get("tenant") == "obs-event-tenant"
    assert event.get("trace_id") == "obs-event-trace"
    assert event.get("case_id") == "obs-event-case"


def test_span_service_id_for_s2s_hops(langfuse_mock, chaos_env):
    """S2S Hop spans include service_id for identity tracking."""
    meta = _build_chaos_meta(
        tenant_id="obs-s2s-tenant",
        trace_id="obs-s2s-trace",
        service_id="chaos-test-service",  # S2S identity
        run_id="obs-s2s-run",
    )

    scope_context = meta["scope_context"]

    # Emit a span with service_id
    observability.record_span(
        "test.s2s.operation",
        trace_id=scope_context["trace_id"],
        attributes={
            "tenant_id": scope_context["tenant_id"],
            "service_id": scope_context.get("service_id"),
            "run_id": scope_context["run_id"],
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    assert span.metadata.get("service_id") == "chaos-test-service"
    assert "user_id" not in span.metadata  # S2S Hop, no user_id


def test_span_without_optional_business_context(langfuse_mock, chaos_env):
    """Spans work correctly with scope context only (no business context)."""
    meta = _build_chaos_meta(
        tenant_id="obs-minimal-tenant",
        trace_id="obs-minimal-trace",
        run_id="obs-minimal-run",
        # No business context fields
    )

    scope_context = meta["scope_context"]

    # Emit a span with scope only
    observability.record_span(
        "test.minimal.operation",
        trace_id=scope_context["trace_id"],
        attributes={
            "tenant_id": scope_context["tenant_id"],
            "run_id": scope_context["run_id"],
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    # Scope context present
    assert span.metadata.get("tenant_id") == "obs-minimal-tenant"
    assert span.metadata.get("run_id") == "obs-minimal-run"

    # Business context fields absent (as expected)
    assert "case_id" not in span.metadata
    assert "collection_id" not in span.metadata


def test_multiple_spans_share_trace_id(langfuse_mock, chaos_env):
    """Multiple operations in the same trace share trace_id for correlation."""
    meta = _build_chaos_meta(
        tenant_id="obs-multi-tenant",
        trace_id="obs-multi-trace",
        run_id="obs-multi-run",
        case_id="obs-multi-case",
    )

    trace_id = meta["scope_context"]["trace_id"]

    # Emit multiple spans with same trace_id
    for i in range(3):
        observability.record_span(
            f"test.operation.{i}",
            trace_id=trace_id,
            attributes={
                "tenant_id": meta["scope_context"]["tenant_id"],
                "step": i,
            },
        )

    assert len(langfuse_mock.spans) == 3

    # All spans share the same trace_id
    for span in langfuse_mock.spans:
        assert span.trace_id == "obs-multi-trace"


def test_span_with_ingestion_run_id(langfuse_mock, chaos_env):
    """Ingestion spans include ingestion_run_id for ingestion-specific tracking."""
    meta = _build_chaos_meta(
        tenant_id="obs-ingestion-tenant",
        trace_id="obs-ingestion-trace",
        ingestion_run_id="obs-ingestion-run-123",  # Ingestion-specific ID
        collection_id="obs-ingestion-collection",
    )

    scope_context = meta["scope_context"]

    # Emit a span with ingestion_run_id
    observability.record_span(
        "test.ingestion.operation",
        trace_id=scope_context["trace_id"],
        attributes={
            "tenant_id": scope_context["tenant_id"],
            "ingestion_run_id": scope_context.get("ingestion_run_id"),
        },
    )

    assert len(langfuse_mock.spans) == 1
    span = langfuse_mock.spans[0]

    assert span.metadata.get("ingestion_run_id") == "obs-ingestion-run-123"


@pytest.mark.skip(reason="Graph I/O spec metadata requires graph execution mocking")
def test_graph_io_spec_in_span_metadata():
    """Graph execution spans include io_spec metadata (schema_id, schema_version).

    NOTE: This test is skipped as it requires:
    - Mocking LangGraph execution
    - Capturing graph transition spans
    - Verifying io_spec propagation from graph input/output models

    Future implementation should:
    1. Execute UniversalIngestionGraph with langfuse_mock
    2. Verify spans include:
       - schema_id: "noesis.graphs.universal_ingestion"
       - schema_version: "1.0.0"
       - input_model/output_model metadata
    """
    pass
