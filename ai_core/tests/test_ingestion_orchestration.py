"""Tests for ingestion orchestration helpers."""

from types import SimpleNamespace

from ai_core.ingestion_orchestration import (
    IngestionContext,
    IngestionContextBuilder,
    ObservabilityWrapper,
)


def test_ingestion_context_builder_prioritizes_trace_values():
    builder = IngestionContextBuilder()
    state = {
        "tenant_id": "state-tenant",
        "case_id": "state-case",
        "workflow_id": "state-workflow",
        "source": "state-source",
        "raw_payload_path": "state-path",
        "raw_document": {
            "workflow_id": "raw-workflow",
            "payload_path": "raw-payload",
            "metadata": {"workflow_id": "raw-meta"},
        },
    }
    meta = {"workflow_id": "meta-workflow", "source": "meta-source"}
    trace_context = {
        "tenant_id": "trace-tenant",
        "case_id": "trace-case",
        "workflow_id": "trace-workflow",
        "document_id": "trace-doc",
        "trace_id": "trace-id",
    }

    ctx = builder.build_from_state(state, meta, trace_context)

    assert ctx.tenant_id == "trace-tenant"
    assert ctx.case_id == "trace-case"
    assert ctx.workflow_id == "trace-workflow"
    assert ctx.source == "meta-source"
    assert ctx.document_id == "trace-doc"
    assert ctx.raw_payload_path == "state-path"


def test_observability_wrapper_honors_task_and_metadata():
    class HelpersSpy:
        def __init__(self):
            self.started = []
            self.ended = 0
            self.updated = []

        def start_trace(self, name, user_id, session_id, metadata, trace_id=None):
            self.started.append(
                {
                    "name": name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "metadata": metadata,
                    "trace_id": trace_id,
                }
            )

        def end_trace(self):
            self.ended += 1

        def update_observation(self, metadata):
            self.updated.append(metadata)

    helpers = HelpersSpy()
    wrapper = ObservabilityWrapper(helpers)
    ingestion_ctx = IngestionContext(
        tenant_id="tenant-x",
        case_id="case-x",
        workflow_id=None,
        trace_id=None,
        collection_id=None,
        source=None,
        document_id="doc-x",
        run_id=None,
        ingestion_run_id=None,
        raw_payload_path=None,
    )
    trace_context = {"trace_id": "trace-123"}
    task_request = SimpleNamespace(id="celery-1")

    obs_ctx = wrapper.create_context(ingestion_ctx, trace_context, task_request)

    assert obs_ctx.metadata["trace_id"] == "trace-123"
    assert obs_ctx.metadata["tenant_id"] == "tenant-x"
    assert obs_ctx.metadata["case_id"] == "case-x"
    assert obs_ctx.task_identifier == "celery-1"

    wrapper.start_trace(obs_ctx)
    assert helpers.started[0]["metadata"] == obs_ctx.metadata
    assert helpers.updated == [{"celery.task_id": "celery-1"}]

    wrapper.end_trace()
    assert helpers.ended == 1
