from __future__ import annotations

from uuid import uuid4

import pytest

from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.agent.scope_policy import PolicyViolation
from ai_core.contracts.scope import ScopeContext
from documents.repository import DocumentLifecycleStore, InMemoryDocumentsRepository


def _scope_context() -> ScopeContext:
    return ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )


def test_system_scope_blocks_all_instrumented_mutations() -> None:
    runtime_config = RuntimeConfig(execution_scope="SYSTEM")
    scope = _scope_context()
    repo = InMemoryDocumentsRepository()
    store = DocumentLifecycleStore()

    with pytest.raises(PolicyViolation):
        repo.delete(
            "tenant",
            uuid4(),
            workflow_id="wf",
            runtime_config=runtime_config,
            scope=scope,
        )

    with pytest.raises(PolicyViolation):
        repo.delete_asset(
            "tenant",
            uuid4(),
            workflow_id="wf",
            runtime_config=runtime_config,
            scope=scope,
        )

    with pytest.raises(PolicyViolation):
        store.record_document_state(
            tenant_id="tenant",
            document_id=uuid4(),
            workflow_id="wf",
            state="active",
            runtime_config=runtime_config,
            scope=scope,
        )

    with pytest.raises(PolicyViolation):
        store.record_ingestion_run_queued(
            tenant_id="tenant",
            case="case-1",
            run_id="run-1",
            document_ids=(),
            invalid_document_ids=(),
            queued_at="2026-01-01T00:00:00Z",
            runtime_config=runtime_config,
            scope=scope,
        )

    with pytest.raises(PolicyViolation):
        store.mark_ingestion_run_completed(
            tenant_id="tenant",
            case="case-1",
            run_id="run-1",
            finished_at="2026-01-01T00:00:00Z",
            duration_ms=1.0,
            inserted_documents=0,
            replaced_documents=0,
            skipped_documents=0,
            inserted_chunks=0,
            invalid_document_ids=(),
            document_ids=(),
            error=None,
            runtime_config=runtime_config,
            scope=scope,
        )
