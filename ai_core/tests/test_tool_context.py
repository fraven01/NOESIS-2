from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


@pytest.fixture
def base_payload() -> dict:
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-1",
        invocation_id=str(uuid4()),
        run_id="run-1",
        timestamp=datetime.now(timezone.utc),
    )
    business = BusinessContext()
    return {"scope": scope, "business": business}


def test_tool_context_both_run_ids_allowed(base_payload: dict) -> None:
    """Both run_id and ingestion_run_id can co-exist (Pre-MVP ID Contract)."""
    scope = base_payload["scope"].model_copy(
        update={"run_id": "run-1", "ingestion_run_id": "ingestion-run-1"}
    )
    context = ToolContext(scope=scope, business=base_payload["business"])
    assert context.run_id == "run-1"
    assert context.ingestion_run_id == "ingestion-run-1"


def test_tool_context_no_run_ids_fails(base_payload: dict) -> None:
    with pytest.raises(ValueError):
        scope = base_payload["scope"].model_copy(
            update={"run_id": None, "ingestion_run_id": None}
        )
        ToolContext(scope=scope, business=base_payload["business"])


def test_tool_context_with_run_id_passes(base_payload: dict) -> None:
    scope = base_payload["scope"].model_copy(update={"run_id": "run-1"})
    context = ToolContext(scope=scope, business=base_payload["business"])
    assert context.run_id == "run-1"
    assert context.ingestion_run_id is None


def test_tool_context_with_ingestion_run_id_passes(base_payload: dict) -> None:
    scope = base_payload["scope"].model_copy(
        update={"run_id": None, "ingestion_run_id": "ingestion-run-1"}
    )
    context = ToolContext(scope=scope, business=base_payload["business"])
    assert context.ingestion_run_id == "ingestion-run-1"
    assert context.run_id is None
